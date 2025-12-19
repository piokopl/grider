import json
import asyncio
import math
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple, Tuple
from dataclasses import dataclass
import logging

from database import Database, Grid, GridLevel, GridOrder
from exchange import BitgetExchange, OrderResult
from trend import TrendDetector, TrendSignal


@dataclass
class GridConfig:
    pair: str
    enabled: bool
    leverage: int
    margin_mode: str
    
    grid_mode: str  # neutral / long / short
    total_capital: float
    capital_per_grid: float
    
    upper_price: float
    lower_price: float
    range_percent: float
    grid_levels: int
    grid_spacing: str  # arithmetic / geometric
    
    order_type: str  # limit / market
    min_profit_percent: float
    
    sl_percent: float
    tp_percent: float
    
    trailing_stop: bool
    trailing_stop_activation: float
    trailing_stop_callback: float
    
    trend_filter: bool
    trend_indicator: str
    trend_tf: str
    trend_params: Dict
    
    rebalance_enabled: bool
    rebalance_threshold: float
    
    check_interval: int
    cooldown_after_fill: float
    max_open_orders: int


@dataclass
class AutoAdjustConfig:
    enabled: bool
    rebuild_on_exit: bool
    exit_buffer_percent: float
    scheduled_check_hours: int
    rebuild_on_inefficiency: bool
    inefficiency_threshold: int
    min_grid_age_minutes: int
    volatility_lookback_hours: int
    volatility_multiplier: float
    min_range_percent: float
    max_range_percent: float
    close_position_on_rebuild: bool
    min_rebuild_interval_minutes: int


class GridManager:
    # Trading fee (0.02% per side - maker fee)
    FEE_PERCENT = 0.02
    
    def __init__(self, config: GridConfig, exchange: BitgetExchange,
                 db: Database, logger: logging.Logger,
                 auto_adjust_config: Optional[AutoAdjustConfig] = None,
                 notifications = None):
        self.config = config
        self.exchange = exchange
        self.db = db
        self.logger = logger
        self.trend_detector = TrendDetector(logger)
        self.auto_adjust = auto_adjust_config
        self.notifications = notifications
        
        self._current_trend: Optional[TrendSignal] = None
        self._trailing_high: float = 0
        self._trailing_low: float = float('inf')
        self._last_fill_time: Optional[datetime] = None
        self._last_rebuild_time: Optional[datetime] = None
        self._cached_volatility: Optional[float] = None
        self._volatility_cache_time: Optional[datetime] = None
    
    async def initialize(self):
        """Initialize leverage and check/create grid"""
        try:
            await self.exchange.set_leverage(
                self.config.pair,
                self.config.leverage,
                self.config.margin_mode
            )
            self.logger.info(f"Set leverage to {self.config.leverage}x ({self.config.margin_mode})")
        except Exception as e:
            self.logger.warning(f"Could not set leverage: {e}")
        
        # Check for existing active grid
        active_grid = self.db.get_active_grid(self.config.pair)
        if not active_grid:
            self.logger.info(f"No active grid for {self.config.pair}, will create on first update")
    
    async def update(self):
        """Main update loop - called periodically"""
        try:
            # Get current price
            ticker = await self.exchange.get_ticker(self.config.pair)
            current_price = ticker["last"]
            
            # Update bot state
            self.db.update_bot_state(
                self.config.pair,
                last_price=current_price,
                last_update=datetime.utcnow().isoformat()
            )
            
            # Update trend if filter enabled
            if self.config.trend_filter:
                await self._update_trend()
            
            # Get or create active grid
            grid = self.db.get_active_grid(self.config.pair)
            if not grid:
                grid = await self._create_grid(current_price)
                if not grid:
                    return
            
            # === AUTO-ADJUSTMENT CHECKS ===
            if self.auto_adjust and self.auto_adjust.enabled:
                rebuild_reason = await self._check_auto_adjust(grid, current_price)
                if rebuild_reason:
                    self.logger.info(f"Auto-rebuild triggered: {rebuild_reason}")
                    grid = await self._rebuild_grid(grid, current_price, rebuild_reason)
                    if not grid:
                        return
            
            # Check paper orders
            if self.exchange.paper_trading:
                await self.exchange.check_paper_orders(self.config.pair)
            
            # Process grid
            await self._process_grid(grid, current_price)
            
        except Exception as e:
            self.logger.error(f"Error in update loop: {e}", exc_info=True)
    
    async def _update_trend(self):
        """Update trend signal"""
        try:
            klines = await self.exchange.get_klines(
                self.config.pair,
                self.config.trend_tf,
                limit=200
            )
            
            if not klines:
                return
            
            self._current_trend = self.trend_detector.detect(
                klines,
                self.config.trend_indicator,
                self.config.trend_params
            )
            
            self.db.update_bot_state(
                self.config.pair,
                trend_direction=self._current_trend.direction
            )
            
        except Exception as e:
            self.logger.warning(f"Error updating trend: {e}")
    
    async def _create_grid(self, current_price: float) -> Optional[Grid]:
        """Create a new grid"""
        try:
            # Calculate grid bounds
            if self.config.upper_price > 0 and self.config.lower_price > 0:
                upper = self.config.upper_price
                lower = self.config.lower_price
            else:
                # Auto-calculate from range_percent
                upper = current_price * (1 + self.config.range_percent / 100)
                lower = current_price * (1 - self.config.range_percent / 100)
            
            # Calculate capital per grid
            capital_per_grid = self.config.capital_per_grid
            if capital_per_grid <= 0:
                capital_per_grid = self.config.total_capital / self.config.grid_levels
            
            # Create grid record
            grid = Grid(
                id=None,
                pair=self.config.pair,
                status="active",
                grid_mode=self.config.grid_mode,
                upper_price=upper,
                lower_price=lower,
                grid_levels=self.config.grid_levels,
                grid_spacing=self.config.grid_spacing,
                total_capital=self.config.total_capital,
                capital_per_grid=capital_per_grid,
                total_bought=0,
                total_sold=0,
                net_position=0,
                average_buy_price=0,
                average_sell_price=0,
                total_buy_cost=0,
                total_sell_revenue=0,
                realized_pnl=0,
                unrealized_pnl=0,
                total_trades=0,
                grid_profits=0,
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                stopped_at=None,
                sl_price=None,
                tp_price=None,
                stop_reason=None
            )
            
            grid_id = self.db.create_grid(grid)
            grid.id = grid_id
            
            # Calculate grid levels
            levels = self._calculate_grid_levels(upper, lower)
            
            # Create level records
            for i, price in enumerate(levels):
                level = GridLevel(
                    id=None,
                    grid_id=grid_id,
                    pair=self.config.pair,
                    level_index=i,
                    price=price,
                    buy_order_id=None,
                    sell_order_id=None,
                    buy_status="none",
                    sell_status="none",
                    last_buy_price=None,
                    last_buy_amount=None,
                    last_buy_time=None,
                    last_sell_price=None,
                    last_sell_amount=None,
                    last_sell_time=None,
                    total_buys=0,
                    total_sells=0,
                    level_pnl=0
                )
                level_id = self.db.create_grid_level(level)
                level.id = level_id
            
            # Update bot state
            self.db.update_bot_state(self.config.pair, active_grid_id=grid_id)
            
            self.logger.info(f"Created grid: {lower:.2f} - {upper:.2f}, "
                           f"{self.config.grid_levels} levels, "
                           f"${capital_per_grid:.2f}/level")
            
            return grid
            
        except Exception as e:
            self.logger.error(f"Error creating grid: {e}", exc_info=True)
            return None
    
    def _calculate_grid_levels(self, upper: float, lower: float) -> List[float]:
        """Calculate grid level prices"""
        levels = []
        n = self.config.grid_levels
        
        if self.config.grid_spacing == "arithmetic":
            # Equal price spacing
            step = (upper - lower) / (n - 1) if n > 1 else 0
            for i in range(n):
                levels.append(lower + step * i)
        else:
            # Geometric (equal % spacing)
            ratio = (upper / lower) ** (1 / (n - 1)) if n > 1 else 1
            price = lower
            for i in range(n):
                levels.append(price)
                price *= ratio
        
        return levels
    
    async def _process_grid(self, grid: Grid, current_price: float):
        """Process grid - check fills and place orders"""
        levels = self.db.get_grid_levels(grid.id)
        
        # Check for filled orders
        await self._check_filled_orders(grid, levels)
        
        # Update unrealized P&L
        self._update_unrealized_pnl(grid, current_price)
        
        # Check stop loss / take profit
        if await self._check_grid_stops(grid, current_price):
            return
        
        # Update trailing stop
        self._update_trailing(grid, current_price)
        
        # Check cooldown
        if self._last_fill_time and self.config.cooldown_after_fill > 0:
            elapsed = (datetime.utcnow() - self._last_fill_time).total_seconds()
            if elapsed < self.config.cooldown_after_fill:
                return
        
        # Place missing orders based on grid mode
        await self._place_grid_orders(grid, levels, current_price)
    
    async def _check_filled_orders(self, grid: Grid, levels: List[GridLevel]):
        """Check and process filled orders"""
        pending_orders = self.db.get_pending_orders(grid.id)
        
        for db_order in pending_orders:
            try:
                exchange_order = await self.exchange.get_order(
                    self.config.pair, db_order.order_id
                )
                
                if exchange_order and exchange_order.status == "filled":
                    await self._process_fill(grid, db_order, exchange_order)
                    
            except Exception as e:
                error_str = str(e)
                
                # Handle "order not found" error (40109) - clean up stale orders
                if "40109" in error_str or "cannot be found" in error_str.lower() or "order not found" in error_str.lower():
                    self.logger.info(f"Order {db_order.order_id} no longer exists on exchange, cleaning up...")
                    
                    # Mark order as cancelled/stale in database
                    self.db.update_order_status_by_order_id(
                        db_order.order_id,
                        status="not_found",
                        filled_at=datetime.utcnow().isoformat()
                    )
                    
                    # Clear order ID from level
                    level = self.db.get_level_by_id(db_order.level_id)
                    if level:
                        if db_order.order_type == "buy":
                            level.buy_order_id = None
                            level.buy_status = "none"
                        else:
                            level.sell_order_id = None
                            level.sell_status = "none"
                        self.db.update_grid_level(level)
                else:
                    self.logger.warning(f"Error checking order {db_order.order_id}: {e}")
    
    async def _process_fill(self, grid: Grid, db_order: GridOrder, 
                           exchange_order: OrderResult):
        """Process a filled order"""
        # Use exchange data, but fallback to order data if exchange returns 0
        fill_price = exchange_order.avg_fill_price if exchange_order.avg_fill_price > 0 else db_order.price
        fill_amount = exchange_order.filled_amount if exchange_order.filled_amount > 0 else db_order.amount
        fill_time = datetime.utcnow().isoformat()
        
        # Safety check - don't process if no amount
        if fill_amount <= 0:
            self.logger.warning(f"Skipping fill with 0 amount for order {db_order.order_id}")
            return
        
        # Calculate fee
        fee = fill_price * fill_amount * self.FEE_PERCENT / 100
        
        # Update order record
        self.db.update_order_status_by_order_id(
            db_order.order_id,
            status="filled",
            filled_at=fill_time,
            fill_price=fill_price,
            fill_amount=fill_amount,
            fee=fee
        )
        
        # Get and update level
        level = self.db.get_level_by_id(db_order.level_id)
        if not level:
            return
        
        # Update grid and level based on order type
        if db_order.order_type == "buy":
            # Buy filled
            grid.total_bought += fill_amount
            grid.total_buy_cost += fill_price * fill_amount
            grid.net_position += fill_amount
            
            # Update average buy price
            if grid.total_bought > 0:
                grid.average_buy_price = grid.total_buy_cost / grid.total_bought
            
            # Update level
            level.buy_status = "filled"
            level.buy_order_id = None
            level.last_buy_price = fill_price
            level.last_buy_amount = fill_amount
            level.last_buy_time = fill_time
            level.total_buys += 1
            
            self.logger.info(f"BUY filled @ {fill_price:.4f} ({fill_amount:.6f})")
            
        else:  # sell
            # Sell filled
            grid.total_sold += fill_amount
            grid.total_sell_revenue += fill_price * fill_amount
            grid.net_position -= fill_amount
            
            # Update average sell price
            if grid.total_sold > 0:
                grid.average_sell_price = grid.total_sell_revenue / grid.total_sold
            
            # Update level
            level.sell_status = "filled"
            level.sell_order_id = None
            level.last_sell_price = fill_price
            level.last_sell_amount = fill_amount
            level.last_sell_time = fill_time
            level.total_sells += 1
            
            # Calculate profit for this grid trade
            if level.last_buy_price and level.last_buy_price > 0:
                trade_profit = (fill_price - level.last_buy_price) * fill_amount
                # Subtract fees for both sides (buy and sell)
                fee_buy = level.last_buy_price * fill_amount * self.FEE_PERCENT / 100
                fee_sell = fill_price * fill_amount * self.FEE_PERCENT / 100
                total_fees = fee_buy + fee_sell
                trade_profit -= total_fees
                level.level_pnl += trade_profit
                grid.realized_pnl += trade_profit
                grid.grid_profits += 1
                
                self.logger.info(f"SELL filled @ {fill_price:.4f}, profit: ${trade_profit:.2f} (fees: ${total_fees:.4f})")
            else:
                self.logger.info(f"SELL filled @ {fill_price:.4f}")
        
        # Update counts
        grid.total_trades += 1
        grid.updated_at = datetime.utcnow().isoformat()
        
        # Save updates
        self.db.update_grid_level(level)
        self.db.update_grid(grid)
        
        # Send notification
        if self.notifications:
            try:
                if db_order.order_type == "buy":
                    await self.notifications.notify_trade(
                        pair=self.config.pair,
                        side="buy",
                        price=fill_price,
                        amount=fill_amount,
                        profit=None
                    )
                else:
                    trade_profit = None
                    if level.last_buy_price and level.last_buy_price > 0:
                        fee_buy = level.last_buy_price * fill_amount * self.FEE_PERCENT / 100
                        fee_sell = fill_price * fill_amount * self.FEE_PERCENT / 100
                        trade_profit = (fill_price - level.last_buy_price) * fill_amount - fee_buy - fee_sell
                    await self.notifications.notify_trade(
                        pair=self.config.pair,
                        side="sell",
                        price=fill_price,
                        amount=fill_amount,
                        profit=trade_profit
                    )
            except Exception as e:
                self.logger.warning(f"Failed to send notification: {e}")
        
        # Track fill time for cooldown
        self._last_fill_time = datetime.utcnow()
    
    def _update_unrealized_pnl(self, grid: Grid, current_price: float):
        """Update unrealized P&L based on current position"""
        if grid.net_position != 0 and grid.average_buy_price > 0:
            if grid.net_position > 0:
                # Long position
                grid.unrealized_pnl = (current_price - grid.average_buy_price) * grid.net_position
            else:
                # Short position
                avg_short = grid.average_sell_price if grid.average_sell_price > 0 else current_price
                grid.unrealized_pnl = (avg_short - current_price) * abs(grid.net_position)
        else:
            grid.unrealized_pnl = 0
        
        grid.updated_at = datetime.utcnow().isoformat()
        self.db.update_grid(grid)
    
    async def _check_grid_stops(self, grid: Grid, current_price: float) -> bool:
        """Check stop loss and take profit for grid"""
        # Calculate SL/TP prices if not set
        if grid.average_buy_price > 0 and grid.net_position > 0:
            if self.config.sl_percent > 0:
                grid.sl_price = grid.average_buy_price * (1 - self.config.sl_percent / 100)
            if self.config.tp_percent > 0:
                grid.tp_price = grid.average_buy_price * (1 + self.config.tp_percent / 100)
        
        # Check stop loss
        if grid.sl_price and current_price <= grid.sl_price:
            await self._stop_grid(grid, "sl")
            return True
        
        # Check take profit
        if grid.tp_price and current_price >= grid.tp_price:
            await self._stop_grid(grid, "tp")
            return True
        
        # Check trailing stop
        if self.config.trailing_stop and grid.net_position > 0:
            if await self._check_trailing_stop(grid, current_price):
                return True
        
        return False
    
    def _update_trailing(self, grid: Grid, current_price: float):
        """Update trailing stop values"""
        if current_price > self._trailing_high:
            self._trailing_high = current_price
        if current_price < self._trailing_low:
            self._trailing_low = current_price
    
    async def _check_trailing_stop(self, grid: Grid, current_price: float) -> bool:
        """Check trailing stop"""
        if grid.average_buy_price <= 0:
            return False
        
        profit_percent = ((current_price - grid.average_buy_price) / grid.average_buy_price) * 100
        
        # Check if trailing stop is activated
        if profit_percent >= self.config.trailing_stop_activation:
            # Calculate callback price
            callback_price = self._trailing_high * (1 - self.config.trailing_stop_callback / 100)
            
            if current_price <= callback_price:
                await self._stop_grid(grid, "trailing_stop")
                return True
        
        return False
    
    async def _stop_grid(self, grid: Grid, reason: str):
        """Stop the grid and close positions"""
        try:
            # Cancel all pending orders with retry
            levels = self.db.get_grid_levels(grid.id)
            failed_cancels = []
            
            for level in levels:
                if level.buy_order_id:
                    for attempt in range(3):
                        try:
                            await self.exchange.cancel_order(self.config.pair, level.buy_order_id)
                            break
                        except Exception as e:
                            if attempt == 2:
                                self.logger.warning(f"Failed to cancel buy order {level.buy_order_id} after 3 attempts: {e}")
                                failed_cancels.append(level.buy_order_id)
                            else:
                                await asyncio.sleep(0.5)
                                
                if level.sell_order_id:
                    for attempt in range(3):
                        try:
                            await self.exchange.cancel_order(self.config.pair, level.sell_order_id)
                            break
                        except Exception as e:
                            if attempt == 2:
                                self.logger.warning(f"Failed to cancel sell order {level.sell_order_id} after 3 attempts: {e}")
                                failed_cancels.append(level.sell_order_id)
                            else:
                                await asyncio.sleep(0.5)
            
            # Fallback: try cancel_all_orders if individual cancels failed
            if failed_cancels:
                self.logger.warning(f"Attempting fallback cancel_all_orders due to {len(failed_cancels)} failed cancels")
                try:
                    await self.exchange.cancel_all_orders(self.config.pair)
                except Exception as e:
                    self.logger.error(f"Fallback cancel_all_orders also failed: {e}")
            
            # Close any open position
            if grid.net_position != 0:
                try:
                    if grid.net_position > 0:
                        # Close long
                        result = await self.exchange.place_market_order(
                            self.config.pair, "sell", abs(grid.net_position), reduce_only=True
                        )
                    else:
                        # Close short
                        result = await self.exchange.place_market_order(
                            self.config.pair, "buy", abs(grid.net_position), reduce_only=True
                        )
                    
                    if result and result.status == "filled":
                        # Calculate final P&L with fees
                        if grid.net_position > 0:
                            # Closing long: sold at fill_price, bought at average_buy_price
                            gross_pnl = (result.avg_fill_price - grid.average_buy_price) * grid.net_position
                            fee_open = grid.average_buy_price * grid.net_position * self.FEE_PERCENT / 100
                            fee_close = result.avg_fill_price * grid.net_position * self.FEE_PERCENT / 100
                            final_pnl = gross_pnl - fee_open - fee_close
                        else:
                            # Closing short: bought at fill_price, sold at average_sell_price
                            gross_pnl = (grid.average_sell_price - result.avg_fill_price) * abs(grid.net_position)
                            fee_open = grid.average_sell_price * abs(grid.net_position) * self.FEE_PERCENT / 100
                            fee_close = result.avg_fill_price * abs(grid.net_position) * self.FEE_PERCENT / 100
                            final_pnl = gross_pnl - fee_open - fee_close
                        
                        grid.realized_pnl += final_pnl
                        self.logger.info(f"Position closed: gross ${gross_pnl:.2f}, fees ${fee_open + fee_close:.4f}, net ${final_pnl:.2f}")
                except Exception as e:
                    # Ignore "No position to close" error - position already closed
                    if "No position" not in str(e) and "22002" not in str(e):
                        self.logger.warning(f"Could not close position: {e}")
            
            # Update grid status
            grid.status = "stopped"
            grid.stopped_at = datetime.utcnow().isoformat()
            grid.stop_reason = reason
            grid.updated_at = datetime.utcnow().isoformat()
            self.db.update_grid(grid)
            
            # Clear bot state
            self.db.update_bot_state(self.config.pair, active_grid_id=None)
            
            self.logger.info(f"Grid stopped ({reason}), realized P&L: ${grid.realized_pnl:.2f}")
            
        except Exception as e:
            self.logger.error(f"Error stopping grid: {e}", exc_info=True)
    
    async def _place_grid_orders(self, grid: Grid, levels: List[GridLevel], 
                                 current_price: float):
        """Place grid orders based on current price and mode"""
        # Determine which orders to place based on grid mode and trend
        place_buys = True
        place_sells = True
        
        if self.config.grid_mode == "long":
            place_sells = False
        elif self.config.grid_mode == "short":
            place_buys = False
        
        # Apply trend filter if enabled
        if self.config.trend_filter and self._current_trend:
            if self._current_trend.direction == "long":
                place_sells = False  # Only accumulate in uptrend
            elif self._current_trend.direction == "short":
                place_buys = False   # Only distribute in downtrend
        
        # Count pending orders
        pending_count = sum(1 for l in levels if l.buy_status == "pending" or l.sell_status == "pending")
        max_orders = self.config.max_open_orders if self.config.max_open_orders > 0 else len(levels) * 2
        
        for level in levels:
            if pending_count >= max_orders:
                break
            
            # Place buy order below current price
            if place_buys and level.price < current_price:
                if level.buy_status in ["none", "filled"]:
                    # Only place buy if we haven't bought at this level
                    # or if we've sold since last buy
                    if level.buy_status == "none" or (level.sell_status == "filled" and level.last_sell_time and 
                        (not level.last_buy_time or level.last_sell_time > level.last_buy_time)):
                        
                        if await self._place_buy_order(grid, level):
                            pending_count += 1
            
            # Place sell order above current price
            if place_sells and level.price > current_price:
                if level.sell_status in ["none", "filled"]:
                    # Only place sell if we have bought at this level
                    if level.buy_status == "filled" and level.last_buy_time:
                        if await self._place_sell_order(grid, level):
                            pending_count += 1
    
    # Minimum order value (Bitget requires at least 5 USDT)
    MIN_ORDER_VALUE_USDT = 5.0
    
    async def _place_buy_order(self, grid: Grid, level: GridLevel) -> bool:
        """Place a buy order at grid level"""
        try:
            # capital_per_grid is the margin per level
            # With leverage, actual position size = margin * leverage
            amount = (grid.capital_per_grid * self.config.leverage) / level.price
            
            # Check minimum order value (Bitget requires 5 USDT minimum)
            order_value = amount * level.price
            if order_value < self.MIN_ORDER_VALUE_USDT:
                self.logger.debug(f"Skip BUY: order value ${order_value:.2f} < ${self.MIN_ORDER_VALUE_USDT} minimum")
                return False
            
            result = await self.exchange.place_limit_order(
                self.config.pair, "buy", level.price, amount
            )
            
            if result:
                # Delay to avoid rate limiting (Bitget limit: ~10 req/s)
                await asyncio.sleep(0.5)
                
                # Create order record
                order = GridOrder(
                    id=None,
                    grid_id=grid.id,
                    level_id=level.id,
                    pair=self.config.pair,
                    order_id=result.order_id,
                    order_type="buy",
                    side="buy",
                    price=level.price,
                    amount=amount,
                    status="pending",
                    created_at=datetime.utcnow().isoformat(),
                    filled_at=None,
                    fill_price=None,
                    fill_amount=None,
                    fee=None
                )
                self.db.create_grid_order(order)
                
                # Update level
                level.buy_order_id = result.order_id
                level.buy_status = "pending"
                self.db.update_grid_level(level)
                
                self.logger.debug(f"Placed BUY @ {level.price:.4f}")
                return True
                
        except Exception as e:
            self.logger.warning(f"Error placing buy order: {e}")
        
        return False
    
    async def _place_sell_order(self, grid: Grid, level: GridLevel) -> bool:
        """Place a sell order at grid level"""
        try:
            # Use the amount from last buy if available
            if level.last_buy_amount:
                amount = level.last_buy_amount
            else:
                # capital_per_grid is margin, multiply by leverage for position size
                amount = (grid.capital_per_grid * self.config.leverage) / level.price
            
            # Check minimum order value (Bitget requires 5 USDT minimum)
            order_value = amount * level.price
            if order_value < self.MIN_ORDER_VALUE_USDT:
                self.logger.debug(f"Skip SELL: order value ${order_value:.2f} < ${self.MIN_ORDER_VALUE_USDT} minimum")
                return False
            
            # Calculate minimum profitable sell price
            # Fee is charged on notional value, so effective fee % of margin = FEE * leverage
            # For positive ROI: price_change% > round_trip_fee% = 2 * FEE_PERCENT
            # min_profit_percent is the target net price change after fees
            if level.last_buy_price:
                round_trip_fee_percent = 2 * self.FEE_PERCENT  # 0.12% of price
                min_sell = level.last_buy_price * (1 + (self.config.min_profit_percent + round_trip_fee_percent) / 100)
                if level.price < min_sell:
                    # Adjust to ensure profit
                    level.price = min_sell
                    self.logger.debug(f"Adjusted sell price to {min_sell:.4f} for min profit")
            
            result = await self.exchange.place_limit_order(
                self.config.pair, "sell", level.price, amount
            )
            
            if result:
                # Delay to avoid rate limiting (Bitget limit: ~10 req/s)
                await asyncio.sleep(0.5)
                
                # Create order record
                order = GridOrder(
                    id=None,
                    grid_id=grid.id,
                    level_id=level.id,
                    pair=self.config.pair,
                    order_id=result.order_id,
                    order_type="sell",
                    side="sell",
                    price=level.price,
                    amount=amount,
                    status="pending",
                    created_at=datetime.utcnow().isoformat(),
                    filled_at=None,
                    fill_price=None,
                    fill_amount=None,
                    fee=None
                )
                self.db.create_grid_order(order)
                
                # Update level
                level.sell_order_id = result.order_id
                level.sell_status = "pending"
                self.db.update_grid_level(level)
                
                self.logger.debug(f"Placed SELL @ {level.price:.4f}")
                return True
                
        except Exception as e:
            self.logger.warning(f"Error placing sell order: {e}")
        
        return False
    
    async def cancel_all_orders(self):
        """Cancel all pending orders for this pair"""
        try:
            await self.exchange.cancel_all_orders(self.config.pair)
            
            grid = self.db.get_active_grid(self.config.pair)
            if grid:
                levels = self.db.get_grid_levels(grid.id)
                for level in levels:
                    if level.buy_status == "pending":
                        level.buy_status = "none"
                        level.buy_order_id = None
                    if level.sell_status == "pending":
                        level.sell_status = "none"
                        level.sell_order_id = None
                    self.db.update_grid_level(level)
                
            self.logger.info("Cancelled all orders")
            
        except Exception as e:
            self.logger.error(f"Error cancelling orders: {e}")
    
    async def stop_grid_manual(self, close_position: bool = True):
        """Manually stop the grid"""
        grid = self.db.get_active_grid(self.config.pair)
        if grid:
            if close_position:
                await self._stop_grid(grid, "manual")
            else:
                # Just cancel orders, keep position
                await self.cancel_all_orders()
                grid.status = "stopped"
                grid.stopped_at = datetime.utcnow().isoformat()
                grid.stop_reason = "manual_no_close"
                self.db.update_grid(grid)
                self.db.update_bot_state(self.config.pair, active_grid_id=None)
    
    def get_grid_info(self) -> Optional[Dict[str, Any]]:
        """Get current grid information"""
        grid = self.db.get_active_grid(self.config.pair)
        if not grid:
            return None
        
        levels = self.db.get_grid_levels(grid.id)
        stats = self.db.get_grid_stats(grid.id)
        
        return {
            "grid_id": grid.id,
            "status": grid.status,
            "mode": grid.grid_mode,
            "range": f"{grid.lower_price:.2f} - {grid.upper_price:.2f}",
            "levels": grid.grid_levels,
            "capital": grid.total_capital,
            "capital_per_grid": grid.capital_per_grid,
            "net_position": grid.net_position,
            "avg_buy_price": grid.average_buy_price,
            "realized_pnl": grid.realized_pnl,
            "unrealized_pnl": grid.unrealized_pnl,
            "total_pnl": grid.realized_pnl + grid.unrealized_pnl,
            "total_trades": grid.total_trades,
            "grid_profits": grid.grid_profits,
            "stats": stats,
            "levels_data": [
                {
                    "index": l.level_index,
                    "price": l.price,
                    "buy_status": l.buy_status,
                    "sell_status": l.sell_status,
                    "buys": l.total_buys,
                    "sells": l.total_sells,
                    "pnl": l.level_pnl
                }
                for l in levels
            ]
        }
    
    # === AUTO-ADJUSTMENT METHODS ===
    
    async def _check_auto_adjust(self, grid: Grid, current_price: float) -> Optional[str]:
        """Check if grid needs to be rebuilt. Returns reason or None."""
        
        # Check cooldown
        if self._last_rebuild_time:
            elapsed = (datetime.utcnow() - self._last_rebuild_time).total_seconds() / 60
            if elapsed < self.auto_adjust.min_rebuild_interval_minutes:
                return None
        
        # Check minimum grid age
        grid_created = datetime.fromisoformat(grid.created_at)
        grid_age_minutes = (datetime.utcnow() - grid_created).total_seconds() / 60
        if grid_age_minutes < self.auto_adjust.min_grid_age_minutes:
            return None
        
        # Check 1: Price exited grid range
        if self.auto_adjust.rebuild_on_exit:
            reason = self._check_price_exit(grid, current_price)
            if reason:
                return reason
        
        # Check 2: Grid inefficiency
        if self.auto_adjust.rebuild_on_inefficiency:
            reason = self._check_grid_inefficiency(grid, current_price)
            if reason:
                return reason
        
        return None
    
    def _check_price_exit(self, grid: Grid, current_price: float) -> Optional[str]:
        """Check if price has exited the grid range"""
        buffer = self.auto_adjust.exit_buffer_percent / 100
        
        upper_threshold = grid.upper_price * (1 + buffer)
        lower_threshold = grid.lower_price * (1 - buffer)
        
        if current_price > upper_threshold:
            return f"price_above_range (${current_price:.2f} > ${grid.upper_price:.2f})"
        
        if current_price < lower_threshold:
            return f"price_below_range (${current_price:.2f} < ${grid.lower_price:.2f})"
        
        return None
    
    def _check_grid_inefficiency(self, grid: Grid, current_price: float) -> Optional[str]:
        """Check if grid has become inefficient (too many levels on one side)"""
        levels = self.db.get_grid_levels(grid.id)
        if not levels:
            return None
        
        levels_below = sum(1 for l in levels if l.price < current_price)
        levels_above = len(levels) - levels_below
        
        total_levels = len(levels)
        if total_levels == 0:
            return None
        
        # Calculate percentage on the dominant side
        max_side_percent = max(levels_below, levels_above) / total_levels * 100
        
        if max_side_percent >= self.auto_adjust.inefficiency_threshold:
            side = "below" if levels_below > levels_above else "above"
            return f"inefficient ({max_side_percent:.0f}% levels {side} price)"
        
        return None
    
    async def _get_volatility(self) -> float:
        """Get current volatility, with caching"""
        # Check cache (valid for 1 hour)
        if self._cached_volatility and self._volatility_cache_time:
            cache_age = (datetime.utcnow() - self._volatility_cache_time).total_seconds() / 3600
            if cache_age < 1:
                return self._cached_volatility
        
        try:
            # Get klines for volatility calculation
            klines = await self.exchange.get_klines(
                self.config.pair,
                "1H",
                limit=self.auto_adjust.volatility_lookback_hours
            )
            
            if not klines or len(klines) < 10:
                return self.config.range_percent  # Fallback to config
            
            # Calculate volatility as standard deviation of returns
            closes = [float(k["close"]) for k in klines]
            returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 
                      for i in range(1, len(closes))]
            
            if not returns:
                return self.config.range_percent
            
            avg = sum(returns) / len(returns)
            variance = sum((r - avg) ** 2 for r in returns) / len(returns)
            std = variance ** 0.5
            
            # Annualize and scale
            volatility = std * (24 ** 0.5)  # Daily volatility approximation
            
            self._cached_volatility = volatility
            self._volatility_cache_time = datetime.utcnow()
            
            return volatility
            
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {e}")
            return self.config.range_percent
    
    def _calculate_new_range(self, current_price: float, volatility: float) -> Tuple[float, float]:
        """Calculate new grid range based on price and volatility"""
        # Calculate range percent
        range_percent = volatility * self.auto_adjust.volatility_multiplier
        
        # Clamp to min/max
        range_percent = max(self.auto_adjust.min_range_percent, 
                          min(self.auto_adjust.max_range_percent, range_percent))
        
        # Calculate bounds
        upper = current_price * (1 + range_percent / 100)
        lower = current_price * (1 - range_percent / 100)
        
        return lower, upper
    
    async def _rebuild_grid(self, old_grid: Grid, current_price: float, 
                           reason: str) -> Optional[Grid]:
        """Rebuild grid with new range, optionally carrying over position"""
        try:
            self.logger.info(f"Rebuilding grid for {self.config.pair}: {reason}")
            
            # Store position info before closing old grid
            carry_position = not self.auto_adjust.close_position_on_rebuild
            old_net_position = old_grid.net_position
            old_avg_buy_price = old_grid.average_buy_price
            old_total_bought = old_grid.total_bought
            old_total_buy_cost = old_grid.total_buy_cost
            old_realized_pnl = old_grid.realized_pnl
            
            # Cancel all pending orders with retry
            levels = self.db.get_grid_levels(old_grid.id)
            failed_cancels = []
            
            for level in levels:
                if level.buy_order_id:
                    for attempt in range(3):
                        try:
                            await self.exchange.cancel_order(self.config.pair, level.buy_order_id)
                            break
                        except Exception as e:
                            if attempt == 2:
                                self.logger.warning(f"Failed to cancel buy order {level.buy_order_id} after 3 attempts: {e}")
                                failed_cancels.append(level.buy_order_id)
                            else:
                                await asyncio.sleep(0.5)
                                
                if level.sell_order_id:
                    for attempt in range(3):
                        try:
                            await self.exchange.cancel_order(self.config.pair, level.sell_order_id)
                            break
                        except Exception as e:
                            if attempt == 2:
                                self.logger.warning(f"Failed to cancel sell order {level.sell_order_id} after 3 attempts: {e}")
                                failed_cancels.append(level.sell_order_id)
                            else:
                                await asyncio.sleep(0.5)
            
            # Fallback: try cancel_all_orders if individual cancels failed
            if failed_cancels:
                self.logger.warning(f"Attempting fallback cancel_all_orders due to {len(failed_cancels)} failed cancels")
                try:
                    await self.exchange.cancel_all_orders(self.config.pair)
                except Exception as e:
                    self.logger.error(f"Fallback cancel_all_orders also failed: {e}")
            
            # Close position if configured
            if self.auto_adjust.close_position_on_rebuild and old_net_position != 0:
                if old_net_position > 0:
                    result = await self.exchange.place_market_order(
                        self.config.pair, "sell", abs(old_net_position), reduce_only=True
                    )
                else:
                    result = await self.exchange.place_market_order(
                        self.config.pair, "buy", abs(old_net_position), reduce_only=True
                    )
                
                if result and result.status == "filled":
                    # Calculate close P&L with fees
                    if old_net_position > 0:
                        gross_pnl = (result.avg_fill_price - old_avg_buy_price) * old_net_position
                        fee_open = old_avg_buy_price * old_net_position * self.FEE_PERCENT / 100
                        fee_close = result.avg_fill_price * old_net_position * self.FEE_PERCENT / 100
                        close_pnl = gross_pnl - fee_open - fee_close
                    else:
                        gross_pnl = (old_grid.average_sell_price - result.avg_fill_price) * abs(old_net_position)
                        fee_open = old_grid.average_sell_price * abs(old_net_position) * self.FEE_PERCENT / 100
                        fee_close = result.avg_fill_price * abs(old_net_position) * self.FEE_PERCENT / 100
                        close_pnl = gross_pnl - fee_open - fee_close
                    old_realized_pnl += close_pnl
                    self.logger.info(f"Closed position for rebuild: gross ${gross_pnl:.2f}, fees ${fee_open + fee_close:.4f}, net ${close_pnl:.2f}")
            
            # Mark old grid as stopped
            old_grid.status = "rebuilt"
            old_grid.stopped_at = datetime.utcnow().isoformat()
            old_grid.stop_reason = f"auto_rebuild: {reason}"
            old_grid.realized_pnl = old_realized_pnl
            old_grid.updated_at = datetime.utcnow().isoformat()
            self.db.update_grid(old_grid)
            
            # Calculate new range
            volatility = await self._get_volatility()
            lower, upper = self._calculate_new_range(current_price, volatility)
            
            # Calculate capital per grid
            capital_per_grid = self.config.capital_per_grid
            if capital_per_grid <= 0:
                capital_per_grid = self.config.total_capital / self.config.grid_levels
            
            # Create new grid
            new_grid = Grid(
                id=None,
                pair=self.config.pair,
                status="active",
                grid_mode=self.config.grid_mode,
                upper_price=upper,
                lower_price=lower,
                grid_levels=self.config.grid_levels,
                grid_spacing=self.config.grid_spacing,
                total_capital=self.config.total_capital,
                capital_per_grid=capital_per_grid,
                # Carry over position if configured
                total_bought=old_total_bought if carry_position else 0,
                total_sold=old_grid.total_sold if carry_position else 0,
                net_position=old_net_position if carry_position else 0,
                average_buy_price=old_avg_buy_price if carry_position else 0,
                average_sell_price=old_grid.average_sell_price if carry_position else 0,
                total_buy_cost=old_total_buy_cost if carry_position else 0,
                total_sell_revenue=old_grid.total_sell_revenue if carry_position else 0,
                realized_pnl=old_realized_pnl,  # Always carry realized PnL
                unrealized_pnl=0,
                total_trades=old_grid.total_trades,  # Carry trade count
                grid_profits=old_grid.grid_profits,  # Carry profit count
                created_at=datetime.utcnow().isoformat(),
                updated_at=datetime.utcnow().isoformat(),
                stopped_at=None,
                sl_price=None,
                tp_price=None,
                stop_reason=None
            )
            
            grid_id = self.db.create_grid(new_grid)
            new_grid.id = grid_id
            
            # Create new grid levels
            level_prices = self._calculate_grid_levels(upper, lower)
            
            for i, price in enumerate(level_prices):
                # Determine initial status based on carried position
                buy_status = "none"
                sell_status = "none"
                
                # If carrying position and price is below current, mark as if bought
                if carry_position and old_net_position > 0 and price < current_price:
                    buy_status = "filled"
                
                level = GridLevel(
                    id=None,
                    grid_id=grid_id,
                    pair=self.config.pair,
                    level_index=i,
                    price=price,
                    buy_order_id=None,
                    sell_order_id=None,
                    buy_status=buy_status,
                    sell_status=sell_status,
                    last_buy_price=old_avg_buy_price if buy_status == "filled" else None,
                    last_buy_amount=(capital_per_grid * self.config.leverage) / price if buy_status == "filled" else None,
                    last_buy_time=datetime.utcnow().isoformat() if buy_status == "filled" else None,
                    last_sell_price=None,
                    last_sell_amount=None,
                    last_sell_time=None,
                    total_buys=0,
                    total_sells=0,
                    level_pnl=0
                )
                self.db.create_grid_level(level)
            
            # Update bot state
            self.db.update_bot_state(self.config.pair, active_grid_id=grid_id)
            
            # Update rebuild time
            self._last_rebuild_time = datetime.utcnow()
            
            # Reset trailing values
            self._trailing_high = current_price
            self._trailing_low = current_price
            
            self.logger.info(f"Grid rebuilt: ${lower:.2f} - ${upper:.2f} "
                           f"(volatility: {volatility:.1f}%, range: {((upper-lower)/current_price*100):.1f}%)"
                           f"{' [position carried]' if carry_position and old_net_position != 0 else ''}")
            
            return new_grid
            
        except Exception as e:
            self.logger.error(f"Error rebuilding grid: {e}", exc_info=True)
            return None
    
    async def force_rebuild(self, reason: str = "manual") -> Optional[Grid]:
        """Force immediate grid rebuild"""
        grid = self.db.get_active_grid(self.config.pair)
        if not grid:
            return None
        
        ticker = await self.exchange.get_ticker(self.config.pair)
        current_price = ticker["last"]
        
        return await self._rebuild_grid(grid, current_price, reason)
