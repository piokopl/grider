"""
Grid Manager v2.0 - Zgodny z logiką BULL/BEAR/NEUTRAL

BULL (long grid):
  - BUY LIMIT poniżej ceny (łapanie pullbacków)
  - Po wypełnieniu BUY → SELL TP powyżej (realizacja zysku)
  - Trailing grid w górę gdy cena rośnie

BEAR (short grid):
  - SELL LIMIT powyżej ceny (łapanie odbić)
  - Po wypełnieniu SELL → BUY TP poniżej (realizacja zysku)
  - Trailing grid w dół gdy cena spada

NEUTRAL (symetryczny grid):
  - BUY LIMIT w dolnej części kanału
  - SELL LIMIT w górnej części kanału
  - Po BUY fill → SELL TP wyżej
  - Po SELL fill → BUY TP niżej
"""

import json
import asyncio
import math
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from database import Database, Grid, GridLevel, GridOrder
from exchange import BitgetExchange, OrderResult, Position
from trend import TrendDetector, TrendSignal


@dataclass
class GridConfig:
    pair: str
    enabled: bool
    leverage: int
    margin_mode: str
    
    grid_mode: str  # neutral / long / short (może być zmieniane przez AI)
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
    
    # Nowe parametry dla v2
    tp_steps: int = 1  # ile kroków w górę/dół dla TP (domyślnie 1)
    reanchor_threshold: float = 0.03  # 3% - po ilu % przesunięcia re-anchor grid
    max_position_levels: int = 10  # max ile poziomów może być wypełnionych
    max_orders_per_side: int = 3  # DYNAMIC: max zleceń BUY/SELL per para (default 3)


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
    """
    Grid Manager v2.0 - Prawidłowa logika dla BULL/BEAR/NEUTRAL
    """
    
    # Trading fee (0.02% per side - maker fee)
    FEE_PERCENT = 0.02
    
    # Minimum order value (Bitget requires at least 5 USDT)
    MIN_ORDER_VALUE_USDT = 5.0
    
    def __init__(self, config: GridConfig, exchange: BitgetExchange,
                 db: Database, logger: logging.Logger,
                 auto_adjust_config: Optional[AutoAdjustConfig] = None,
                 notifications = None,
                 ai_advisor = None):
        self.config = config
        self.exchange = exchange
        self.db = db
        self.logger = logger
        self.trend_detector = TrendDetector(logger)
        self.auto_adjust = auto_adjust_config
        self.notifications = notifications
        self.ai_advisor = ai_advisor  # Optional AI advisor for initial mode selection
        
        self._current_trend: Optional[TrendSignal] = None
        self._anchor_price: float = 0  # Cena odniesienia dla grida
        self._last_fill_time: Optional[datetime] = None
        self._last_rebuild_time: Optional[datetime] = None
        self._last_reanchor_time: Optional[datetime] = None
        self._cached_volatility: Optional[float] = None
        self._volatility_cache_time: Optional[datetime] = None
        
        # Tracking dla trailing grid
        self._trailing_high: float = 0
        self._trailing_low: float = float('inf')
        
        # Flag to track if initial AI analysis was done
        self._initial_analysis_done: bool = False
    
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
    
    async def import_existing_position(self) -> bool:
        """
        Przejmij istniejącą pozycję z giełdy i zbuduj grid wokół niej.
        Wywoływane przy starcie bota.
        """
        try:
            # Pobierz pozycję z giełdy
            positions = await self.exchange.get_real_positions()
            position = next((p for p in positions if p.pair == self.config.pair), None)
            
            if not position or position.size == 0:
                self.logger.info(f"{self.config.pair}: No existing position to import")
                return False
            
            self.logger.info(f"{self.config.pair}: Found existing position: {position.side} {position.size} @ {position.entry_price}")
            
            # Pobierz aktualną cenę
            ticker = await self.exchange.get_ticker(self.config.pair)
            current_price = ticker["last"]
            
            # Określ tryb grida na podstawie pozycji
            if position.side == "long":
                grid_mode = "long"
            elif position.side == "short":
                grid_mode = "short"
            else:
                grid_mode = "neutral"
            
            # Sprawdź czy mamy aktywny grid
            active_grid = self.db.get_active_grid(self.config.pair)
            
            if active_grid:
                # Aktualizuj istniejący grid
                self.logger.info(f"Updating existing grid with position data")
                
                # Zaktualizuj net_position i average prices
                if position.side == "long":
                    active_grid.net_position = position.size
                    active_grid.average_buy_price = position.entry_price
                    active_grid.total_bought = position.size
                else:
                    active_grid.net_position = -position.size
                    active_grid.average_sell_price = position.entry_price
                    active_grid.total_sold = position.size
                
                active_grid.unrealized_pnl = position.unrealized_pnl
                active_grid.grid_mode = grid_mode
                active_grid.updated_at = datetime.utcnow().isoformat()
                
                self.db.update_grid(active_grid)
                
                # Postaw zlecenia TP dla istniejącej pozycji
                await self._place_tp_for_existing_position(active_grid, position, current_price)
                
            else:
                # Utwórz nowy grid wokół pozycji
                self.logger.info(f"Creating new grid around existing position")
                
                # Oblicz zakres grida
                range_pct = self.config.range_percent / 100
                
                if position.side == "long":
                    # Grid LONG: BUY poniżej entry, SELL TP powyżej
                    lower_price = position.entry_price * (1 - range_pct)
                    upper_price = position.entry_price * (1 + range_pct * 0.5)  # Mniejszy zakres w górę dla TP
                else:
                    # Grid SHORT: SELL powyżej entry, BUY TP poniżej
                    upper_price = position.entry_price * (1 + range_pct)
                    lower_price = position.entry_price * (1 - range_pct * 0.5)  # Mniejszy zakres w dół dla TP
                
                # Calculate capital per grid if not set
                capital_per_grid = self.config.capital_per_grid
                if capital_per_grid <= 0:
                    capital_per_grid = self.config.total_capital / self.config.grid_levels
                
                # Utwórz grid
                grid = Grid(
                    id=None,
                    pair=self.config.pair,
                    status="active",
                    grid_mode=grid_mode,
                    upper_price=upper_price,
                    lower_price=lower_price,
                    grid_levels=self.config.grid_levels,
                    grid_spacing=self.config.grid_spacing,
                    total_capital=self.config.total_capital,
                    capital_per_grid=capital_per_grid,
                    total_bought=position.size if position.side == "long" else 0,
                    total_sold=position.size if position.side == "short" else 0,
                    net_position=position.size if position.side == "long" else -position.size,
                    average_buy_price=position.entry_price if position.side == "long" else 0,
                    average_sell_price=position.entry_price if position.side == "short" else 0,
                    total_buy_cost=position.size * position.entry_price if position.side == "long" else 0,
                    total_sell_revenue=position.size * position.entry_price if position.side == "short" else 0,
                    realized_pnl=0,
                    unrealized_pnl=position.unrealized_pnl,
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
                
                # Utwórz poziomy
                await self._create_grid_levels(grid, current_price)
                
                # Oznacz poziom najbliższy entry jako "wypełniony"
                await self._mark_entry_level_as_filled(grid, position)
                
                # Postaw zlecenia (DYNAMIC - tylko najbliższe!)
                await self._place_grid_orders_dynamic(grid, current_price)
                
                self.logger.info(f"Created grid #{grid_id} around existing {position.side} position")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing position: {e}", exc_info=True)
            return False
    
    async def _place_tp_for_existing_position(self, grid: Grid, position: Position, current_price: float):
        """Postaw zlecenia TP dla przejętej pozycji"""
        try:
            levels = self.db.get_grid_levels(grid.id)
            tp_step = getattr(self.config, 'tp_steps', 1)
            
            if position.side == "long":
                # Dla LONG: SELL TP powyżej entry
                tp_price = position.entry_price * (1 + self.config.min_profit_percent / 100 * tp_step)
                
                # Znajdź najbliższy poziom dla TP
                closest_level = min(levels, key=lambda l: abs(l.price - tp_price))
                
                if closest_level.sell_status != "pending":
                    # Postaw SELL TP
                    amount = position.size
                    result = await self.exchange.place_limit_order(
                        self.config.pair, "sell", tp_price, amount
                    )
                    
                    if result:
                        self._record_order(grid, closest_level, result, "sell", tp_price, amount)
                        self.logger.info(f"Placed SELL TP @ {tp_price:.4f} for existing long")
                        
            else:  # short
                # Dla SHORT: BUY TP poniżej entry
                tp_price = position.entry_price * (1 - self.config.min_profit_percent / 100 * tp_step)
                
                closest_level = min(levels, key=lambda l: abs(l.price - tp_price))
                
                if closest_level.buy_status != "pending":
                    amount = position.size
                    result = await self.exchange.place_limit_order(
                        self.config.pair, "buy", tp_price, amount
                    )
                    
                    if result:
                        self._record_order(grid, closest_level, result, "buy", tp_price, amount)
                        self.logger.info(f"Placed BUY TP @ {tp_price:.4f} for existing short")
                        
        except Exception as e:
            self.logger.error(f"Error placing TP for existing position: {e}")
    
    async def _mark_entry_level_as_filled(self, grid: Grid, position: Position):
        """Oznacz poziom najbliższy entry jako wypełniony"""
        levels = self.db.get_grid_levels(grid.id)
        closest_level = min(levels, key=lambda l: abs(l.price - position.entry_price))
        
        if position.side == "long":
            closest_level.buy_status = "filled"
            closest_level.last_buy_price = position.entry_price
            closest_level.last_buy_amount = position.size
            closest_level.last_buy_time = datetime.utcnow().isoformat()
            closest_level.total_buys = 1
        else:
            closest_level.sell_status = "filled"
            closest_level.last_sell_price = position.entry_price
            closest_level.last_sell_amount = position.size
            closest_level.last_sell_time = datetime.utcnow().isoformat()
            closest_level.total_sells = 1
        
        self.db.update_grid_level(closest_level)
    
    async def update(self):
        """Main update loop - called periodically"""
        try:
            # Get current price
            ticker = await self.exchange.get_ticker(self.config.pair)
            current_price = ticker["last"]
            
            # Update trailing
            self._trailing_high = max(self._trailing_high, current_price) if self._trailing_high else current_price
            self._trailing_low = min(self._trailing_low, current_price) if self._trailing_low != float('inf') else current_price
            
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
            
            # === REANCHOR CHECK (trailing grid) ===
            await self._check_reanchor(grid, current_price)
            
            # Check paper orders
            if self.exchange.paper_trading:
                await self.exchange.check_paper_orders(self.config.pair)
            
            # Process grid
            await self._process_grid(grid, current_price)
            
        except Exception as e:
            self.logger.error(f"Error in update loop: {e}", exc_info=True)
    
    async def _check_reanchor(self, grid: Grid, current_price: float):
        """
        Sprawdź czy trzeba przesunąć grid (trailing/reanchor).
        Jeśli cena uciekła za daleko od anchor, przebuduj grid.
        """
        if not self._anchor_price:
            self._anchor_price = (grid.upper_price + grid.lower_price) / 2
        
        reanchor_threshold = getattr(self.config, 'reanchor_threshold', 0.03)
        min_interval = timedelta(minutes=5)  # Nie reanchor częściej niż co 5 min
        
        if self._last_reanchor_time and datetime.utcnow() - self._last_reanchor_time < min_interval:
            return
        
        price_change = (current_price - self._anchor_price) / self._anchor_price
        
        if abs(price_change) > reanchor_threshold:
            grid_mode = grid.grid_mode
            
            if grid_mode == "long" and price_change > reanchor_threshold:
                # Cena poszła w górę - przesuń grid w górę
                self.logger.info(f"Reanchoring LONG grid up: price moved {price_change*100:.1f}%")
                await self._reanchor_grid(grid, current_price, "up")
                
            elif grid_mode == "short" and price_change < -reanchor_threshold:
                # Cena poszła w dół - przesuń grid w dół
                self.logger.info(f"Reanchoring SHORT grid down: price moved {price_change*100:.1f}%")
                await self._reanchor_grid(grid, current_price, "down")
                
            elif grid_mode == "neutral" and abs(price_change) > reanchor_threshold * 1.5:
                # Neutral: reanchor gdy mocno wyjdzie z kanału
                direction = "up" if price_change > 0 else "down"
                self.logger.info(f"Reanchoring NEUTRAL grid {direction}: price moved {price_change*100:.1f}%")
                await self._reanchor_grid(grid, current_price, direction)
    
    async def _reanchor_grid(self, grid: Grid, current_price: float, direction: str):
        """Przesuń grid w kierunku ruchu ceny"""
        try:
            # Anuluj stare zlecenia
            await self._cancel_stale_orders(grid, current_price)
            
            # Oblicz nowy zakres
            range_size = grid.upper_price - grid.lower_price
            
            if direction == "up":
                # Przesuń w górę - nowe BUY bliżej ceny
                new_lower = current_price - range_size * 0.6
                new_upper = current_price + range_size * 0.4
            else:
                # Przesuń w dół - nowe SELL bliżej ceny
                new_lower = current_price - range_size * 0.4
                new_upper = current_price + range_size * 0.6
            
            # Aktualizuj grid
            grid.lower_price = new_lower
            grid.upper_price = new_upper
            grid.updated_at = datetime.utcnow().isoformat()
            self.db.update_grid(grid)
            
            # Przebuduj poziomy
            await self._rebuild_grid_levels(grid, current_price)
            
            # Postaw nowe zlecenia (DYNAMIC - tylko najbliższe!)
            await self._place_grid_orders_dynamic(grid, current_price)
            
            # Update tracking
            self._anchor_price = current_price
            self._last_reanchor_time = datetime.utcnow()
            self._trailing_high = current_price
            self._trailing_low = current_price
            
            self.logger.info(f"Grid reanchored: new range [{new_lower:.4f} - {new_upper:.4f}]")
            
        except Exception as e:
            self.logger.error(f"Error reanchoring grid: {e}")
    
    async def _cancel_stale_orders(self, grid: Grid, current_price: float):
        """Anuluj zlecenia które są za daleko od ceny"""
        levels = self.db.get_grid_levels(grid.id)
        stale_threshold = getattr(self.config, 'reanchor_threshold', 0.03) * 2
        
        for level in levels:
            price_diff = abs(level.price - current_price) / current_price
            
            if price_diff > stale_threshold:
                # Anuluj buy order jeśli pending
                if level.buy_status == "pending" and level.buy_order_id:
                    try:
                        await self.exchange.cancel_order(self.config.pair, level.buy_order_id)
                    except:
                        pass  # Order may not exist - that's OK
                    # Always reset in DB
                    self.db.update_order_status_by_order_id(level.buy_order_id, "cancelled")
                    level.buy_status = "none"
                    level.buy_order_id = None
                
                # Anuluj sell order jeśli pending
                if level.sell_status == "pending" and level.sell_order_id:
                    try:
                        await self.exchange.cancel_order(self.config.pair, level.sell_order_id)
                    except:
                        pass
                    self.db.update_order_status_by_order_id(level.sell_order_id, "cancelled")
                    level.sell_status = "none"
                    level.sell_order_id = None
                
                self.db.update_grid_level(level)
    
    async def _rebuild_grid_levels(self, grid: Grid, current_price: float):
        """Przebuduj poziomy grida zachowując stan wypełnionych"""
        levels = self.db.get_grid_levels(grid.id)
        
        # Zapamiętaj wypełnione poziomy
        filled_buys = [l for l in levels if l.buy_status == "filled"]
        filled_sells = [l for l in levels if l.sell_status == "filled"]
        
        # Oblicz nowe ceny poziomów
        new_prices = self._calculate_level_prices(grid)
        
        # Aktualizuj lub utwórz poziomy
        for i, new_price in enumerate(new_prices):
            if i < len(levels):
                level = levels[i]
                old_price = level.price
                level.price = new_price
                
                # Resetuj status jeśli cena się znacząco zmieniła
                price_change = abs(new_price - old_price) / old_price if old_price > 0 else 1
                if price_change > 0.01:  # >1% zmiana
                    if level.buy_status == "pending":
                        level.buy_status = "none"
                    if level.sell_status == "pending":
                        level.sell_status = "none"
                
                self.db.update_grid_level(level)
            else:
                # Utwórz nowy poziom
                new_level = GridLevel(
                    id=None,
                    grid_id=grid.id,
                    pair=self.config.pair,
                    level_index=i,
                    price=new_price,
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
                    level_pnl=0.0
                )
                new_level.id = self.db.create_grid_level(new_level)  # WAŻNE: przypisz ID!
    
    def _calculate_level_prices(self, grid: Grid) -> List[float]:
        """Oblicz ceny poziomów dla grida"""
        prices = []
        
        if grid.grid_spacing == "geometric":
            ratio = (grid.upper_price / grid.lower_price) ** (1 / (grid.grid_levels - 1))
            for i in range(grid.grid_levels):
                prices.append(grid.lower_price * (ratio ** i))
        else:  # arithmetic
            step = (grid.upper_price - grid.lower_price) / (grid.grid_levels - 1)
            for i in range(grid.grid_levels):
                prices.append(grid.lower_price + step * i)
        
        return sorted(prices)
    
    async def _process_grid(self, grid: Grid, current_price: float):
        """Process grid - check fills, place new orders"""
        
        # 1. Check for filled orders
        await self._check_filled_orders(grid, current_price)
        
        # 2. Place new orders based on grid mode (DYNAMIC - only nearest levels)
        await self._place_grid_orders_dynamic(grid, current_price)
        
        # 3. Update unrealized PnL
        await self._update_unrealized_pnl(grid, current_price)
    
    async def _place_grid_orders_dynamic(self, grid: Grid, current_price: float):
        """
        DYNAMIC ORDER PLACEMENT - tylko najbliższe poziomy!
        
        Zamiast stawiać zlecenia na wszystkich poziomach, stawiamy tylko
        kilka najbliższych i dynamicznie dodajemy nowe gdy się wypełnią.
        
        Parametry:
        - max_orders_per_side: max zleceń BUY/SELL per para (domyślnie 3)
        
        Logika:
        - BULL: max N BUY najbliżej ceny + SELL TP dla wypełnionych
        - BEAR: max N SELL najbliżej ceny + BUY TP dla wypełnionych
        - NEUTRAL: max N BUY poniżej + max N SELL powyżej + TP
        """
        # === MARGIN CHECK ===
        # Skip placing orders if we recently had margin issues
        if hasattr(self, '_margin_cooldown_until'):
            if datetime.utcnow() < self._margin_cooldown_until:
                remaining = (self._margin_cooldown_until - datetime.utcnow()).seconds
                if remaining % 60 == 0:  # Log every minute
                    self.logger.info(f"Margin cooldown active - {remaining//60} minutes remaining")
                return  # Still in cooldown, don't try to place orders
        
        levels = self.db.get_grid_levels(grid.id)
        grid_mode = grid.grid_mode
        
        if not levels:
            self.logger.warning(f"No levels found for grid {grid.id}")
            return
        
        # Konfiguracja - ile zleceń max per side
        max_orders_per_side = getattr(self.config, 'max_orders_per_side', 3)
        max_position = getattr(self.config, 'max_position_levels', 10)
        
        # Policz obecne pending i filled
        pending_buys = [l for l in levels if l.buy_status == "pending"]
        pending_sells = [l for l in levels if l.sell_status == "pending"]
        filled_buys = [l for l in levels if l.buy_status == "filled"]
        filled_sells = [l for l in levels if l.sell_status == "filled"]
        
        # Poziomy posortowane od najbliższych do najdalszych
        levels_below = sorted([l for l in levels if l.price < current_price], 
                             key=lambda l: current_price - l.price)  # najbliższe pierwsze
        levels_above = sorted([l for l in levels if l.price > current_price],
                             key=lambda l: l.price - current_price)  # najbliższe pierwsze
        
        # DEBUG logging (set level to DEBUG to see these)
        self.logger.debug(f"Grid mode: {grid_mode}, price: {current_price:.4f}")
        self.logger.debug(f"Levels: {len(levels)} total, {len(levels_below)} below, {len(levels_above)} above")
        self.logger.debug(f"Pending: {len(pending_buys)} buys, {len(pending_sells)} sells")
        
        # ========== PLACE ENTRY ORDERS ==========
        orders_placed = 0
        
        if grid_mode == "long":
            # LONG: BUY poniżej ceny (pullback entries)
            buys_to_place = max_orders_per_side - len(pending_buys)
            
            for level in levels_below:
                # Check if margin cooldown was triggered
                if hasattr(self, '_margin_cooldown_until') and datetime.utcnow() < self._margin_cooldown_until:
                    break
                if buys_to_place <= 0:
                    break
                if len(filled_buys) >= max_position:
                    break
                if level.buy_status == "none" and level.sell_status != "pending":
                    if await self._place_buy_order(grid, level):
                        buys_to_place -= 1
                        orders_placed += 1
                        self.logger.info(f"Placed BUY @ {level.price:.6f}")
        
        elif grid_mode == "short":
            # SHORT: SELL powyżej ceny (bounce entries)
            sells_to_place = max_orders_per_side - len(pending_sells)
            
            for level in levels_above:
                # Check if margin cooldown was triggered
                if hasattr(self, '_margin_cooldown_until') and datetime.utcnow() < self._margin_cooldown_until:
                    break
                if sells_to_place <= 0:
                    break
                if len(filled_sells) >= max_position:
                    break
                if level.sell_status == "none" and level.buy_status != "pending":
                    if await self._place_sell_order(grid, level):
                        sells_to_place -= 1
                        orders_placed += 1
                        self.logger.info(f"Placed SELL @ {level.price:.6f}")
        
        else:  # neutral
            # NEUTRAL: BUY poniżej + SELL powyżej
            buys_to_place = max_orders_per_side - len(pending_buys)
            sells_to_place = max_orders_per_side - len(pending_sells)
            
            # BUY na najbliższych poziomach poniżej
            for level in levels_below:
                # Check if margin cooldown was triggered
                if hasattr(self, '_margin_cooldown_until') and datetime.utcnow() < self._margin_cooldown_until:
                    break
                if buys_to_place <= 0:
                    break
                if len(filled_buys) >= max_position:
                    break
                if level.buy_status == "none":
                    if await self._place_buy_order(grid, level):
                        buys_to_place -= 1
                        orders_placed += 1
                        self.logger.info(f"Placed BUY @ {level.price:.6f}")
            
            # SELL na najbliższych poziomach powyżej
            for level in levels_above:
                # Check if margin cooldown was triggered
                if hasattr(self, '_margin_cooldown_until') and datetime.utcnow() < self._margin_cooldown_until:
                    break
                if sells_to_place <= 0:
                    break
                if len(filled_sells) >= max_position:
                    break
                if level.sell_status == "none":
                    if await self._place_sell_order(grid, level):
                        sells_to_place -= 1
                        orders_placed += 1
                        self.logger.info(f"Placed SELL @ {level.price:.6f}")
        
        # ========== PLACE TP ORDERS (zawsze dla wszystkich filled) ==========
        
        # SELL TP dla wypełnionych BUY (LONG i NEUTRAL)
        if grid_mode in ["long", "neutral"]:
            for level in filled_buys:
                # Check if margin cooldown was triggered
                if hasattr(self, '_margin_cooldown_until') and datetime.utcnow() < self._margin_cooldown_until:
                    break
                if level.sell_status in ["none", "filled"]:
                    tp_price = self._calculate_tp_price(level, "sell")
                    if tp_price > current_price * 0.995:
                        if await self._place_sell_tp(grid, level, tp_price):
                            orders_placed += 1
                            self.logger.info(f"Placed TP SELL @ {tp_price:.6f}")
        
        # BUY TP dla wypełnionych SELL (SHORT i NEUTRAL)
        if grid_mode in ["short", "neutral"]:
            for level in filled_sells:
                # Check if margin cooldown was triggered
                if hasattr(self, '_margin_cooldown_until') and datetime.utcnow() < self._margin_cooldown_until:
                    break
                if level.buy_status in ["none", "filled"]:
                    tp_price = self._calculate_tp_price(level, "buy")
                    if tp_price < current_price * 1.005:
                        if await self._place_buy_tp(grid, level, tp_price):
                            orders_placed += 1
                            self.logger.info(f"Placed TP BUY @ {tp_price:.6f}")
        
        if orders_placed > 0:
            self.logger.info(f"Dynamic placement: {orders_placed} orders placed (mode={grid_mode}, pending: {len(pending_buys)}B/{len(pending_sells)}S)")
    
    # Legacy function for compatibility
    async def _place_grid_orders(self, grid: Grid, current_price: float):
        """Wrapper - używa dynamicznej wersji"""
        await self._place_grid_orders_dynamic(grid, current_price)
    
    def _calculate_tp_price(self, level: GridLevel, tp_side: str) -> float:
        """Oblicz cenę TP dla poziomu"""
        tp_steps = getattr(self.config, 'tp_steps', 1)
        min_profit_pct = self.config.min_profit_percent / 100
        
        # Spacing między poziomami
        spacing = (self.config.upper_price - self.config.lower_price) / (self.config.grid_levels - 1)
        
        if tp_side == "sell":
            # TP dla long: entry + profit
            entry_price = level.last_buy_price if level.last_buy_price else level.price
            tp_price = entry_price * (1 + min_profit_pct * tp_steps)
            # Minimum spacing
            tp_price = max(tp_price, entry_price + spacing * tp_steps)
        else:  # buy
            # TP dla short: entry - profit
            entry_price = level.last_sell_price if level.last_sell_price else level.price
            tp_price = entry_price * (1 - min_profit_pct * tp_steps)
            # Minimum spacing
            tp_price = min(tp_price, entry_price - spacing * tp_steps)
        
        return tp_price
    
    async def _place_buy_order(self, grid: Grid, level: GridLevel) -> bool:
        """Postaw zlecenie BUY na poziomie (entry dla LONG lub TP dla SHORT)"""
        try:
            amount = (grid.capital_per_grid * self.config.leverage) / level.price
            
            # Check minimum order value
            order_value = amount * level.price
            if order_value < self.MIN_ORDER_VALUE_USDT:
                self.logger.warning(f"Order too small: ${order_value:.2f} < ${self.MIN_ORDER_VALUE_USDT}")
                return False
            
            result = await self.exchange.place_limit_order(
                self.config.pair, "buy", level.price, amount
            )
            
            if result:
                await asyncio.sleep(0.3)  # Rate limit delay
                self._record_order(grid, level, result, "buy", level.price, amount)
                level.buy_status = "pending"
                self.db.update_grid_level(level)
                return True
            else:
                self.logger.warning(f"Failed to place BUY order - no result from exchange")
                return False
            
        except Exception as e:
            error_msg = str(e)
            # Check for margin/balance errors
            if "40762" in error_msg or "exceeds the balance" in error_msg.lower():
                self.logger.warning(f"Insufficient margin for BUY order - pausing for 5 minutes")
                self._margin_cooldown_until = datetime.utcnow() + timedelta(minutes=5)
            else:
                self.logger.error(f"Error placing buy order: {e}")
            return False
    
    async def _place_sell_order(self, grid: Grid, level: GridLevel) -> bool:
        """Postaw zlecenie SELL na poziomie (entry dla SHORT)"""
        try:
            amount = (grid.capital_per_grid * self.config.leverage) / level.price
            
            order_value = amount * level.price
            if order_value < self.MIN_ORDER_VALUE_USDT:
                self.logger.warning(f"Order too small: ${order_value:.2f} < ${self.MIN_ORDER_VALUE_USDT}")
                return False
            
            result = await self.exchange.place_limit_order(
                self.config.pair, "sell", level.price, amount
            )
            
            if result:
                await asyncio.sleep(0.3)
                self._record_order(grid, level, result, "sell", level.price, amount)
                level.sell_status = "pending"
                self.db.update_grid_level(level)
                return True
            else:
                self.logger.warning(f"Failed to place SELL order - no result from exchange")
                return False
            
        except Exception as e:
            error_msg = str(e)
            # Check for margin/balance errors
            if "40762" in error_msg or "exceeds the balance" in error_msg.lower():
                self.logger.warning(f"Insufficient margin for SELL order - pausing for 5 minutes")
                self._margin_cooldown_until = datetime.utcnow() + timedelta(minutes=5)
            else:
                self.logger.error(f"Error placing sell order: {e}")
            return False
    
    async def _place_sell_tp(self, grid: Grid, level: GridLevel, tp_price: float) -> bool:
        """Postaw zlecenie SELL TP dla wypełnionego BUY"""
        try:
            # Użyj ilości z BUY
            amount = level.last_buy_amount if level.last_buy_amount else \
                     (grid.capital_per_grid * self.config.leverage) / level.price
            
            order_value = amount * tp_price
            if order_value < self.MIN_ORDER_VALUE_USDT:
                return False
            
            result = await self.exchange.place_limit_order(
                self.config.pair, "sell", tp_price, amount
            )
            
            if result:
                await asyncio.sleep(0.3)
                self._record_order(grid, level, result, "sell", tp_price, amount)
                level.sell_status = "pending"
                self.db.update_grid_level(level)
                self.logger.info(f"Placed SELL TP @ {tp_price:.4f} for BUY @ {level.last_buy_price:.4f}")
                return True
            
            return False
            
        except Exception as e:
            error_msg = str(e)
            if "40762" in error_msg or "exceeds the balance" in error_msg.lower():
                self.logger.warning(f"Insufficient margin for SELL TP - pausing for 5 minutes")
                self._margin_cooldown_until = datetime.utcnow() + timedelta(minutes=5)
            else:
                self.logger.error(f"Error placing sell TP: {e}")
            return False
    
    async def _place_buy_tp(self, grid: Grid, level: GridLevel, tp_price: float) -> bool:
        """Postaw zlecenie BUY TP dla wypełnionego SELL"""
        try:
            amount = level.last_sell_amount if level.last_sell_amount else \
                     (grid.capital_per_grid * self.config.leverage) / level.price
            
            order_value = amount * tp_price
            if order_value < self.MIN_ORDER_VALUE_USDT:
                return False
            
            result = await self.exchange.place_limit_order(
                self.config.pair, "buy", tp_price, amount
            )
            
            if result:
                await asyncio.sleep(0.3)
                self._record_order(grid, level, result, "buy", tp_price, amount)
                level.buy_status = "pending"
                self.db.update_grid_level(level)
                self.logger.info(f"Placed BUY TP @ {tp_price:.4f} for SELL @ {level.last_sell_price:.4f}")
                return True
            
            return False
            
        except Exception as e:
            error_msg = str(e)
            if "40762" in error_msg or "exceeds the balance" in error_msg.lower():
                self.logger.warning(f"Insufficient margin for BUY TP - pausing for 5 minutes")
                self._margin_cooldown_until = datetime.utcnow() + timedelta(minutes=5)
            else:
                self.logger.error(f"Error placing buy TP: {e}")
            return False
    
    def _record_order(self, grid: Grid, level: GridLevel, result: OrderResult, 
                      side: str, price: float, amount: float):
        """Zapisz zlecenie do bazy i zaktualizuj level"""
        order = GridOrder(
            id=None,
            grid_id=grid.id,
            level_id=level.id,
            pair=self.config.pair,
            order_id=result.order_id,
            order_type="limit",
            side=side,
            price=price,
            amount=amount,
            status="pending",
            created_at=datetime.utcnow().isoformat(),
            filled_at=None,
            fill_price=None,
            fill_amount=None,
            fee=None
        )
        self.db.create_grid_order(order)
        
        # Update level with order ID
        if side == "buy":
            level.buy_order_id = result.order_id
        else:
            level.sell_order_id = result.order_id
        self.db.update_grid_level(level)
    
    async def _check_filled_orders(self, grid: Grid, current_price: float):
        """Sprawdź wypełnione zlecenia i zaktualizuj stan"""
        pending_orders = self.db.get_pending_orders(grid.id)
        
        for order in pending_orders:
            try:
                # Check order status on exchange
                order_info = await self.exchange.get_order(self.config.pair, order.order_id)
                
                if not order_info:
                    # Order not found - mark as cancelled
                    self.logger.info(f"Order {order.order_id} not found, marking as cancelled")
                    self._mark_order_cancelled(order)
                    continue
                
                # OrderResult is a dataclass, use attributes not .get()
                if order_info.status == "filled":
                    await self._handle_filled_order(grid, order, order_info, current_price)
                    
                elif order_info.status == "cancelled":
                    self._mark_order_cancelled(order)
                        
            except Exception as e:
                error_msg = str(e)
                # If order doesn't exist on exchange, mark it as cancelled
                if "does not exist" in error_msg or "40034" in error_msg:
                    self.logger.info(f"Order {order.order_id} no longer exists, marking as cancelled")
                    self._mark_order_cancelled(order)
                else:
                    self.logger.warning(f"Error checking order {order.order_id}: {e}")
    
    def _mark_order_cancelled(self, order: GridOrder):
        """Oznacz zlecenie jako anulowane i zresetuj status poziomu"""
        self.db.update_order_status_by_order_id(order.order_id, "cancelled")
        
        # Check if order was cancelled immediately (within 60 seconds of creation)
        # This typically indicates margin issues
        try:
            created = datetime.fromisoformat(order.created_at.replace('Z', '+00:00').replace('+00:00', ''))
            age_seconds = (datetime.utcnow() - created).total_seconds()
            
            if age_seconds < 60:
                # Order was cancelled very quickly - likely margin issue
                if not hasattr(self, '_rapid_cancel_count'):
                    self._rapid_cancel_count = 0
                self._rapid_cancel_count += 1
                
                if self._rapid_cancel_count >= 3:
                    self.logger.warning(f"Multiple rapid order cancellations detected - pausing for 5 minutes (likely margin issue)")
                    self._margin_cooldown_until = datetime.utcnow() + timedelta(minutes=5)
                    self._rapid_cancel_count = 0
            else:
                # Order was cancelled after some time - reset counter
                self._rapid_cancel_count = 0
        except:
            pass  # Ignore datetime parsing errors
        
        # Reset level status
        level = self.db.get_level_by_id(order.level_id)
        if level:
            if order.side == "buy":
                level.buy_status = "none"
                level.buy_order_id = None
            else:
                level.sell_status = "none"
                level.sell_order_id = None
            self.db.update_grid_level(level)
    
    async def _handle_filled_order(self, grid: Grid, order: GridOrder, 
                                   order_info: 'OrderResult', current_price: float):
        """
        Obsłuż wypełnione zlecenie:
        - BUY filled → aktualizuj pozycję, postaw SELL TP
        - SELL filled → aktualizuj pozycję, postaw BUY TP (lub zamknij)
        """
        # OrderResult is a dataclass - use attributes
        fill_price = order_info.avg_fill_price or order_info.price or order.price
        fill_amount = order_info.filled_amount or order_info.amount or order.amount
        fee = 0.0  # Fee not available in OrderResult, calculate later if needed
        
        # Update order in database
        self.db.update_order_status_by_order_id(
            order.order_id,
            status="filled",
            filled_at=datetime.utcnow().isoformat(),
            fill_price=fill_price,
            fill_amount=fill_amount,
            fee=fee
        )
        
        # Get level
        level = self.db.get_level_by_id(order.level_id)
        if not level:
            return
        
        grid_mode = grid.grid_mode
        
        if order.side == "buy":
            # ========== BUY FILLED ==========
            level.buy_status = "filled"
            level.last_buy_price = fill_price
            level.last_buy_amount = fill_amount
            level.last_buy_time = datetime.utcnow().isoformat()
            level.total_buys = (level.total_buys or 0) + 1
            
            # Update grid totals
            grid.total_bought = (grid.total_bought or 0) + fill_amount
            grid.total_buy_cost = (grid.total_buy_cost or 0) + (fill_price * fill_amount)
            grid.average_buy_price = grid.total_buy_cost / grid.total_bought if grid.total_bought > 0 else 0
            
            # Sprawdź czy to był BUY TP (zamknięcie SHORT)
            if grid_mode == "short" or (grid_mode == "neutral" and level.sell_status == "filled"):
                # To jest realizacja zysku ze SHORT
                if level.last_sell_price:
                    profit = (level.last_sell_price - fill_price) * fill_amount
                    profit -= fee + (level.last_sell_price * fill_amount * self.FEE_PERCENT / 100)
                    
                    grid.realized_pnl = (grid.realized_pnl or 0) + profit
                    grid.grid_profits = (grid.grid_profits or 0) + 1
                    level.level_pnl = (level.level_pnl or 0) + profit
                    
                    self.logger.info(f"✅ SHORT TP: SELL @ {level.last_sell_price:.4f} → BUY @ {fill_price:.4f} = +${profit:.4f}")
                    
                    # Reset level for next cycle
                    level.sell_status = "none"
                    level.last_sell_price = 0
                    level.last_sell_amount = 0
                    level.buy_status = "none"  # Reset buy też
                    
            else:
                # To jest entry LONG - będzie potrzebny SELL TP
                self.logger.info(f"📈 LONG entry: BUY @ {fill_price:.4f} x {fill_amount:.4f}")
            
            # Update net position
            grid.net_position = (grid.net_position or 0) + fill_amount
            grid.total_trades = (grid.total_trades or 0) + 1
            
        else:  # sell
            # ========== SELL FILLED ==========
            level.sell_status = "filled"
            level.last_sell_price = fill_price
            level.last_sell_amount = fill_amount
            level.last_sell_time = datetime.utcnow().isoformat()
            level.total_sells = (level.total_sells or 0) + 1
            
            # Update grid totals
            grid.total_sold = (grid.total_sold or 0) + fill_amount
            grid.total_sell_revenue = (grid.total_sell_revenue or 0) + (fill_price * fill_amount)
            grid.average_sell_price = grid.total_sell_revenue / grid.total_sold if grid.total_sold > 0 else 0
            
            # Sprawdź czy to był SELL TP (zamknięcie LONG)
            if grid_mode == "long" or (grid_mode == "neutral" and level.buy_status == "filled"):
                # To jest realizacja zysku z LONG
                if level.last_buy_price:
                    profit = (fill_price - level.last_buy_price) * fill_amount
                    profit -= fee + (level.last_buy_price * fill_amount * self.FEE_PERCENT / 100)
                    
                    grid.realized_pnl = (grid.realized_pnl or 0) + profit
                    grid.grid_profits = (grid.grid_profits or 0) + 1
                    level.level_pnl = (level.level_pnl or 0) + profit
                    
                    self.logger.info(f"✅ LONG TP: BUY @ {level.last_buy_price:.4f} → SELL @ {fill_price:.4f} = +${profit:.4f}")
                    
                    # Reset level for next cycle
                    level.buy_status = "none"
                    level.last_buy_price = 0
                    level.last_buy_amount = 0
                    level.sell_status = "none"
                    
            else:
                # To jest entry SHORT - będzie potrzebny BUY TP
                self.logger.info(f"📉 SHORT entry: SELL @ {fill_price:.4f} x {fill_amount:.4f}")
            
            # Update net position
            grid.net_position = (grid.net_position or 0) - fill_amount
            grid.total_trades = (grid.total_trades or 0) + 1
        
        # Save updates
        grid.updated_at = datetime.utcnow().isoformat()
        self.db.update_grid_level(level)
        self.db.update_grid(grid)
        
        self._last_fill_time = datetime.utcnow()
        
        # Send notification
        if self.notifications:
            await self._send_fill_notification(order, fill_price, fill_amount, grid)
    
    async def _send_fill_notification(self, order: GridOrder, fill_price: float, 
                                       fill_amount: float, grid: Grid):
        """Wyślij powiadomienie o wypełnieniu"""
        try:
            side_emoji = "🟢" if order.side == "buy" else "🔴"
            message = (
                f"{side_emoji} {self.config.pair} {order.side.upper()}\n"
                f"Price: ${fill_price:.4f}\n"
                f"Amount: {fill_amount:.4f}\n"
                f"Grid PnL: ${grid.realized_pnl:.2f}"
            )
            await self.notifications.send(message)
        except Exception as e:
            self.logger.warning(f"Failed to send notification: {e}")
    
    async def _create_grid(self, current_price: float) -> Optional[Grid]:
        """Utwórz nowy grid - najpierw przeprowadź analizę AI jeśli dostępna"""
        try:
            # === INITIAL AI ANALYSIS ===
            grid_mode = self.config.grid_mode  # default from config
            
            if self.ai_advisor and not self._initial_analysis_done:
                self.logger.info(f"🤖 Running initial AI analysis before creating grid...")
                try:
                    # Get market data for analysis
                    klines = await self.exchange.get_klines(self.config.pair, "1H", limit=48)
                    
                    if klines:
                        # Calculate volatility
                        closes = [float(k["close"]) for k in klines]
                        if len(closes) > 1:
                            import math
                            returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 
                                      for i in range(1, len(closes))]
                            volatility = math.sqrt(sum(r**2 for r in returns) / len(returns)) * math.sqrt(24)
                        else:
                            volatility = 5.0
                        
                        # Get trend info
                        trend_info = {
                            "direction": self._current_trend.direction if self._current_trend else "unknown",
                            "strength": self._current_trend.strength if self._current_trend else 50
                        }
                        
                        # Get AI recommendation
                        recommendation = await self.ai_advisor.get_trading_recommendation(
                            self.config.pair, klines, current_price, volatility, trend_info
                        )
                        
                        if recommendation:
                            final = recommendation.get("final_recommendation", {})
                            sentiment = recommendation.get("sentiment", {})
                            regime = recommendation.get("regime", {})
                            confidence = final.get("confidence", 0)
                            suggested_mode = final.get("grid_mode", "neutral")
                            
                            # Log analysis results
                            if sentiment:
                                self.logger.info(f"  📰 Sentiment: {sentiment.get('sentiment', 'N/A')} "
                                               f"(score: {sentiment.get('score', 0):.2f})")
                            if regime:
                                self.logger.info(f"  📊 Regime: {regime.get('regime', 'N/A')} "
                                               f"(conf: {regime.get('confidence', 0):.2f})")
                            self.logger.info(f"  🎯 Recommendation: {suggested_mode} (conf: {confidence:.2f})")
                            
                            # Apply if confident enough (lower threshold for initial setup)
                            min_confidence = 0.5  # Lower than normal for initial grid
                            if confidence >= min_confidence:
                                grid_mode = suggested_mode
                                self.logger.info(f"  ✅ Using AI-recommended mode: {grid_mode}")
                            else:
                                self.logger.info(f"  ⚠️ Low confidence ({confidence:.2f}), using default: {grid_mode}")
                    
                    self._initial_analysis_done = True
                    
                except Exception as e:
                    self.logger.warning(f"AI analysis failed, using default mode: {e}")
                    grid_mode = self.config.grid_mode
            
            # Calculate price range
            if self.config.upper_price > 0 and self.config.lower_price > 0:
                upper = self.config.upper_price
                lower = self.config.lower_price
            else:
                range_pct = self.config.range_percent / 100
                upper = current_price * (1 + range_pct / 2)
                lower = current_price * (1 - range_pct / 2)
            
            # Calculate capital per grid if not set
            capital_per_grid = self.config.capital_per_grid
            if capital_per_grid <= 0:
                capital_per_grid = self.config.total_capital / self.config.grid_levels
            
            grid = Grid(
                id=None,
                pair=self.config.pair,
                status="active",
                grid_mode=grid_mode,  # Use AI-determined or default mode
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
            
            # Update config with selected mode (for consistency)
            self.config.grid_mode = grid_mode
            
            # Create grid levels
            await self._create_grid_levels(grid, current_price)
            
            self._anchor_price = current_price
            self._last_rebuild_time = datetime.utcnow()  # Prevent immediate efficiency check
            
            self.logger.info(f"Created grid #{grid_id}: {lower:.4f} - {upper:.4f} ({self.config.grid_levels} levels, mode={grid_mode})")
            
            return grid
            
        except Exception as e:
            self.logger.error(f"Error creating grid: {e}", exc_info=True)
            return None
    
    async def _create_grid_levels(self, grid: Grid, current_price: float):
        """Utwórz poziomy grida"""
        prices = self._calculate_level_prices(grid)
        
        for i, price in enumerate(prices):
            level = GridLevel(
                id=None,
                grid_id=grid.id,
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
                level_pnl=0.0
            )
            level.id = self.db.create_grid_level(level)  # WAŻNE: przypisz ID!
    
    async def _update_unrealized_pnl(self, grid: Grid, current_price: float):
        """Aktualizuj unrealized PnL"""
        if grid.net_position > 0:
            # Long position
            avg_price = grid.average_buy_price or current_price
            grid.unrealized_pnl = (current_price - avg_price) * grid.net_position
        elif grid.net_position < 0:
            # Short position
            avg_price = grid.average_sell_price or current_price
            grid.unrealized_pnl = (avg_price - current_price) * abs(grid.net_position)
        else:
            grid.unrealized_pnl = 0
        
        self.db.update_grid(grid)
    
    async def _update_trend(self):
        """Update trend signal"""
        try:
            klines = await self.exchange.get_klines(
                self.config.pair,
                self.config.trend_tf,
                limit=100
            )
            
            if klines:
                self._current_trend = self.trend_detector.detect(
                    klines,
                    self.config.trend_indicator,
                    self.config.trend_params
                )
        except Exception as e:
            self.logger.warning(f"Error updating trend: {e}")
    
    async def switch_regime(self, new_mode: str, current_price: float) -> bool:
        """
        Przełącz reżim grida (BULL/BEAR/NEUTRAL).
        Implementuje procedurę bezpiecznego przejścia.
        """
        grid = self.db.get_active_grid(self.config.pair)
        if not grid:
            return False
        
        old_mode = grid.grid_mode
        
        if old_mode == new_mode:
            return True
        
        self.logger.info(f"Switching regime: {old_mode} → {new_mode}")
        
        try:
            # === NEUTRAL → BULL ===
            if old_mode == "neutral" and new_mode == "long":
                # Anuluj SELL z górnej części (hamują trend)
                await self._cancel_orders_by_side(grid, "sell", current_price, "above")
                # Zamknij ewentualny short
                if grid.net_position < 0:
                    await self._close_position(grid, "short", current_price)
            
            # === NEUTRAL → BEAR ===
            elif old_mode == "neutral" and new_mode == "short":
                # Anuluj BUY z dolnej części (łapanie spadającego noża)
                await self._cancel_orders_by_side(grid, "buy", current_price, "below")
                # Zamknij ewentualny long
                if grid.net_position > 0:
                    await self._close_position(grid, "long", current_price)
            
            # === BULL → NEUTRAL ===
            elif old_mode == "long" and new_mode == "neutral":
                # Zredukuj pozycję long
                if grid.net_position > 0:
                    await self._reduce_position(grid, "long", current_price, 0.5)  # Zamknij 50%
            
            # === BEAR → NEUTRAL ===
            elif old_mode == "short" and new_mode == "neutral":
                # Zredukuj pozycję short
                if grid.net_position < 0:
                    await self._reduce_position(grid, "short", current_price, 0.5)
            
            # === BULL ↔ BEAR (bezpośrednie) ===
            elif (old_mode == "long" and new_mode == "short") or \
                 (old_mode == "short" and new_mode == "long"):
                # Najpierw zamknij całą pozycję
                if grid.net_position != 0:
                    side = "long" if grid.net_position > 0 else "short"
                    await self._close_position(grid, side, current_price)
                # Anuluj wszystkie zlecenia
                await self._cancel_all_orders(grid)
            
            # Update grid mode
            grid.grid_mode = new_mode
            grid.updated_at = datetime.utcnow().isoformat()
            self.db.update_grid(grid)
            
            # Rebuild levels for new mode
            await self._rebuild_grid_levels(grid, current_price)
            
            self.logger.info(f"Regime switched to {new_mode}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error switching regime: {e}")
            return False
    
    async def _cancel_orders_by_side(self, grid: Grid, side: str, 
                                      current_price: float, position: str):
        """Anuluj zlecenia po określonej stronie ceny"""
        pending = self.db.get_pending_orders(grid.id)
        
        for order in pending:
            if order.side != side:
                continue
            
            should_cancel = False
            if position == "above" and order.price > current_price:
                should_cancel = True
            elif position == "below" and order.price < current_price:
                should_cancel = True
            
            if should_cancel:
                try:
                    await self.exchange.cancel_order(self.config.pair, order.order_id)
                except:
                    pass  # Order may not exist
                
                # Always update DB regardless of exchange result
                self.db.update_order_status_by_order_id(order.order_id, "cancelled")
                
                level = self.db.get_level_by_id(order.level_id)
                if level:
                    if side == "buy":
                        level.buy_status = "none"
                        level.buy_order_id = None
                    else:
                        level.sell_status = "none"
                        level.sell_order_id = None
                    self.db.update_grid_level(level)
    
    async def _cancel_all_orders(self, grid: Grid):
        """Anuluj wszystkie pending zlecenia"""
        pending = self.db.get_pending_orders(grid.id)
        
        for order in pending:
            try:
                await self.exchange.cancel_order(self.config.pair, order.order_id)
            except Exception as e:
                # Order may already be cancelled/filled - that's OK
                self.logger.debug(f"Cancel order {order.order_id}: {e}")
            
            # Always mark as cancelled in DB regardless of exchange result
            self.db.update_order_status_by_order_id(order.order_id, "cancelled")
        
        # Reset all levels - clear order IDs and statuses
        levels = self.db.get_grid_levels(grid.id)
        for level in levels:
            if level.buy_status == "pending":
                level.buy_status = "none"
                level.buy_order_id = None
            if level.sell_status == "pending":
                level.sell_status = "none"
                level.sell_order_id = None
            self.db.update_grid_level(level)
    
    async def _close_position(self, grid: Grid, side: str, current_price: float):
        """Zamknij całą pozycję"""
        try:
            size = abs(grid.net_position)
            if size < 0.00001:
                return
            
            close_side = "sell" if side == "long" else "buy"
            
            result = await self.exchange.place_market_order(
                self.config.pair, close_side, size
            )
            
            if result:
                self.logger.info(f"Closed {side} position: {size} @ market")
                
                # Update grid
                grid.net_position = 0
                grid.updated_at = datetime.utcnow().isoformat()
                self.db.update_grid(grid)
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    async def _reduce_position(self, grid: Grid, side: str, 
                               current_price: float, fraction: float):
        """Zredukuj pozycję o fraction (0-1)"""
        try:
            size = abs(grid.net_position) * fraction
            if size < 0.00001:
                return
            
            close_side = "sell" if side == "long" else "buy"
            
            result = await self.exchange.place_market_order(
                self.config.pair, close_side, size
            )
            
            if result:
                self.logger.info(f"Reduced {side} position by {fraction*100}%: {size}")
                
                # Update grid
                if side == "long":
                    grid.net_position -= size
                else:
                    grid.net_position += size
                grid.updated_at = datetime.utcnow().isoformat()
                self.db.update_grid(grid)
                
        except Exception as e:
            self.logger.error(f"Error reducing position: {e}")
    
    async def _check_auto_adjust(self, grid: Grid, current_price: float) -> Optional[str]:
        """Sprawdź czy potrzebna jest automatyczna korekta grida"""
        if not self.auto_adjust or not self.auto_adjust.enabled:
            return None
        
        # Don't check if we just rebuilt (give time for orders to be placed)
        if self._last_rebuild_time:
            time_since_rebuild = datetime.utcnow() - self._last_rebuild_time
            min_interval = timedelta(minutes=self.auto_adjust.min_rebuild_interval_minutes)
            if time_since_rebuild < min_interval:
                return None  # Too soon after last rebuild
        
        # Check price exit
        if self.auto_adjust.rebuild_on_exit:
            buffer = self.auto_adjust.exit_buffer_percent / 100
            
            if current_price > grid.upper_price * (1 + buffer):
                return f"Price {current_price:.4f} above upper {grid.upper_price:.4f}"
            
            if current_price < grid.lower_price * (1 - buffer):
                return f"Price {current_price:.4f} below lower {grid.lower_price:.4f}"
        
        # Check inefficiency
        if self.auto_adjust.rebuild_on_inefficiency:
            levels = self.db.get_grid_levels(grid.id)
            total_levels = len(levels)
            
            if total_levels > 0:
                active_levels = sum(1 for l in levels 
                                   if l.buy_status != "none" or l.sell_status != "none")
                efficiency = (active_levels / total_levels) * 100
                
                if efficiency < self.auto_adjust.inefficiency_threshold:
                    return f"Low efficiency: {efficiency:.0f}%"
        
        return None
    
    async def _rebuild_grid(self, grid: Grid, current_price: float, 
                            reason: str) -> Optional[Grid]:
        """Przebuduj grid"""
        try:
            # Check minimum rebuild interval
            if self._last_rebuild_time:
                min_interval = timedelta(minutes=self.auto_adjust.min_rebuild_interval_minutes)
                if datetime.utcnow() - self._last_rebuild_time < min_interval:
                    return grid
            
            self.logger.info(f"Rebuilding grid: {reason}")
            
            # Optionally close position
            if self.auto_adjust.close_position_on_rebuild and grid.net_position != 0:
                side = "long" if grid.net_position > 0 else "short"
                await self._close_position(grid, side, current_price)
            
            # Cancel all orders
            await self._cancel_all_orders(grid)
            
            # Calculate new range
            range_pct = self.config.range_percent / 100
            
            # Use volatility if available
            if self.auto_adjust.volatility_multiplier > 0:
                volatility = await self._get_volatility()
                if volatility:
                    range_pct = max(
                        self.auto_adjust.min_range_percent / 100,
                        min(
                            self.auto_adjust.max_range_percent / 100,
                            volatility * self.auto_adjust.volatility_multiplier
                        )
                    )
            
            # Update grid range
            grid.upper_price = current_price * (1 + range_pct / 2)
            grid.lower_price = current_price * (1 - range_pct / 2)
            grid.updated_at = datetime.utcnow().isoformat()
            self.db.update_grid(grid)
            
            # Rebuild levels
            await self._rebuild_grid_levels(grid, current_price)
            
            self._anchor_price = current_price
            self._last_rebuild_time = datetime.utcnow()
            
            self.logger.info(f"Grid rebuilt: {grid.lower_price:.4f} - {grid.upper_price:.4f}")
            
            return grid
            
        except Exception as e:
            self.logger.error(f"Error rebuilding grid: {e}")
            return grid
    
    async def _get_volatility(self) -> Optional[float]:
        """Pobierz zmienność dla auto-adjust"""
        try:
            # Use cached if fresh
            if self._cached_volatility and self._volatility_cache_time:
                cache_age = datetime.utcnow() - self._volatility_cache_time
                if cache_age < timedelta(hours=1):
                    return self._cached_volatility
            
            # Calculate from klines
            klines = await self.exchange.get_klines(
                self.config.pair,
                "1H",
                limit=self.auto_adjust.volatility_lookback_hours
            )
            
            if klines and len(klines) > 10:
                highs = [k["high"] for k in klines]
                lows = [k["low"] for k in klines]
                
                # ATR-like calculation
                ranges = [(h - l) / ((h + l) / 2) for h, l in zip(highs, lows)]
                volatility = sum(ranges) / len(ranges)
                
                self._cached_volatility = volatility
                self._volatility_cache_time = datetime.utcnow()
                
                return volatility
                
        except Exception as e:
            self.logger.warning(f"Error calculating volatility: {e}")
        
        return None
    
    def get_status(self) -> dict:
        """Pobierz status grida"""
        grid = self.db.get_active_grid(self.config.pair)
        
        if not grid:
            return {
                "pair": self.config.pair,
                "status": "no_grid",
                "enabled": self.config.enabled
            }
        
        levels = self.db.get_grid_levels(grid.id)
        pending_orders = self.db.get_pending_orders(grid.id)
        
        return {
            "pair": self.config.pair,
            "status": grid.status,
            "enabled": self.config.enabled,
            "mode": grid.grid_mode,
            "range": f"{grid.lower_price:.4f} - {grid.upper_price:.4f}",
            "levels": len(levels),
            "pending_orders": len(pending_orders),
            "net_position": grid.net_position,
            "realized_pnl": grid.realized_pnl,
            "unrealized_pnl": grid.unrealized_pnl,
            "total_trades": grid.total_trades,
            "grid_profits": grid.grid_profits,
            "anchor_price": self._anchor_price,
            "current_trend": self._current_trend.direction if self._current_trend else None
        }
    
    async def stop_grid(self, close_positions: bool = False):
        """Zatrzymaj grid"""
        grid = self.db.get_active_grid(self.config.pair)
        
        if not grid:
            return
        
        # Cancel all orders
        await self._cancel_all_orders(grid)
        
        # Close positions if requested
        if close_positions and grid.net_position != 0:
            ticker = await self.exchange.get_ticker(self.config.pair)
            current_price = ticker["last"]
            
            side = "long" if grid.net_position > 0 else "short"
            await self._close_position(grid, side, current_price)
        
        # Update grid status
        grid.status = "stopped"
        grid.stopped_at = datetime.utcnow().isoformat()
        grid.updated_at = datetime.utcnow().isoformat()
        self.db.update_grid(grid)
        
        self.logger.info(f"Grid stopped for {self.config.pair}")
    
    async def sync_with_exchange(self) -> bool:
        """
        Synchronizuj stan lokalny z giełdą.
        Używane przy starcie bota do przejęcia istniejących pozycji i zleceń.
        """
        try:
            self.logger.info(f"Syncing {self.config.pair} with exchange...")
            
            # 1. Import existing position
            await self.import_existing_position()
            
            # 2. Sync open orders
            real_orders = await self.exchange.get_real_open_orders()
            pair_orders = [o for o in real_orders if o.pair == self.config.pair]
            
            if pair_orders:
                self.logger.info(f"Found {len(pair_orders)} open orders on exchange")
                
                grid = self.db.get_active_grid(self.config.pair)
                if grid:
                    levels = self.db.get_grid_levels(grid.id)
                    
                    # Match exchange orders to grid levels
                    for order in pair_orders:
                        closest_level = min(levels, key=lambda l: abs(l.price - order.price))
                        
                        # Update level status based on order
                        if order.side == "buy":
                            if closest_level.buy_status != "pending":
                                closest_level.buy_status = "pending"
                                self.db.update_grid_level(closest_level)
                        else:
                            if closest_level.sell_status != "pending":
                                closest_level.sell_status = "pending"
                                self.db.update_grid_level(closest_level)
            
            self.logger.info(f"Sync completed for {self.config.pair}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error syncing with exchange: {e}")
            return False
