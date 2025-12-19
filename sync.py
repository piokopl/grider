"""
Synchronization module for Grid Bot
Handles syncing state between exchange and local database
"""

import asyncio
from datetime import datetime
from typing import Dict, List, Optional
import logging

from database import Database, Grid, GridLevel
from exchange import BitgetExchange


async def full_sync(exchange: BitgetExchange, db: Database,
                   pairs: List[str], logger: logging.Logger) -> Dict[str, bool]:
    """
    Perform full synchronization for all pairs
    Returns dict of pair -> success status
    """
    results = {}
    
    for pair in pairs:
        try:
            success = await sync_pair(exchange, db, pair, logger)
            results[pair] = success
        except Exception as e:
            logger.error(f"Error syncing {pair}: {e}")
            results[pair] = False
    
    return results


async def sync_pair(exchange: BitgetExchange, db: Database,
                   pair: str, logger: logging.Logger) -> bool:
    """
    Synchronize state for a single pair
    - Check for open positions
    - Verify pending orders
    - Update grid state
    """
    try:
        # Get active grid
        grid = db.get_active_grid(pair)
        if not grid:
            logger.debug(f"No active grid for {pair}")
            return True
        
        # Get open orders from exchange
        exchange_orders = await exchange.get_open_orders(pair)
        exchange_order_ids = {o.order_id for o in exchange_orders}
        
        # Get grid levels
        levels = db.get_grid_levels(grid.id)
        
        # Check each level's orders
        for level in levels:
            # Check buy order
            if level.buy_order_id and level.buy_status == "pending":
                if level.buy_order_id not in exchange_order_ids:
                    # Order no longer on exchange - check if filled or cancelled
                    try:
                        order_info = await exchange.get_order(pair, level.buy_order_id)
                        if order_info and order_info.status == "filled":
                            # Mark as filled (will be processed on next update)
                            logger.info(f"Synced: BUY {level.buy_order_id} was filled")
                        else:
                            # Order was cancelled
                            level.buy_status = "none"
                            level.buy_order_id = None
                            db.update_grid_level(level)
                            logger.info(f"Synced: BUY {level.buy_order_id} was cancelled")
                    except:
                        level.buy_status = "none"
                        level.buy_order_id = None
                        db.update_grid_level(level)
            
            # Check sell order
            if level.sell_order_id and level.sell_status == "pending":
                if level.sell_order_id not in exchange_order_ids:
                    try:
                        order_info = await exchange.get_order(pair, level.sell_order_id)
                        if order_info and order_info.status == "filled":
                            logger.info(f"Synced: SELL {level.sell_order_id} was filled")
                        else:
                            level.sell_status = "none"
                            level.sell_order_id = None
                            db.update_grid_level(level)
                            logger.info(f"Synced: SELL {level.sell_order_id} was cancelled")
                    except:
                        level.sell_status = "none"
                        level.sell_order_id = None
                        db.update_grid_level(level)
        
        # Sync position
        positions = await exchange.get_positions(pair)
        if positions:
            pos = positions[0]
            if pos.size > 0:
                # Update grid with position info
                if grid.net_position != pos.size:
                    logger.info(f"{pair}: Synced position: {grid.net_position} -> {pos.size}")
                    grid.net_position = pos.size
                    grid.average_buy_price = pos.entry_price
                    grid.updated_at = datetime.utcnow().isoformat()
                    db.update_grid(grid)
        else:
            # No position on exchange - reset local state if we thought we had one
            if grid.net_position != 0:
                logger.info(f"{pair}: Position closed on exchange, resetting: {grid.net_position} -> 0")
                grid.net_position = 0
                grid.unrealized_pnl = 0
                grid.updated_at = datetime.utcnow().isoformat()
                db.update_grid(grid)
                
                # Also reset level buy statuses so grid can place new orders
                levels = db.get_grid_levels(grid.id)
                for level in levels:
                    if level.buy_status == "filled":
                        level.buy_status = "none"
                        level.sell_status = "none"
                        level.sell_order_id = None
                        db.update_grid_level(level)
        
        return True
        
    except Exception as e:
        logger.error(f"Error in sync_pair {pair}: {e}", exc_info=True)
        return False


async def recover_grid_state(exchange: BitgetExchange, db: Database,
                            pair: str, logger: logging.Logger) -> bool:
    """
    Attempt to recover grid state from exchange data
    Useful after bot restart or crash
    """
    try:
        grid = db.get_active_grid(pair)
        if not grid:
            return False
        
        # Get current positions
        positions = await exchange.get_positions(pair)
        
        if positions:
            pos = positions[0]
            
            # Update grid with current position
            if pos.size > 0:
                grid.net_position = pos.size
                grid.average_buy_price = pos.entry_price
                
                # Estimate bought amount from position
                if grid.total_bought < pos.size:
                    grid.total_bought = pos.size
                    grid.total_buy_cost = pos.size * pos.entry_price
                
                grid.updated_at = datetime.utcnow().isoformat()
                db.update_grid(grid)
                
                logger.info(f"Recovered grid state: position={pos.size}, avg={pos.entry_price}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error recovering grid state: {e}")
        return False


async def cancel_orphaned_orders(exchange: BitgetExchange, db: Database,
                                pair: str, logger: logging.Logger) -> int:
    """
    Cancel any orders on exchange that aren't tracked in database
    Returns count of cancelled orders
    """
    cancelled = 0
    
    try:
        grid = db.get_active_grid(pair)
        if not grid:
            # No active grid - cancel all orders
            await exchange.cancel_all_orders(pair)
            return -1  # Unknown count
        
        # Get tracked order IDs
        levels = db.get_grid_levels(grid.id)
        tracked_ids = set()
        for level in levels:
            if level.buy_order_id:
                tracked_ids.add(level.buy_order_id)
            if level.sell_order_id:
                tracked_ids.add(level.sell_order_id)
        
        # Get exchange orders
        exchange_orders = await exchange.get_open_orders(pair)
        
        # Cancel orphaned orders
        for order in exchange_orders:
            if order.order_id not in tracked_ids:
                try:
                    await exchange.cancel_order(pair, order.order_id)
                    cancelled += 1
                    logger.info(f"Cancelled orphaned order: {order.order_id}")
                except Exception as e:
                    logger.warning(f"Could not cancel order {order.order_id}: {e}")
        
    except Exception as e:
        logger.error(f"Error cancelling orphaned orders: {e}")
    
    return cancelled


async def verify_grid_consistency(db: Database, pair: str, 
                                  logger: logging.Logger) -> Dict:
    """
    Verify grid data consistency
    Returns dict with any issues found
    """
    issues = []
    
    try:
        grid = db.get_active_grid(pair)
        if not grid:
            return {"status": "no_grid", "issues": []}
        
        levels = db.get_grid_levels(grid.id)
        
        # Check level count
        if len(levels) != grid.grid_levels:
            issues.append(f"Level count mismatch: {len(levels)} vs {grid.grid_levels}")
        
        # Check level order
        prices = [l.price for l in levels]
        if prices != sorted(prices):
            issues.append("Levels not sorted by price")
        
        # Check for duplicate prices
        if len(prices) != len(set(prices)):
            issues.append("Duplicate level prices found")
        
        # Check position consistency
        total_bought = sum(l.total_buys * (l.last_buy_amount or 0) for l in levels)
        total_sold = sum(l.total_sells * (l.last_sell_amount or 0) for l in levels)
        
        # Note: This is approximate check
        if abs(grid.total_trades - sum(l.total_buys + l.total_sells for l in levels)) > 1:
            issues.append("Trade count mismatch between grid and levels")
        
        return {
            "status": "ok" if not issues else "issues_found",
            "issues": issues,
            "grid_id": grid.id,
            "level_count": len(levels)
        }
        
    except Exception as e:
        logger.error(f"Error verifying grid consistency: {e}")
        return {"status": "error", "issues": [str(e)]}


async def import_exchange_positions(exchange: BitgetExchange, db: Database,
                                    managers: dict, logger: logging.Logger) -> Dict[str, bool]:
    """
    Import open positions from exchange into local state.
    This allows the bot to take over management of existing positions.
    
    Returns dict of pair -> imported status
    """
    results = {}
    
    try:
        # Get all real positions from exchange
        positions = await exchange.get_real_positions()
        
        if not positions:
            logger.info("No open positions on exchange to import")
            return results
        
        logger.info(f"Found {len(positions)} open positions on exchange")
        
        for pos in positions:
            pair = pos.pair
            
            # Skip if not in our managed pairs
            if pair not in managers:
                logger.info(f"Skipping {pair} - not in managed pairs")
                continue
            
            manager = managers[pair]
            
            try:
                # Check if we already have an active grid
                grid = db.get_active_grid(pair)
                
                if not grid:
                    # Create a new grid for this position
                    logger.info(f"Creating grid for imported position: {pair}")
                    
                    # Get current price
                    ticker = await exchange.get_ticker(pair)
                    current_price = ticker["last"]
                    
                    # Calculate grid range based on position entry price
                    entry_price = pos.entry_price
                    range_percent = manager.config.range_percent or 15
                    
                    # Center grid around entry price
                    price_low = entry_price * (1 - range_percent / 200)
                    price_high = entry_price * (1 + range_percent / 200)
                    
                    # Create grid
                    grid = Grid(
                        id=None,
                        pair=pair,
                        status="active",
                        price_low=price_low,
                        price_high=price_high,
                        grid_levels=manager.config.grid_levels,
                        total_capital=manager.config.total_capital,
                        capital_per_grid=manager.config.total_capital / manager.config.grid_levels,
                        created_at=datetime.utcnow().isoformat(),
                        closed_at=None,
                        realized_pnl=0,
                        total_trades=0,
                        total_fees=0
                    )
                    grid_id = db.create_grid(grid)
                    grid.id = grid_id
                    
                    # Create grid levels
                    price_step = (price_high - price_low) / (manager.config.grid_levels - 1)
                    
                    for i in range(manager.config.grid_levels):
                        level_price = price_low + i * price_step
                        
                        level = GridLevel(
                            id=None,
                            grid_id=grid_id,
                            level_index=i,
                            price=level_price,
                            buy_order_id=None,
                            sell_order_id=None,
                            buy_status="none",
                            sell_status="none",
                            last_buy_price=None,
                            last_sell_price=None,
                            last_buy_amount=None,
                            last_sell_amount=None,
                            last_buy_time=None,
                            last_sell_time=None,
                            total_buys=0,
                            total_sells=0
                        )
                        db.create_grid_level(level)
                    
                    logger.info(f"Created grid for {pair}: {price_low:.4f} - {price_high:.4f}")
                
                # Now mark levels as "filled" based on position
                grid = db.get_active_grid(pair)
                levels = db.get_grid_levels(grid.id)
                
                # Find level closest to entry price
                position_size = pos.size
                amount_per_level = position_size / max(1, len([l for l in levels if l.price <= pos.entry_price]))
                
                # Mark levels below entry as bought (for long position)
                if pos.side == "long":
                    for level in levels:
                        if level.price <= pos.entry_price and level.buy_status != "filled":
                            level.buy_status = "filled"
                            level.last_buy_price = pos.entry_price
                            level.last_buy_amount = amount_per_level
                            level.last_buy_time = datetime.utcnow().isoformat()
                            level.total_buys = 1
                            db.update_grid_level(level)
                            logger.debug(f"{pair} level {level.price:.4f} marked as bought")
                
                # For short positions
                elif pos.side == "short":
                    for level in levels:
                        if level.price >= pos.entry_price and level.sell_status != "filled":
                            level.sell_status = "filled"
                            level.last_sell_price = pos.entry_price
                            level.last_sell_amount = amount_per_level
                            level.last_sell_time = datetime.utcnow().isoformat()
                            level.total_sells = 1
                            db.update_grid_level(level)
                
                results[pair] = True
                logger.info(f"✓ Imported {pair} position: {pos.side} {pos.size:.4f} @ {pos.entry_price:.4f}")
                
                # Small delay to avoid rate limit
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error importing {pair} position: {e}")
                results[pair] = False
        
        return results
        
    except Exception as e:
        logger.error(f"Error importing exchange positions: {e}")
        return results


async def cleanup_stale_orders(exchange: BitgetExchange, db: Database,
                               pairs: List[str], logger: logging.Logger) -> int:
    """
    Clean up stale orders that no longer exist on exchange.
    Called at startup to prevent API errors when checking old orders.
    
    Strategy: If pending order is NOT in exchange's open orders list, mark it as stale.
    This avoids calling get_order for each stale order (which generates API errors).
    
    Returns number of cleaned orders.
    """
    cleaned_count = 0
    
    for pair in pairs:
        try:
            grid = db.get_active_grid(pair)
            if not grid:
                continue
            
            # Get all pending orders from database
            pending_orders = db.get_pending_orders(grid.id)
            if not pending_orders:
                continue
            
            # Get open orders from exchange (this is ONE API call per pair)
            try:
                exchange_orders = await exchange.get_open_orders(pair)
                exchange_order_ids = {o.order_id for o in exchange_orders}
            except Exception as e:
                logger.warning(f"Could not get open orders for {pair}: {e}")
                exchange_order_ids = set()
            
            # Check each pending order - if NOT on exchange, clean it up
            for order in pending_orders:
                if order.order_id not in exchange_order_ids:
                    # Order is not on exchange - mark as stale (DON'T call get_order!)
                    db.update_order_status_by_order_id(
                        order.order_id,
                        status="stale",
                        filled_at=datetime.utcnow().isoformat()
                    )
                    
                    # Clear from level
                    level = db.get_level_by_id(order.level_id)
                    if level:
                        if order.order_type == "buy":
                            level.buy_order_id = None
                            level.buy_status = "none"
                        else:
                            level.sell_order_id = None
                            level.sell_status = "none"
                        db.update_grid_level(level)
                    
                    cleaned_count += 1
            
            # Small delay between pairs
            await asyncio.sleep(0.2)
            
        except Exception as e:
            logger.warning(f"Error cleaning orders for {pair}: {e}")
    
    if cleaned_count > 0:
        logger.info(f"Cleaned up {cleaned_count} stale orders")
    
    return cleaned_count
