#!/usr/bin/env python3
"""
Exchange Sync Task - Runs in background and syncs Bitget data to local database

This provides:
1. Fast dashboard loading (reads from DB instead of API)
2. Reduced API calls to Bitget (rate limiting)
3. Consistent data across dashboard refreshes
4. Historical fills tracking
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from database import Database
from exchange import BitgetExchange


class ExchangeSyncTask:
    """Background task that syncs exchange data to local database"""
    
    def __init__(self, db: Database, exchange: BitgetExchange, 
                 logger: logging.Logger, sync_interval: int = 30):
        self.db = db
        self.exchange = exchange
        self.logger = logger
        self.sync_interval = sync_interval  # seconds
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._last_sync: Optional[datetime] = None
        self._sync_count = 0
        self._error_count = 0
    
    async def start(self):
        """Start the sync task"""
        if self._running:
            return
        
        self._running = True
        self._task = asyncio.create_task(self._sync_loop())
        self.logger.info(f"Exchange sync task started (interval: {self.sync_interval}s)")
    
    async def stop(self):
        """Stop the sync task"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self.logger.info("Exchange sync task stopped")
    
    async def _sync_loop(self):
        """Main sync loop"""
        # Initial sync immediately
        await self._do_sync()
        
        while self._running:
            try:
                await asyncio.sleep(self.sync_interval)
                await self._do_sync()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                self._error_count += 1
                await asyncio.sleep(5)  # Brief pause on error
    
    async def _do_sync(self):
        """Perform one sync cycle"""
        try:
            start_time = datetime.now()
            
            # 1. Sync balance
            await self._sync_balance()
            
            # 2. Sync positions
            await self._sync_positions()
            
            # 3. Sync recent fills
            await self._sync_fills()
            
            # 4. Sync orders count
            await self._sync_orders_count()
            
            # Update stats
            self._last_sync = datetime.now()
            self._sync_count += 1
            
            elapsed = (datetime.now() - start_time).total_seconds()
            
            if self._sync_count % 10 == 0:  # Log every 10th sync
                self.logger.debug(f"Exchange sync #{self._sync_count} completed in {elapsed:.2f}s")
                
        except Exception as e:
            self.logger.error(f"Sync error: {e}")
            self._error_count += 1
    
    async def _sync_balance(self):
        """Sync account balance"""
        try:
            balance = await self.exchange.get_real_account_balance()
            if balance and balance.get('total', 0) > 0:
                self.db.save_exchange_balance(balance)
        except Exception as e:
            self.logger.warning(f"Balance sync failed: {e}")
    
    async def _sync_positions(self):
        """Sync open positions"""
        try:
            positions = await self.exchange.get_real_positions()
            self.db.save_exchange_positions(positions)
        except Exception as e:
            self.logger.warning(f"Positions sync failed: {e}")
    
    async def _sync_fills(self):
        """Sync recent fills"""
        try:
            fills = await self.exchange.get_recent_fills(limit=50)
            if fills:
                # Add trade_id for deduplication
                for i, fill in enumerate(fills):
                    if 'trade_id' not in fill:
                        fill['trade_id'] = f"{fill.get('pair')}_{fill.get('filled_at')}_{i}"
                self.db.save_exchange_fills(fills)
        except Exception as e:
            self.logger.warning(f"Fills sync failed: {e}")
    
    async def _sync_orders_count(self):
        """Sync open orders count"""
        try:
            orders = await self.exchange.get_real_open_orders()
            total = len(orders)
            buys = sum(1 for o in orders if o.side == 'buy')
            sells = total - buys
            self.db.save_exchange_orders_count(total, buys, sells)
        except Exception as e:
            self.logger.warning(f"Orders count sync failed: {e}")
    
    async def force_sync(self):
        """Force an immediate sync"""
        await self._do_sync()
    
    def get_status(self) -> dict:
        """Get sync task status"""
        return {
            'running': self._running,
            'last_sync': self._last_sync.isoformat() if self._last_sync else None,
            'sync_count': self._sync_count,
            'error_count': self._error_count,
            'interval': self.sync_interval
        }


# Standalone test
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from utils import load_global_config
    
    async def test():
        logging.basicConfig(level=logging.DEBUG)
        logger = logging.getLogger("sync_test")
        
        config = load_global_config()
        db = Database("grid_bot.db")
        
        exchange = BitgetExchange(
            api_key=config.api_key,
            api_secret=config.api_secret,
            passphrase=config.passphrase,
            paper_trading=False,
            logger=logger
        )
        
        sync = ExchangeSyncTask(db, exchange, logger, sync_interval=30)
        
        print("Running single sync...")
        await sync._do_sync()
        
        print("\nCached balance:")
        print(db.get_exchange_balance())
        
        print("\nCached positions:")
        for pos in db.get_exchange_positions()[:5]:
            print(f"  {pos}")
        
        print("\nCached fills:")
        for fill in db.get_exchange_fills(5):
            print(f"  {fill}")
        
        print("\nCached orders count:")
        print(db.get_exchange_orders_count())
        
        print("\nCache age:", db.get_exchange_cache_age(), "seconds")
    
    asyncio.run(test())
