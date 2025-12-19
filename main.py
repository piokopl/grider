import asyncio
import signal
import sys
from typing import Dict, List
from pathlib import Path

from database import Database
from exchange import BitgetExchange
# Use v2 grid manager with proper BULL/BEAR/NEUTRAL logic
from grid_manager_v2 import GridManager, GridConfig
from sync import full_sync, import_exchange_positions, cleanup_stale_orders
from webserver import WebServer
from optimizer import GridOptimizer, AutoPilot
from ai_analysis import AITradingAdvisor, AIConfig
from notifications import NotificationManager, NotificationConfig
from exchange_sync import ExchangeSyncTask
from utils import (
    load_global_config, load_all_pair_configs,
    setup_logger, setup_pair_logger, GlobalConfig
)


class GridBot:
    def __init__(self, global_config: GlobalConfig, pair_configs: List[GridConfig]):
        self.global_config = global_config
        self.pair_configs = pair_configs
        
        # Setup main logger
        self.logger = setup_logger(
            "GridBot",
            "logs/main.log",
            global_config.log_level
        )
        
        # Initialize database
        self.db = Database("grid_bot.db")
        
        # Initialize exchange
        self.exchange = BitgetExchange(
            api_key=global_config.api_key,
            api_secret=global_config.api_secret,
            passphrase=global_config.passphrase,
            paper_trading=global_config.paper_trading,
            logger=self.logger
        )
        
        # Grid managers per pair
        self.managers: Dict[str, GridManager] = {}
        
        # Web server
        self.webserver: WebServer = None
        
        # Optimizer and AutoPilot
        self.optimizer: GridOptimizer = None
        self.autopilot: AutoPilot = None
        
        # AI Advisor
        self.ai_advisor: AITradingAdvisor = None
        
        # Notification Manager
        self.notifications: NotificationManager = None
        
        # Running flag
        self._running = False
        self._tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start the bot"""
        self.logger.info("=" * 50)
        self.logger.info("Starting GRID Bot")
        self.logger.info(f"Mode: {'PAPER TRADING' if self.global_config.paper_trading else 'LIVE'}")
        self.logger.info(f"Pairs: {len(self.pair_configs)}")
        self.logger.info(f"Max orders per side: {self.global_config.max_orders_per_side}")
        self.logger.info("=" * 50)
        
        # Initialize notification manager
        notif_config = NotificationConfig(
            discord_enabled=self.global_config.notifications.discord_enabled,
            discord_webhook_url=self.global_config.notifications.discord_webhook_url,
            telegram_enabled=self.global_config.notifications.telegram_enabled,
            telegram_bot_token=self.global_config.notifications.telegram_bot_token,
            telegram_chat_id=self.global_config.notifications.telegram_chat_id,
            notify_trades=self.global_config.notifications.notify_trades,
            notify_rebuilds=self.global_config.notifications.notify_rebuilds,
            notify_optimizations=self.global_config.notifications.notify_optimizations,
            notify_errors=self.global_config.notifications.notify_errors,
            notify_daily_summary=self.global_config.notifications.notify_daily_summary,
            notify_risk_alerts=self.global_config.notifications.notify_risk_alerts,
            min_interval_seconds=self.global_config.notifications.min_interval_seconds,
            batch_trades=self.global_config.notifications.batch_trades,
            batch_interval_seconds=self.global_config.notifications.batch_interval_seconds
        )
        self.notifications = NotificationManager(notif_config, self.logger)
        await self.notifications.start()
        
        if notif_config.discord_enabled or notif_config.telegram_enabled:
            self.logger.info(f"Notifications enabled: Discord={notif_config.discord_enabled}, Telegram={notif_config.telegram_enabled}")
        
        # Create grid managers
        for config in self.pair_configs:
            if not config.enabled:
                self.logger.info(f"Skipping disabled pair: {config.pair}")
                continue
            
            # Apply global max_orders_per_side if not set per-pair (still at default 3)
            if config.max_orders_per_side == 3:  # Default value
                config.max_orders_per_side = self.global_config.max_orders_per_side
            
            pair_logger = setup_pair_logger(
                config.pair,
                "logs",
                self.global_config.log_level
            )
            
            # Convert global auto_adjust config to GridManager format
            from grid_manager import AutoAdjustConfig as GridAutoAdjust
            auto_adjust = None
            if self.global_config.auto_adjust.enabled:
                auto_adjust = GridAutoAdjust(
                    enabled=self.global_config.auto_adjust.enabled,
                    rebuild_on_exit=self.global_config.auto_adjust.rebuild_on_exit,
                    exit_buffer_percent=self.global_config.auto_adjust.exit_buffer_percent,
                    scheduled_check_hours=self.global_config.auto_adjust.scheduled_check_hours,
                    rebuild_on_inefficiency=self.global_config.auto_adjust.rebuild_on_inefficiency,
                    inefficiency_threshold=self.global_config.auto_adjust.inefficiency_threshold,
                    min_grid_age_minutes=self.global_config.auto_adjust.min_grid_age_minutes,
                    volatility_lookback_hours=self.global_config.auto_adjust.volatility_lookback_hours,
                    volatility_multiplier=self.global_config.auto_adjust.volatility_multiplier,
                    min_range_percent=self.global_config.auto_adjust.min_range_percent,
                    max_range_percent=self.global_config.auto_adjust.max_range_percent,
                    close_position_on_rebuild=self.global_config.auto_adjust.close_position_on_rebuild,
                    min_rebuild_interval_minutes=self.global_config.auto_adjust.min_rebuild_interval_minutes
                )
            
            manager = GridManager(config, self.exchange, self.db, pair_logger, auto_adjust, self.notifications)
            self.managers[config.pair] = manager
            
            self.logger.info(f"Loaded config for {config.pair}: "
                           f"{config.grid_levels} levels, ${config.total_capital}, "
                           f"max_orders={config.max_orders_per_side}/side"
                           f"{' [auto-adjust ON]' if auto_adjust else ''}")
        
        if not self.managers:
            self.logger.error("No enabled pairs found!")
            return
        
        # Sync state
        self.logger.info("Syncing state with exchange...")
        sync_results = await full_sync(
            self.exchange,
            self.db,
            list(self.managers.keys()),
            self.logger
        )
        
        for pair, success in sync_results.items():
            status = "OK" if success else "FAILED"
            self.logger.info(f"  {pair}: {status}")
        
        # Set position mode to one-way (required for Bitget)
        if not self.global_config.paper_trading:
            self.logger.info("Setting position mode to one-way...")
            await self.exchange.set_position_mode(one_way=True)
            
            # Import existing positions from exchange
            self.logger.info("Checking for existing positions on exchange...")
            import_results = await import_exchange_positions(
                self.exchange, self.db, self.managers, self.logger
            )
            if import_results:
                self.logger.info(f"Imported {len([r for r in import_results.values() if r])} positions from exchange")
            
            # Clean up stale orders that no longer exist on exchange
            self.logger.info("Cleaning up stale orders...")
            cleaned = await cleanup_stale_orders(
                self.exchange, self.db, list(self.managers.keys()), self.logger
            )
        
        # Initialize managers (with delays to avoid rate limiting)
        for i, (pair, manager) in enumerate(self.managers.items()):
            await manager.initialize()
            
            # Sync with exchange to import existing positions
            if not self.global_config.paper_trading:
                await manager.sync_with_exchange()
            
            # Small delay between initializations to avoid rate limit
            if not self.global_config.paper_trading and i < len(self.managers) - 1:
                await asyncio.sleep(1.0)  # Increased delay for sync
        
        # Start web server
        self.webserver = WebServer(
            db=self.db,
            exchange=self.exchange,
            managers=self.managers,
            logger=self.logger,
            port=self.global_config.web_port
        )
        await self.webserver.start()
        
        # Initialize optimizer and autopilot
        self.optimizer = GridOptimizer(self.db, self.logger)
        self.autopilot = AutoPilot(self.db, self.optimizer, self.logger)
        
        # Initialize AI advisor if configured
        if hasattr(self.global_config, 'ai') and self.global_config.ai.enabled:
            self.ai_advisor = AITradingAdvisor(self.global_config.ai, self.logger)
            self.logger.info("AI Trading Advisor enabled")
            # Pass AI advisor reference to webserver for dashboard
            self.webserver.ai_advisor = self.ai_advisor
            # Pass AI advisor to all managers for initial grid mode selection
            for manager in self.managers.values():
                manager.ai_advisor = self.ai_advisor
        
        # Start main loop
        self._running = True
        self.logger.info("Starting main loop...")
        
        # Create tasks for each pair with staggered start
        for i, (pair, manager) in enumerate(self.managers.items()):
            interval = manager.config.check_interval or self.global_config.update_interval
            # Stagger start: each pair starts 3 seconds after the previous one
            start_delay = i * 5  # 5 seconds between each pair start
            task = asyncio.create_task(self._run_pair(pair, manager, interval, start_delay))
            self._tasks.append(task)
        
        # Start position sync task (every 5 minutes)
        if not self.global_config.paper_trading:
            task = asyncio.create_task(self._run_position_sync())
            self._tasks.append(task)
            self.logger.info("Position sync task started (every 5 min)")
        
        # Start exchange sync task for dashboard (every 30s)
        if not self.global_config.paper_trading:
            self.exchange_sync = ExchangeSyncTask(
                db=self.db,
                exchange=self.exchange,
                logger=self.logger,
                sync_interval=30  # Sync every 30 seconds
            )
            await self.exchange_sync.start()
            # Pass sync task to webserver
            self.webserver.exchange_sync = self.exchange_sync
            self.logger.info("Exchange sync task started (every 30s) - dashboard uses cached data")
        
        # Start auto-adjust scheduled check if enabled
        if self.global_config.auto_adjust.enabled:
            task = asyncio.create_task(self._run_scheduled_check())
            self._tasks.append(task)
            self.logger.info(f"Auto-adjust enabled: rebuild on exit={self.global_config.auto_adjust.rebuild_on_exit}, "
                           f"inefficiency threshold={self.global_config.auto_adjust.inefficiency_threshold}%")
        
        # Start auto-optimization if enabled
        if self.global_config.auto_optimize.enabled:
            task = asyncio.create_task(self._run_auto_optimization())
            self._tasks.append(task)
            self.logger.info(f"Auto-optimize enabled: interval={self.global_config.auto_optimize.optimize_interval_hours}h, "
                           f"auto_apply={self.global_config.auto_optimize.auto_apply}")
        
        # Start AI analysis if enabled
        if self.ai_advisor:
            task = asyncio.create_task(self._run_ai_analysis())
            self._tasks.append(task)
            self.logger.info("AI analysis task started")
        
        # Wait for all tasks
        try:
            await asyncio.gather(*self._tasks)
        except asyncio.CancelledError:
            pass
    
    async def _run_pair(self, pair: str, manager: GridManager, interval: int, start_delay: int = 0):
        """Run loop for a single pair"""
        # Staggered start to avoid rate limiting
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        
        while self._running:
            try:
                await manager.update()
            except Exception as e:
                self.logger.error(f"Error in {pair} loop: {e}", exc_info=True)
            
            await asyncio.sleep(interval)
    
    async def _run_scheduled_check(self):
        """Run scheduled grid efficiency checks and sync with exchange"""
        interval_seconds = self.global_config.auto_adjust.scheduled_check_hours * 3600
        
        # Wait before first check
        await asyncio.sleep(300)  # 5 minutes
        
        while self._running:
            try:
                self.logger.info("=" * 40)
                self.logger.info("Scheduled sync & status report")
                self.logger.info("=" * 40)
                
                # First, sync positions with exchange
                self.logger.info("Syncing positions with exchange...")
                sync_results = await full_sync(
                    self.exchange,
                    self.db,
                    list(self.managers.keys()),
                    self.logger
                )
                synced = sum(1 for v in sync_results.values() if v)
                self.logger.info(f"Synced {synced}/{len(sync_results)} pairs")
                
                # Then report status
                for pair, manager in self.managers.items():
                    try:
                        grid = self.db.get_active_grid(pair)
                        if not grid:
                            self.logger.info(f"{pair}: No active grid")
                            continue
                        
                        # Calculate efficiency
                        levels = self.db.get_grid_levels(grid.id)
                        bot_state = self.db.get_bot_state(pair)
                        current_price = bot_state.get("last_price", 0)
                        
                        if levels and current_price:
                            levels_below = sum(1 for l in levels if l.price < current_price)
                            levels_above = len(levels) - levels_below
                            balance = f"{levels_below}↓/{levels_above}↑"
                        else:
                            balance = "N/A"
                        
                        self.logger.info(
                            f"{pair}: ${grid.lower_price:.0f}-${grid.upper_price:.0f} | "
                            f"levels={balance} | trades={grid.total_trades} | "
                            f"P&L=${grid.realized_pnl + grid.unrealized_pnl:.2f}"
                        )
                            
                    except Exception as e:
                        self.logger.warning(f"Status check error for {pair}: {e}")
                
                self.logger.info("=" * 40)
                
            except Exception as e:
                self.logger.error(f"Scheduled check error: {e}", exc_info=True)
            
            await asyncio.sleep(interval_seconds)
    
    async def _run_position_sync(self):
        """Periodically sync positions with exchange (every 5 minutes)"""
        # Wait 2 minutes before first sync
        await asyncio.sleep(120)
        
        while self._running:
            try:
                # Run sync for all pairs
                sync_results = await full_sync(
                    self.exchange,
                    self.db,
                    list(self.managers.keys()),
                    self.logger
                )
                
                # Log only if there were changes
                synced_count = sum(1 for v in sync_results.values() if v)
                if synced_count > 0:
                    self.logger.debug(f"Position sync complete: {synced_count} pairs checked")
                    
            except Exception as e:
                self.logger.warning(f"Position sync error: {e}")
            
            # Wait 5 minutes
            await asyncio.sleep(300)
    
    async def _run_auto_optimization(self):
        """Run automatic strategy optimization"""
        interval_seconds = self.global_config.auto_optimize.optimize_interval_hours * 3600
        opt_config = self.global_config.auto_optimize
        
        # Wait before first optimization (let grids collect some data)
        initial_wait = min(1800, interval_seconds / 2)  # 30 min or half interval
        await asyncio.sleep(initial_wait)
        
        while self._running:
            try:
                self.logger.info("=" * 50)
                self.logger.info("🔧 Running Auto-Optimization")
                self.logger.info("=" * 50)
                
                optimization_results = []
                
                for pair, manager in self.managers.items():
                    try:
                        # Get market data for analysis
                        klines = await self.exchange.get_klines(
                            pair, "1H",
                            limit=self.global_config.auto_adjust.volatility_lookback_hours
                        )
                        
                        if not klines:
                            self.logger.warning(f"{pair}: No market data for optimization")
                            continue
                        
                        # Analyze market conditions
                        market = self.optimizer.analyze_market(klines, pair)
                        
                        # Get grid performance
                        grid = self.db.get_active_grid(pair)
                        performance = None
                        if grid:
                            performance = self.optimizer.analyze_grid_performance(grid)
                        
                        # Get current config as dict
                        current_config = {
                            "grid_mode": manager.config.grid_mode,
                            "range_percent": manager.config.range_percent,
                            "grid_levels": manager.config.grid_levels,
                            "min_profit_percent": manager.config.min_profit_percent,
                            "grid_spacing": manager.config.grid_spacing
                        }
                        
                        # Generate optimization
                        result = self.optimizer.optimize_grid(
                            pair, current_config, market, performance
                        )
                        optimization_results.append(result)
                        
                        # Log market conditions
                        self.logger.info(
                            f"{pair}: trend={market.trend}({market.trend_strength:.0f}%), "
                            f"vol={market.volatility:.1f}%, "
                            f"recommended={market.recommended_mode}"
                        )
                        
                        # Log performance if available
                        if performance:
                            self.logger.info(
                                f"  Performance: {performance.trades_per_hour:.2f} trades/hr, "
                                f"win={performance.win_rate:.0f}%, "
                                f"P&L/hr=${performance.pnl_per_hour:.2f}"
                            )
                        
                        # Apply optimization if enabled and meets criteria
                        if result.changes:
                            self.logger.info(f"  Suggested: {', '.join(result.changes)}")
                            self.logger.info(f"  Confidence: {result.confidence:.0f}%, "
                                           f"Expected improvement: {result.expected_improvement:.0f}%")
                            
                            should_apply = (
                                opt_config.auto_apply and
                                self.optimizer.should_apply_optimization(result, opt_config.min_confidence)
                            )
                            
                            if should_apply:
                                await self._apply_optimization(pair, manager, result)
                                self.autopilot.record_optimization(pair, result, applied=True)
                            else:
                                self.autopilot.record_optimization(pair, result, applied=False)
                                if not opt_config.auto_apply:
                                    self.logger.info(f"  ⏸️ Auto-apply disabled, skipping")
                                else:
                                    self.logger.info(f"  ⏸️ Below confidence threshold, skipping")
                        else:
                            self.logger.info(f"  ✓ No changes needed")
                        
                    except Exception as e:
                        self.logger.warning(f"Optimization error for {pair}: {e}")
                
                # Log summary
                summary = self.optimizer.get_optimization_summary(optimization_results)
                self.logger.info("-" * 40)
                self.logger.info(f"Optimization complete: {summary['pairs_to_optimize']}/{summary['total_pairs']} pairs optimized")
                self.logger.info("=" * 50)
                
            except Exception as e:
                self.logger.error(f"Auto-optimization error: {e}", exc_info=True)
            
            await asyncio.sleep(interval_seconds)
    
    async def _apply_optimization(self, pair: str, manager: GridManager, result):
        """Apply optimization changes to a grid"""
        try:
            changes_applied = []
            
            # Update manager config (runtime only)
            if "grid_mode" in result.suggested_params:
                new_mode = result.suggested_params["grid_mode"]
                if new_mode != manager.config.grid_mode:
                    manager.config.grid_mode = new_mode
                    changes_applied.append(f"grid_mode={new_mode}")
            
            if "min_profit_percent" in result.suggested_params:
                new_val = result.suggested_params["min_profit_percent"]
                if new_val != manager.config.min_profit_percent:
                    manager.config.min_profit_percent = new_val
                    changes_applied.append(f"min_profit={new_val}")
            
            if "grid_spacing" in result.suggested_params:
                new_val = result.suggested_params["grid_spacing"]
                if new_val != manager.config.grid_spacing:
                    manager.config.grid_spacing = new_val
                    changes_applied.append(f"spacing={new_val}")
            
            # For range and levels changes, trigger rebuild
            needs_rebuild = False
            
            if "range_percent" in result.suggested_params:
                new_range = result.suggested_params["range_percent"]
                current_range = manager.config.range_percent
                if abs(new_range - current_range) > 2:
                    manager.config.range_percent = new_range
                    needs_rebuild = True
                    changes_applied.append(f"range={new_range}%")
            
            if "grid_levels" in result.suggested_params:
                new_levels = result.suggested_params["grid_levels"]
                current_levels = manager.config.grid_levels
                if abs(new_levels - current_levels) >= 3:
                    manager.config.grid_levels = new_levels
                    needs_rebuild = True
                    changes_applied.append(f"levels={new_levels}")
            
            if changes_applied:
                self.logger.info(f"  ✅ Applied: {', '.join(changes_applied)}")
            
            # Rebuild grid if range/levels changed
            if needs_rebuild:
                self.logger.info(f"  🔄 Triggering grid rebuild...")
                await manager.force_rebuild(f"optimization: {result.reason}")
                
        except Exception as e:
            self.logger.error(f"Error applying optimization for {pair}: {e}")
    
    async def _run_ai_analysis(self):
        """Run AI-based market analysis"""
        ai_config = self.global_config.ai
        
        # Determine interval based on faster of news or regime
        interval_seconds = min(
            ai_config.news_update_interval_minutes,
            ai_config.regime_update_interval_minutes
        ) * 60
        
        # Initial wait to let system stabilize
        await asyncio.sleep(60)
        
        while self._running:
            try:
                self.logger.info("=" * 50)
                self.logger.info("🤖 Running AI Analysis")
                self.logger.info("=" * 50)
                
                for pair, manager in self.managers.items():
                    try:
                        # Get market data
                        klines = await self.exchange.get_klines(
                            pair, "1H",
                            limit=self.global_config.auto_adjust.volatility_lookback_hours
                        )
                        
                        if not klines:
                            continue
                        
                        # Get current state
                        bot_state = self.db.get_bot_state(pair)
                        current_price = bot_state.get("last_price", 0)
                        
                        # Calculate volatility
                        closes = [float(k["close"]) for k in klines]
                        if len(closes) > 1:
                            returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 
                                      for i in range(1, len(closes))]
                            import math
                            volatility = math.sqrt(sum(r**2 for r in returns) / len(returns)) * math.sqrt(24)
                        else:
                            volatility = 5.0
                        
                        # Get trend info
                        trend_info = {
                            "direction": bot_state.get("trend_direction", "unknown"),
                            "strength": bot_state.get("trend_strength", 50)
                        }
                        
                        # Get AI recommendation
                        old_recommendation = self.ai_advisor.get_cached_analysis(pair)
                        recommendation = await self.ai_advisor.get_trading_recommendation(
                            pair, klines, current_price, volatility, trend_info
                        )
                        
                        # Log results
                        final = recommendation.get("final_recommendation", {})
                        sentiment = recommendation.get("sentiment", {})
                        regime = recommendation.get("regime", {})
                        
                        self.logger.info(f"{pair}:")
                        if sentiment:
                            self.logger.info(f"  📰 Sentiment: {sentiment.get('sentiment', 'N/A')} "
                                           f"(score: {sentiment.get('score', 0):.2f})")
                        if regime:
                            self.logger.info(f"  📊 Regime: {regime.get('regime', 'N/A')} "
                                           f"(conf: {regime.get('confidence', 0):.2f})")
                        self.logger.info(f"  🎯 Recommendation: {final.get('grid_mode', 'neutral')} "
                                       f"(conf: {final.get('confidence', 0):.2f})")
                        
                        # Apply changes if enabled and confident
                        if (ai_config.auto_adjust_mode and 
                            final.get("confidence", 0) >= ai_config.min_confidence):
                            
                            current_mode = manager.config.grid_mode
                            suggested_mode = final.get("grid_mode", "neutral")
                            
                            if current_mode != suggested_mode:
                                self.logger.info(f"  🔄 AI changing mode: {current_mode} → {suggested_mode}")
                                manager.config.grid_mode = suggested_mode
                                
                                # Notify
                                await self.ai_advisor.notify_significant_change(
                                    pair, 
                                    {"final_recommendation": {"grid_mode": current_mode}},
                                    recommendation
                                )
                        
                        # Apply range adjustment if enabled
                        if (ai_config.auto_adjust_range and 
                            final.get("confidence", 0) >= ai_config.min_confidence):
                            
                            range_mult = final.get("range_multiplier", 1.0)
                            if abs(range_mult - 1.0) > 0.1:
                                new_range = manager.config.range_percent * range_mult
                                new_range = max(5, min(30, new_range))  # Clamp
                                
                                if abs(new_range - manager.config.range_percent) > 2:
                                    self.logger.info(f"  📏 AI adjusting range: "
                                                   f"{manager.config.range_percent:.1f}% → {new_range:.1f}%")
                                    manager.config.range_percent = new_range
                        
                    except Exception as e:
                        self.logger.warning(f"AI analysis error for {pair}: {e}")
                
                self.logger.info("=" * 50)
                
            except Exception as e:
                self.logger.error(f"AI analysis error: {e}", exc_info=True)
            
            await asyncio.sleep(interval_seconds)
    
    async def stop(self):
        """Stop the bot gracefully"""
        self.logger.info("Stopping bot...")
        self._running = False
        
        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        
        # Wait for tasks to finish
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        
        self.logger.info("Bot stopped")
    
    def get_status(self) -> Dict:
        """Get bot status"""
        status = {
            "running": self._running,
            "mode": "paper" if self.global_config.paper_trading else "live",
            "pairs": {}
        }
        
        for pair, manager in self.managers.items():
            grid = self.db.get_active_grid(pair)
            bot_state = self.db.get_bot_state(pair)
            
            grid_info = None
            if grid:
                grid_info = {
                    "id": grid.id,
                    "range": f"{grid.lower_price:.2f} - {grid.upper_price:.2f}",
                    "levels": grid.grid_levels,
                    "mode": grid.grid_mode,
                    "net_position": grid.net_position,
                    "realized_pnl": grid.realized_pnl,
                    "unrealized_pnl": grid.unrealized_pnl,
                    "total_trades": grid.total_trades
                }
            
            status["pairs"][pair] = {
                "enabled": manager.config.enabled,
                "grid_mode": manager.config.grid_mode,
                "trend": bot_state.get("trend_direction"),
                "last_price": bot_state.get("last_price"),
                "is_paused": bool(bot_state.get("is_paused")),
                "grid": grid_info
            }
        
        return status


async def main():
    # Load configs
    try:
        global_config = load_global_config("config.yaml")
    except FileNotFoundError:
        print("Error: config.yaml not found!")
        print("Please create config.yaml with your API credentials.")
        return
    
    pair_configs = load_all_pair_configs("configs")
    
    if not pair_configs:
        print("Error: No pair configs found in configs/ directory!")
        print("Please create at least one pair config file (e.g., configs/BTCUSDT.yaml)")
        return
    
    # Create bot
    bot = GridBot(global_config, pair_configs)
    
    # Setup signal handlers
    loop = asyncio.get_event_loop()
    
    def signal_handler():
        asyncio.create_task(bot.stop())
    
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)
    
    # Run bot
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())
