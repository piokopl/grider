"""
Web Dashboard for Grid Bot
Provides real-time monitoring and control interface
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, Optional
from aiohttp import web
import logging

from database import Database, Grid
from exchange import BitgetExchange
from grid_manager_v2 import GridManager

# Import new compact dashboard
try:
    from dashboard_template import DASHBOARD_HTML
except ImportError:
    DASHBOARD_HTML = None


class WebServer:
    def __init__(self, db: Database, exchange: BitgetExchange,
                 managers: Dict[str, GridManager], logger: logging.Logger,
                 port: int = 80):
        self.db = db
        self.exchange = exchange
        self.managers = managers
        self.logger = logger
        self.port = port
        self.app = web.Application()
        self._setup_routes()
    
    def _setup_routes(self):
        self.app.router.add_get('/', self.handle_dashboard)
        self.app.router.add_get('/old', self.handle_dashboard_old)  # Old dashboard
        self.app.router.add_get('/api/status', self.handle_status)
        self.app.router.add_get('/api/grids', self.handle_grids)
        self.app.router.add_get('/api/grid/{pair}', self.handle_grid_detail)
        self.app.router.add_get('/api/levels/{pair}', self.handle_levels)
        self.app.router.add_get('/api/orders/{pair}', self.handle_orders)
        self.app.router.add_get('/api/history', self.handle_history)
        self.app.router.add_get('/api/recent_orders', self.handle_recent_orders)
        self.app.router.add_get('/api/stats', self.handle_stats)
        self.app.router.add_get('/api/optimization/{pair}', self.handle_optimization_status)
        self.app.router.add_post('/api/grid/{pair}/stop', self.handle_stop_grid)
        self.app.router.add_post('/api/grid/{pair}/start', self.handle_start_grid)
        self.app.router.add_post('/api/grid/{pair}/rebuild', self.handle_rebuild_grid)
        self.app.router.add_post('/api/grid/{pair}/cancel_orders', self.handle_cancel_orders)
        self.app.router.add_post('/api/pause/{pair}', self.handle_pause)
        self.app.router.add_post('/api/resume/{pair}', self.handle_resume)
        
        # New endpoints for advanced features
        self.app.router.add_get('/api/risk', self.handle_risk_status)
        self.app.router.add_get('/api/statistics', self.handle_statistics)
        self.app.router.add_get('/api/time_rules', self.handle_time_rules)
        self.app.router.add_get('/api/backups', self.handle_backups)
        self.app.router.add_get('/api/ai/{pair}', self.handle_ai_status)
        self.app.router.add_get('/api/ai', self.handle_all_ai_status)  # All AI analyses
        self.app.router.add_post('/api/backup/create', self.handle_create_backup)
        self.app.router.add_post('/api/backup/restore', self.handle_restore_backup)
        
        # Exchange sync endpoints (real data from Bitget)
        self.app.router.add_get('/api/exchange/positions', self.handle_exchange_positions)
        self.app.router.add_get('/api/exchange/orders', self.handle_exchange_orders)
        self.app.router.add_get('/api/exchange/balance', self.handle_exchange_balance)
        self.app.router.add_get('/api/exchange/today_pnl', self.handle_today_pnl)
        self.app.router.add_get('/api/exchange/compare', self.handle_exchange_compare)
        self.app.router.add_get('/api/exchange/sync_status', self.handle_sync_status)
        self.app.router.add_post('/api/exchange/force_sync', self.handle_force_sync)
        
        # Enhanced Grid Features (Phase 1-4)
        self.app.router.add_get('/api/enhanced/status', self.handle_enhanced_status)
        self.app.router.add_get('/api/enhanced/analysis/{pair}', self.handle_enhanced_analysis)
        self.app.router.add_get('/api/enhanced/indicators/{pair}', self.handle_enhanced_indicators)
        self.app.router.add_get('/api/enhanced/tuning', self.handle_auto_tuning)
        self.app.router.add_get('/api/enhanced/rankings', self.handle_pair_rankings)
        self.app.router.add_post('/api/enhanced/force_analysis/{pair}', self.handle_force_analysis)
        
        # Initialize enhanced bot reference (will be set from main.py)
        self.enhanced_bot = None
        self.ai_advisor = None  # Will be set from main.py
        self.exchange_sync = None  # Will be set from main.py
    
    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(runner, '0.0.0.0', self.port)
        await site.start()
        self.logger.info(f"Web dashboard started on port {self.port}")
    
    # === API Handlers ===
    
    async def handle_status(self, request):
        """Get overall bot status"""
        active_count = 0
        
        status = {
            "timestamp": datetime.utcnow().isoformat(),
            "paper_trading": self.exchange.paper_trading,
            "pairs": {}
        }
        
        for pair, manager in self.managers.items():
            grid = self.db.get_active_grid(pair)
            bot_state = self.db.get_bot_state(pair)
            
            if grid and grid.status == 'active':
                active_count += 1
            
            status["pairs"][pair] = {
                "enabled": manager.config.enabled,
                "has_grid": grid is not None,
                "grid_mode": manager.config.grid_mode,
                "trend": bot_state.get("trend_direction"),
                "last_price": bot_state.get("last_price"),
                "is_paused": bool(bot_state.get("is_paused"))
            }
            
            if grid:
                status["pairs"][pair]["grid"] = {
                    "id": grid.id,
                    "status": grid.status,
                    "range": [grid.lower_price, grid.upper_price],
                    "levels": grid.grid_levels,
                    "net_position": round(grid.net_position, 6),
                    "realized_pnl": round(grid.realized_pnl, 2),
                    "unrealized_pnl": round(grid.unrealized_pnl, 2),
                    "total_trades": grid.total_trades,
                    "grid_profits": grid.grid_profits
                }
        
        status["active_pairs"] = active_count
        
        return web.json_response(status)
    
    async def handle_grids(self, request):
        """Get all active grids"""
        grids = []
        for pair in self.managers.keys():
            grid = self.db.get_active_grid(pair)
            if grid:
                # Get current price from bot state
                bot_state = self.db.get_bot_state(pair)
                current_price = bot_state.get("last_price", 0) if bot_state else 0
                
                grids.append({
                    "pair": pair,
                    "id": grid.id,
                    "status": grid.status,
                    "mode": grid.grid_mode,
                    # Names matching dashboard expectations
                    "range_low": grid.lower_price,
                    "range_high": grid.upper_price,
                    "current_price": current_price,
                    "price": current_price,  # Alias for formatting logic
                    "levels": grid.grid_levels,
                    "capital": grid.total_capital,
                    "net_position": grid.net_position,
                    "realized_profit": grid.realized_pnl,  # Dashboard name
                    "realized_pnl": grid.realized_pnl,
                    "unrealized_pnl": grid.unrealized_pnl,
                    "total_pnl": grid.realized_pnl + grid.unrealized_pnl,
                    "total_trades": grid.total_trades,
                    "created_at": grid.created_at
                })
        # Sort alphabetically by pair
        grids.sort(key=lambda x: x["pair"])
        return web.json_response(grids)
    
    async def handle_grid_detail(self, request):
        """Get detailed grid information"""
        pair = request.match_info['pair'].upper()
        
        if pair not in self.managers:
            return web.json_response({"error": "Pair not found"}, status=404)
        
        manager = self.managers[pair]
        info = manager.get_grid_info()
        
        if not info:
            return web.json_response({"error": "No active grid"}, status=404)
        
        return web.json_response(info)
    
    async def handle_levels(self, request):
        """Get grid levels for a pair"""
        pair = request.match_info['pair'].upper()
        
        grid = self.db.get_active_grid(pair)
        if not grid:
            return web.json_response({"error": "No active grid"}, status=404)
        
        levels = self.db.get_grid_levels(grid.id)
        
        # Get current price
        bot_state = self.db.get_bot_state(pair)
        current_price = bot_state.get("last_price", 0)
        
        levels_data = []
        for level in levels:
            level_data = {
                "index": level.level_index,
                "price": level.price,
                "buy_status": level.buy_status,
                "sell_status": level.sell_status,
                "total_buys": level.total_buys,
                "total_sells": level.total_sells,
                "pnl": round(level.level_pnl, 4),
                "is_current": (level.level_index < len(levels) - 1 and 
                              levels[level.level_index].price <= current_price < 
                              levels[level.level_index + 1].price if level.level_index < len(levels) - 1 else False)
            }
            levels_data.append(level_data)
        
        return web.json_response({
            "grid_id": grid.id,
            "current_price": current_price,
            "levels": levels_data
        })
    
    async def handle_orders(self, request):
        """Get recent orders for a pair"""
        pair = request.match_info['pair'].upper()
        limit = int(request.query.get('limit', 50))
        
        grid = self.db.get_active_grid(pair)
        if not grid:
            return web.json_response({"error": "No active grid"}, status=404)
        
        orders = self.db.get_orders_by_grid(grid.id, limit)
        
        orders_data = [{
            "id": o.id,
            "order_id": o.order_id,
            "type": o.order_type,
            "side": o.side,
            "price": o.price,
            "amount": o.amount,
            "status": o.status,
            "created_at": o.created_at,
            "filled_at": o.filled_at,
            "fill_price": o.fill_price,
            "fee": o.fee
        } for o in orders]
        
        return web.json_response(orders_data)
    
    async def handle_history(self, request):
        """Get grid history"""
        pair = request.query.get('pair')
        limit = int(request.query.get('limit', 20))
        
        grids = self.db.get_grids_history(pair.upper() if pair else None, limit)
        
        history = [{
            "id": g.id,
            "pair": g.pair,
            "status": g.status,
            "mode": g.grid_mode,
            "range": f"{g.lower_price:.2f} - {g.upper_price:.2f}",
            "levels": g.grid_levels,
            "capital": g.total_capital,
            "realized_pnl": round(g.realized_pnl, 2),
            "total_trades": g.total_trades,
            "grid_profits": g.grid_profits,
            "created_at": g.created_at,
            "stopped_at": g.stopped_at,
            "stop_reason": g.stop_reason
        } for g in grids]
        
        # Sort: active first, then alphabetically by pair
        history.sort(key=lambda x: (x["status"] != "active", x["pair"]))
        
        return web.json_response(history)
    
    async def handle_recent_orders(self, request):
        """Get recent filled orders from cache"""
        limit = int(request.query.get('limit', 20))
        
        try:
            # Read from database cache (fast!)
            fills = self.db.get_exchange_fills(limit)
            return web.json_response(fills)
        except Exception as e:
            # Fallback to database grid_orders
            orders = self.db.get_recent_filled_orders(limit)
            return web.json_response(orders)
    
    async def handle_stats(self, request):
        """Get overall statistics - comprehensive dashboard data"""
        pair = request.query.get('pair')
        
        # Basic stats from database
        db_stats = self.db.get_all_time_stats(pair.upper() if pair else None)
        
        # Calculate additional metrics from active grids
        total_profit = 0
        today_profit = 0
        today_trades = 0
        open_positions = 0
        open_value = 0
        pending_orders = 0
        pending_buys = 0
        pending_sells = 0
        winning_trades = 0
        losing_trades = 0
        total_capital = 0
        
        from datetime import datetime, timedelta
        today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        
        for p, manager in self.managers.items():
            if pair and p != pair.upper():
                continue
            
            grid = self.db.get_active_grid(p)
            if grid:
                total_profit += grid.realized_pnl
                total_capital += grid.total_capital
                
                # Count positions
                if abs(grid.net_position) > 0.00001:
                    open_positions += 1
                    bot_state = self.db.get_bot_state(p)
                    if bot_state:
                        price = bot_state.get("last_price", 0)
                        open_value += abs(grid.net_position) * price
                
                # Count winning and losing round trips
                # grid_profits = successful round trips (profit > 0)
                # We estimate total rounds from total_trades (each round = 2 trades: buy + sell)
                total_rounds = grid.total_trades // 2  # Approximate number of completed rounds
                winning = grid.grid_profits or 0
                losing = max(0, total_rounds - winning)
                
                winning_trades += winning
                losing_trades += losing
        
        # Get open orders count from cache
        try:
            orders_cache = self.db.get_exchange_orders_count()
            pending_orders = orders_cache.get('total_orders', 0)
            pending_buys = orders_cache.get('buy_orders', 0)
            pending_sells = orders_cache.get('sell_orders', 0)
        except Exception as e:
            # Fallback to database count if cache fails
            for p, manager in self.managers.items():
                grid = self.db.get_active_grid(p)
                if grid:
                    levels = self.db.get_grid_levels(grid.id)
                    for level in levels:
                        if level.buy_status == 'pending':
                            pending_orders += 1
                            pending_buys += 1
                        if level.sell_status == 'pending':
                            pending_orders += 1
                            pending_sells += 1
        
        # Get today's PnL directly from Bitget API (accurate with date filtering)
        try:
            today_data = await self.exchange.get_today_pnl()
            today_trades = today_data.get('today_trades', 0)
            today_profit = today_data.get('today_pnl', 0)
        except Exception as e:
            # Fallback
            today_trades = 0
            today_profit = 0
        
        # Calculate profit percentage
        total_profit_pct = (total_profit / total_capital * 100) if total_capital > 0 else 0
        
        return web.json_response({
            # Dashboard expected fields
            'total_profit': round(total_profit, 2),
            'total_profit_pct': round(total_profit_pct, 2),
            'today_profit': round(today_profit, 2),
            'today_trades': today_trades,
            'open_positions': open_positions,
            'open_value': round(open_value, 2),
            'pending_orders': pending_orders,
            'pending_buys': pending_buys,
            'pending_sells': pending_sells,
            'total_trades': winning_trades + losing_trades,  # Total rounds (not individual trades)
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            # Original fields
            'total_grids': db_stats['total_grids'],
            'active_grids': db_stats['active_grids'],
            'total_realized_pnl': db_stats['total_realized_pnl'],
            'avg_grid_pnl': round(db_stats['avg_grid_pnl'], 2)
        })
    
    async def handle_optimization_status(self, request):
        """Get optimization info for a pair"""
        pair = request.match_info['pair'].upper()
        
        if pair not in self.managers:
            return web.json_response({"error": "Pair not found"}, status=404)
        
        manager = self.managers[pair]
        grid = self.db.get_active_grid(pair)
        
        # Get current config
        current_config = {
            "grid_mode": manager.config.grid_mode,
            "range_percent": manager.config.range_percent,
            "grid_levels": manager.config.grid_levels,
            "min_profit_percent": manager.config.min_profit_percent,
            "grid_spacing": manager.config.grid_spacing
        }
        
        # Get performance metrics if grid exists
        performance = None
        if grid:
            from datetime import datetime
            created = datetime.fromisoformat(grid.created_at)
            duration_hours = (datetime.utcnow() - created).total_seconds() / 3600
            
            performance = {
                "grid_id": grid.id,
                "duration_hours": round(duration_hours, 2),
                "total_trades": grid.total_trades,
                "trades_per_hour": round(grid.total_trades / duration_hours, 2) if duration_hours > 0 else 0,
                "realized_pnl": round(grid.realized_pnl, 2),
                "unrealized_pnl": round(grid.unrealized_pnl, 2),
                "grid_profits": grid.grid_profits,
                "win_rate": round(grid.grid_profits / grid.total_trades * 100, 1) if grid.total_trades > 0 else 0
            }
        
        return web.json_response({
            "pair": pair,
            "current_config": current_config,
            "performance": performance,
            "auto_optimize_enabled": True  # Would need access to global config
        })
    
    async def handle_stop_grid(self, request):
        """Stop a grid"""
        pair = request.match_info['pair'].upper()
        
        if pair not in self.managers:
            return web.json_response({"error": "Pair not found"}, status=404)
        
        try:
            data = await request.json()
            close_position = data.get('close_position', True)
        except:
            close_position = True
        
        manager = self.managers[pair]
        await manager.stop_grid_manual(close_position)
        
        return web.json_response({"success": True, "message": f"Grid stopped for {pair}"})
    
    async def handle_start_grid(self, request):
        """Start a new grid"""
        pair = request.match_info['pair'].upper()
        
        if pair not in self.managers:
            return web.json_response({"error": "Pair not found"}, status=404)
        
        # Check if grid already exists
        existing = self.db.get_active_grid(pair)
        if existing:
            return web.json_response({"error": "Grid already active"}, status=400)
        
        # Grid will be created on next update cycle
        return web.json_response({"success": True, "message": f"Grid will start for {pair}"})
    
    async def handle_rebuild_grid(self, request):
        """Force rebuild grid with new range"""
        pair = request.match_info['pair'].upper()
        
        if pair not in self.managers:
            return web.json_response({"error": "Pair not found"}, status=404)
        
        manager = self.managers[pair]
        
        # Check if grid exists
        existing = self.db.get_active_grid(pair)
        if not existing:
            return web.json_response({"error": "No active grid to rebuild"}, status=400)
        
        try:
            new_grid = await manager.force_rebuild("manual_api")
            if new_grid:
                return web.json_response({
                    "success": True, 
                    "message": f"Grid rebuilt for {pair}",
                    "new_range": [new_grid.lower_price, new_grid.upper_price]
                })
            else:
                return web.json_response({"error": "Failed to rebuild grid"}, status=500)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_cancel_orders(self, request):
        """Cancel all orders for a pair"""
        pair = request.match_info['pair'].upper()
        
        if pair not in self.managers:
            return web.json_response({"error": "Pair not found"}, status=404)
        
        manager = self.managers[pair]
        await manager.cancel_all_orders()
        
        return web.json_response({"success": True, "message": f"Orders cancelled for {pair}"})
    
    async def handle_pause(self, request):
        """Pause trading for a pair"""
        pair = request.match_info['pair'].upper()
        self.db.update_bot_state(pair, is_paused=1)
        return web.json_response({"success": True, "paused": True})
    
    async def handle_resume(self, request):
        """Resume trading for a pair"""
        pair = request.match_info['pair'].upper()
        self.db.update_bot_state(pair, is_paused=0)
        return web.json_response({"success": True, "paused": False})
    
    async def handle_risk_status(self, request):
        """Get current risk status"""
        # This would integrate with RiskManager
        return web.json_response({
            "risk_level": "low",
            "daily_pnl": 0,
            "daily_trades": 0,
            "max_drawdown": 0,
            "paused_pairs": [],
            "message": "Risk manager not yet integrated"
        })
    
    async def handle_statistics(self, request):
        """Get comprehensive statistics"""
        stats = {
            "overview": {
                "total_trades": 0,
                "total_pnl": 0,
                "win_rate": 0
            },
            "by_pair": {}
        }
        
        # Gather from all grids
        for pair in self.managers.keys():
            all_grids = self.db.get_grids_history(pair, limit=100)
            pair_stats = {
                "trades": sum(g.total_trades for g in all_grids),
                "pnl": sum(g.realized_pnl for g in all_grids),
                "grids": len(all_grids)
            }
            stats["by_pair"][pair] = pair_stats
            stats["overview"]["total_trades"] += pair_stats["trades"]
            stats["overview"]["total_pnl"] += pair_stats["pnl"]
        
        return web.json_response(stats)
    
    async def handle_time_rules(self, request):
        """Get time-based rules status"""
        from datetime import datetime
        now = datetime.utcnow()
        
        # Determine current session
        hour = now.hour
        weekday = now.weekday()
        
        if weekday >= 5:
            session = "weekend"
        elif 0 <= hour < 8:
            session = "asia"
        elif 7 <= hour < 16:
            session = "europe"
        elif 13 <= hour < 21:
            session = "us"
        else:
            session = "off_hours"
        
        return web.json_response({
            "current_time": now.isoformat(),
            "current_session": session,
            "modifiers": {
                "position": 1.0,
                "range": 1.0
            },
            "next_session_change": "N/A"
        })
    
    async def handle_backups(self, request):
        """List available backups"""
        import os
        backup_dir = "backups"
        backups = []
        
        if os.path.exists(backup_dir):
            for f in os.listdir(backup_dir):
                if f.endswith('.zip'):
                    filepath = os.path.join(backup_dir, f)
                    backups.append({
                        "filename": f,
                        "size_mb": round(os.path.getsize(filepath) / 1024 / 1024, 2),
                        "created": os.path.getmtime(filepath)
                    })
        
        return web.json_response({"backups": sorted(backups, key=lambda x: x["created"], reverse=True)})
    
    async def handle_ai_status(self, request):
        """Get AI analysis status for a pair"""
        pair = request.match_info['pair'].upper()
        
        # Check if AI advisor is available
        if not hasattr(self, 'ai_advisor') or self.ai_advisor is None:
            return web.json_response({
                "pair": pair,
                "ai_enabled": False,
                "message": "AI advisor not initialized"
            })
        
        try:
            # Get cached analysis from AI advisor
            cached = self.ai_advisor.get_cached_analysis(pair)
            
            if cached:
                # Extract data from cached analysis
                sentiment = cached.get("sentiment", {})
                regime = cached.get("regime", {})
                final = cached.get("final_recommendation", {})
                
                return web.json_response({
                    "pair": pair,
                    "ai_enabled": True,
                    "timestamp": cached.get("timestamp"),
                    "sentiment": {
                        "value": sentiment.get("sentiment") if sentiment else None,
                        "score": sentiment.get("score") if sentiment else 0,
                        "confidence": sentiment.get("confidence") if sentiment else 0,
                        "summary": sentiment.get("summary") if sentiment else None
                    } if sentiment else None,
                    "regime": {
                        "value": regime.get("regime") if regime else None,
                        "confidence": regime.get("confidence") if regime else 0,
                        "characteristics": regime.get("characteristics") if regime else [],
                        "reasoning": regime.get("reasoning") if regime else None
                    } if regime else None,
                    "final_recommendation": {
                        "grid_mode": final.get("grid_mode", "neutral"),
                        "range_multiplier": final.get("range_multiplier", 1.0),
                        "confidence": final.get("confidence", 0.5),
                        "reasoning": final.get("reasoning", [])
                    }
                })
            else:
                return web.json_response({
                    "pair": pair,
                    "ai_enabled": True,
                    "message": "No cached analysis available yet",
                    "final_recommendation": {
                        "grid_mode": "neutral",
                        "confidence": 0.5
                    }
                })
                
        except Exception as e:
            self.logger.warning(f"Error getting AI status for {pair}: {e}")
            return web.json_response({
                "pair": pair,
                "ai_enabled": True,
                "error": str(e),
                "final_recommendation": {
                    "grid_mode": "neutral",
                    "confidence": 0.5
                }
            })
    
    async def handle_all_ai_status(self, request):
        """Get AI analysis for all pairs at once"""
        if not hasattr(self, 'ai_advisor') or self.ai_advisor is None:
            return web.json_response({
                "ai_enabled": False,
                "analyses": {}
            })
        
        analyses = {}
        
        for pair in self.managers.keys():
            try:
                cached = self.ai_advisor.get_cached_analysis(pair)
                
                if cached:
                    sentiment = cached.get("sentiment", {})
                    regime = cached.get("regime", {})
                    final = cached.get("final_recommendation", {})
                    
                    analyses[pair] = {
                        "timestamp": cached.get("timestamp"),
                        "sentiment": sentiment.get("sentiment") if sentiment else None,
                        "score": sentiment.get("score", 0) if sentiment else 0,
                        "regime": regime.get("regime") if regime else None,
                        "grid_mode": final.get("grid_mode", "neutral"),
                        "confidence": final.get("confidence", 0.5),
                        "summary": sentiment.get("summary", "") if sentiment else "",
                        "reasoning": final.get("reasoning", [])
                    }
                else:
                    analyses[pair] = {
                        "sentiment": None,
                        "score": 0,
                        "regime": None,
                        "grid_mode": "neutral",
                        "confidence": 0.5
                    }
            except Exception as e:
                analyses[pair] = {
                    "error": str(e),
                    "grid_mode": "neutral",
                    "confidence": 0.5
                }
        
        return web.json_response({
            "ai_enabled": True,
            "analyses": analyses
        })
    
    async def handle_exchange_positions(self, request):
        """Get REAL positions from Bitget exchange"""
        try:
            positions = await self.exchange.get_real_positions()
            return web.json_response({
                "source": "bitget",
                "count": len(positions),
                "positions": [
                    {
                        "pair": p.pair,
                        "side": p.side,
                        "size": p.size,
                        "entry_price": p.entry_price,
                        "unrealized_pnl": round(p.unrealized_pnl, 2),
                        "leverage": p.leverage,
                        "margin_mode": p.margin_mode
                    }
                    for p in positions
                ]
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_exchange_orders(self, request):
        """Get REAL open orders from Bitget exchange"""
        try:
            orders = await self.exchange.get_real_open_orders()
            return web.json_response({
                "source": "bitget",
                "count": len(orders),
                "orders": [
                    {
                        "order_id": o.order_id,
                        "pair": o.pair,
                        "side": o.side,
                        "price": o.price,
                        "amount": o.amount,
                        "filled": o.filled_amount,
                        "status": o.status
                    }
                    for o in orders
                ]
            })
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_exchange_balance(self, request):
        """Get account balance from cache (synced from Bitget)"""
        try:
            # Read from database cache (fast!)
            balance = self.db.get_exchange_balance()
            cache_age = self.db.get_exchange_cache_age()
            
            return web.json_response({
                "total": balance.get("total", 0),
                "available": balance.get("available", 0),
                "unrealized_pnl": balance.get("unrealized_pnl", 0),
                "used_margin": balance.get("used_margin", 0),
                "wallet_balance": balance.get("wallet_balance", 0),
                "roi_percent": balance.get("roi_percent", 0),
                "bonus": balance.get("bonus", 0),
                "source": "cache",
                "cache_age_seconds": cache_age
            })
        except Exception as e:
            return web.json_response({"error": str(e), "total": 0, "available": 0}, status=500)
    
    async def handle_today_pnl(self, request):
        """Get today's PnL from Bitget fills"""
        try:
            data = await self.exchange.get_today_pnl()
            return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e), "today_pnl": 0, "today_trades": 0}, status=500)
    
    async def handle_sync_status(self, request):
        """Get exchange sync task status"""
        if hasattr(self, 'exchange_sync') and self.exchange_sync:
            status = self.exchange_sync.get_status()
            status['cache_age'] = self.db.get_exchange_cache_age()
            return web.json_response(status)
        else:
            return web.json_response({
                'running': False,
                'message': 'Exchange sync not initialized'
            })
    
    async def handle_force_sync(self, request):
        """Force immediate sync from exchange"""
        if hasattr(self, 'exchange_sync') and self.exchange_sync:
            await self.exchange_sync.force_sync()
            return web.json_response({'success': True, 'message': 'Sync completed'})
        else:
            return web.json_response({'success': False, 'message': 'Sync not available'})
    
    async def handle_exchange_compare(self, request):
        """Compare bot's local state vs exchange state (from cache)"""
        try:
            # Get data from cache (fast!)
            cached_positions = self.db.get_exchange_positions()
            cached_orders = self.db.get_exchange_orders_count()
            cached_balance = self.db.get_exchange_balance()
            cache_age = self.db.get_exchange_cache_age()
            
            # Create lookup for exchange positions
            exchange_pos_lookup = {p['pair']: p for p in cached_positions}
            
            # Get local bot state - but use EXCHANGE side if available
            local_positions = []
            
            for pair, manager in self.managers.items():
                grid = self.db.get_active_grid(pair)
                if grid and abs(grid.net_position) > 0.00001:
                    # Check if we have exchange data for this pair
                    exchange_pos = exchange_pos_lookup.get(pair)
                    
                    if exchange_pos:
                        # Use exchange data (more accurate)
                        local_positions.append({
                            "pair": pair,
                            "side": exchange_pos['side'],  # Use EXCHANGE side!
                            "size": exchange_pos['size'],  # Use EXCHANGE size!
                            "entry": exchange_pos['entry_price'],
                            "pnl": round(exchange_pos['unrealized_pnl'], 2),
                            "synced": True
                        })
                    else:
                        # No exchange data - use bot's estimate
                        local_positions.append({
                            "pair": pair,
                            "side": "long" if grid.net_position > 0 else "short",
                            "size": abs(grid.net_position),
                            "entry": grid.average_buy_price if grid.net_position > 0 else grid.average_sell_price,
                            "pnl": round(grid.unrealized_pnl, 2),
                            "synced": False
                        })
            
            # Build comparison
            comparison = {
                "mode": "live",
                "cache_age_seconds": cache_age,
                "real": {
                    "positions": len(cached_positions),
                    "open_orders": cached_orders.get('total_orders', 0),
                    "balance": cached_balance,
                    "position_details": [
                        {
                            "pair": p['pair'],
                            "side": p['side'],
                            "size": p['size'],
                            "entry": p['entry_price'],
                            "pnl": round(p['unrealized_pnl'], 2)
                        }
                        for p in sorted(cached_positions, key=lambda x: x['pair'])
                    ]
                },
                "local": {
                    "positions": len(local_positions),
                    "open_orders": 0,
                    "position_details": sorted(local_positions, key=lambda x: x['pair'])
                },
                "differences": []
            }
            
            # Find differences
            real_pairs = {p['pair'] for p in cached_positions}
            local_pairs = {p['pair'] for p in local_positions}
            
            for pair in real_pairs - local_pairs:
                comparison["differences"].append({
                    "type": "only_on_exchange",
                    "pair": pair,
                    "message": f"{pair} has position on exchange but not tracked"
                })
            
            for pair in local_pairs - real_pairs:
                comparison["differences"].append({
                    "type": "only_in_local",
                    "pair": pair,
                    "message": f"{pair} tracked but no position on exchange"
                })
            
            return web.json_response(comparison)
            
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_create_backup(self, request):
        """Create a new backup"""
        try:
            from backup import BackupManager
            backup_mgr = BackupManager(self.logger)
            
            data = await request.json() if request.body_exists else {}
            notes = data.get("notes", "")
            
            path = backup_mgr.create_backup(notes=notes)
            return web.json_response({"success": True, "path": path})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_restore_backup(self, request):
        """Restore from backup"""
        try:
            data = await request.json()
            backup_path = data.get("path")
            
            if not backup_path:
                return web.json_response({"error": "No backup path provided"}, status=400)
            
            from backup import BackupManager
            backup_mgr = BackupManager(self.logger)
            
            success = backup_mgr.restore_backup(backup_path)
            return web.json_response({"success": success})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)
    
    # === Enhanced Grid Feature Handlers (Phase 1-4) ===
    
    async def handle_enhanced_status(self, request):
        """Get comprehensive enhanced grid status"""
        try:
            if not hasattr(self, 'enhanced_bot') or self.enhanced_bot is None:
                return web.json_response({
                    "enabled": False,
                    "message": "Enhanced bot not initialized"
                })
            
            data = self.enhanced_bot.get_dashboard_data()
            return web.json_response(data)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced status: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_enhanced_analysis(self, request):
        """Get detailed market analysis for a pair"""
        pair = request.match_info['pair'].upper()
        
        try:
            if not hasattr(self, 'enhanced_bot') or self.enhanced_bot is None:
                # Fallback to basic analysis
                from advanced_indicators import AdvancedIndicators
                indicators = AdvancedIndicators(self.logger)
                
                klines = await self.exchange.get_klines(pair, "1h", limit=100)
                if not klines:
                    return web.json_response({"error": "No market data"}, status=400)
                
                closes = [k["close"] for k in klines]
                current_price = closes[-1]
                
                # Calculate indicators
                atr_percent = indicators.calculate_atr_percent(klines, 14)
                adx_data = indicators.calculate_adx_full(klines, 14)
                rsi_result = indicators.calculate_rsi(closes, 14)
                macd_result = indicators.calculate_macd(closes)
                bb_lower, bb_middle, bb_upper = indicators.calculate_bollinger_bands(closes, 20, 2.0)
                
                analysis = {
                    "pair": pair,
                    "current_price": current_price,
                    "timestamp": datetime.utcnow().isoformat(),
                    "indicators": {
                        "atr_percent": round(atr_percent, 3),
                        "adx": round(adx_data["adx"], 2),
                        "plus_di": round(adx_data["plus_di"], 2),
                        "minus_di": round(adx_data["minus_di"], 2),
                        "trend_strength": adx_data["trend_strength"],
                        "trend_direction": adx_data["trend_direction"],
                        "rsi": round(rsi_result.value, 2),
                        "rsi_signal": rsi_result.signal,
                        "macd_histogram": round(macd_result.value, 4),
                        "macd_signal": macd_result.signal,
                        "bb_lower": round(bb_lower, 2),
                        "bb_middle": round(bb_middle, 2),
                        "bb_upper": round(bb_upper, 2),
                    },
                    "trading_signals": {
                        "trend_pause": adx_data["adx"] > 40,
                        "trend_pause_reason": f"ADX {adx_data['adx']:.1f} > 40" if adx_data["adx"] > 40 else None,
                        "volatility_ok": 0.3 <= atr_percent <= 10.0,
                        "volatility_reason": f"ATR {atr_percent:.2f}%" if atr_percent < 0.3 or atr_percent > 10 else None,
                        "rsi_oversold": rsi_result.value < 30,
                        "rsi_overbought": rsi_result.value > 70,
                        "bb_buy_signal": current_price < bb_lower,
                        "bb_sell_signal": current_price > bb_upper,
                    },
                    "recommendations": {
                        "can_trade": adx_data["adx"] < 40 and 0.3 <= atr_percent <= 10.0,
                        "suggested_mode": "long" if adx_data["plus_di"] > adx_data["minus_di"] else "short" if adx_data["adx"] > 25 else "neutral",
                        "dynamic_range_percent": round(atr_percent * 5, 1),
                    }
                }
                
                return web.json_response(analysis)
            
            # Use enhanced bot
            analysis = await self.enhanced_bot.pre_trade_analysis(pair)
            return web.json_response(analysis)
            
        except Exception as e:
            self.logger.error(f"Error in enhanced analysis: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_enhanced_indicators(self, request):
        """Get all technical indicators for a pair"""
        pair = request.match_info['pair'].upper()
        
        try:
            from advanced_indicators import AdvancedIndicators
            indicators = AdvancedIndicators(self.logger)
            
            # Get multiple timeframes
            results = {}
            for tf in ["15m", "1h", "4h"]:
                klines = await self.exchange.get_klines(pair, tf, limit=100)
                if not klines:
                    continue
                
                closes = [k["close"] for k in klines]
                
                results[tf] = {
                    "atr_percent": round(indicators.calculate_atr_percent(klines, 14), 3),
                    "rsi": round(indicators.calculate_rsi(closes, 14).value, 2),
                    "adx": round(indicators.calculate_adx_full(klines, 14)["adx"], 2),
                    "macd": round(indicators.calculate_macd(closes).value, 4),
                }
            
            # Volume profile from 1h
            klines_1h = await self.exchange.get_klines(pair, "1h", limit=50)
            vol_profile = indicators.calculate_volume_profile(klines_1h) if klines_1h else {}
            
            return web.json_response({
                "pair": pair,
                "timestamp": datetime.utcnow().isoformat(),
                "by_timeframe": results,
                "volume_profile": {
                    "support": round(vol_profile.get("support", 0), 2),
                    "resistance": round(vol_profile.get("resistance", 0), 2),
                    "poc": round(vol_profile.get("poc", 0), 2),
                } if vol_profile else None
            })
            
        except Exception as e:
            self.logger.error(f"Error getting indicators: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_auto_tuning(self, request):
        """Get auto-tuning status and recommendations"""
        try:
            if not hasattr(self, 'enhanced_bot') or self.enhanced_bot is None:
                return web.json_response({
                    "enabled": False,
                    "message": "Enhanced bot not initialized"
                })
            
            if not self.enhanced_bot.auto_tuner:
                return web.json_response({
                    "enabled": False,
                    "message": "Auto-tuner not enabled in config"
                })
            
            report = self.enhanced_bot.get_analysis_report(7)
            return web.json_response(report)
            
        except Exception as e:
            self.logger.error(f"Error in auto-tuning: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_pair_rankings(self, request):
        """Get pair rankings for grid trading suitability"""
        try:
            if hasattr(self, 'enhanced_bot') and self.enhanced_bot and self.enhanced_bot.pair_selector:
                rankings = await self.enhanced_bot.get_pair_rankings()
                return web.json_response({"rankings": rankings})
            
            # Fallback - calculate basic rankings
            from auto_tuner import PairSelector
            selector = PairSelector(self.exchange, self.logger)
            rankings = await selector.rank_pairs()
            
            return web.json_response({"rankings": rankings})
            
        except Exception as e:
            self.logger.error(f"Error getting pair rankings: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def handle_force_analysis(self, request):
        """Force immediate market analysis for a pair"""
        pair = request.match_info['pair'].upper()
        
        try:
            if hasattr(self, 'enhanced_bot') and self.enhanced_bot:
                if pair in self.enhanced_bot.last_analysis_time:
                    del self.enhanced_bot.last_analysis_time[pair]
                
                analysis = await self.enhanced_bot.pre_trade_analysis(pair)
                return web.json_response({
                    "success": True,
                    "pair": pair,
                    "analysis": analysis
                })
            
            return web.json_response({
                "success": False,
                "message": "Enhanced bot not available"
            })
            
        except Exception as e:
            self.logger.error(f"Error forcing analysis: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    # === Dashboard HTML ===
    
    async def handle_dashboard(self, request):
        """Serve the main dashboard (new compact version)"""
        if DASHBOARD_HTML:
            return web.Response(text=DASHBOARD_HTML, content_type='text/html')
        else:
            # Fallback to old dashboard if new one not available
            html = self._generate_dashboard_html()
            return web.Response(text=html, content_type='text/html')
    
    async def handle_dashboard_old(self, request):
        """Serve the old dashboard"""
        html = self._generate_dashboard_html()
        return web.Response(text=html, content_type='text/html')
    
    def _generate_dashboard_html(self) -> str:
        return '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Grid Bot Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }
        h1 { 
            color: #00d4aa; 
            font-size: 28px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00d4aa;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .stats-row {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .stat-label { color: #888; font-size: 12px; text-transform: uppercase; }
        .stat-value { font-size: 28px; font-weight: bold; margin-top: 5px; }
        .stat-value.positive { color: #00d4aa; }
        .stat-value.negative { color: #ff6b6b; }
        
        .grids-section { margin-bottom: 30px; }
        h2 { color: #fff; margin-bottom: 20px; font-size: 20px; }
        
        .grid-card {
            background: rgba(255,255,255,0.05);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.1);
        }
        .grid-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        .pair-name {
            font-size: 24px;
            font-weight: bold;
            color: #fff;
        }
        .grid-status {
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            text-transform: uppercase;
        }
        .grid-status.active { background: rgba(0,212,170,0.2); color: #00d4aa; }
        .grid-status.stopped { background: rgba(255,107,107,0.2); color: #ff6b6b; }
        
        .grid-info {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 20px;
        }
        .info-item { }
        .info-label { color: #888; font-size: 11px; text-transform: uppercase; }
        .info-value { font-size: 16px; margin-top: 3px; }
        
        .levels-visual {
            background: rgba(0,0,0,0.3);
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        .levels-container {
            display: flex;
            gap: 3px;
            height: 60px;
            align-items: flex-end;
        }
        .level-bar {
            flex: 1;
            min-width: 8px;
            border-radius: 2px 2px 0 0;
            transition: all 0.3s;
        }
        .level-bar.buy { background: rgba(0,212,170,0.6); }
        .level-bar.sell { background: rgba(255,193,7,0.6); }
        .level-bar.current { background: #fff; }
        .level-bar:hover { opacity: 0.8; }
        
        .actions {
            display: flex;
            gap: 10px;
            margin-top: 15px;
        }
        button {
            padding: 10px 20px;
            border-radius: 8px;
            border: none;
            cursor: pointer;
            font-size: 14px;
            transition: all 0.2s;
        }
        .btn-stop { background: #ff6b6b; color: #fff; }
        .btn-stop:hover { background: #ff5252; }
        .btn-cancel { background: #ffc107; color: #000; }
        .btn-cancel:hover { background: #ffca2c; }
        .btn-start { background: #00d4aa; color: #000; }
        .btn-start:hover { background: #00e4ba; }
        
        .orders-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .orders-table th, .orders-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .orders-table th {
            color: #888;
            font-size: 11px;
            text-transform: uppercase;
        }
        .order-buy { color: #00d4aa; }
        .order-sell { color: #ff6b6b; }
        .order-filled { color: #ffc107; }
        .order-pending { color: #888; }
        
        .history-section {
            margin-top: 30px;
        }
        .history-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 15px;
        }
        .history-card {
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 15px;
            border: 1px solid rgba(255,255,255,0.05);
        }
        
        .refresh-info {
            color: #666;
            font-size: 12px;
            margin-top: 20px;
            text-align: center;
        }
        
        @media (max-width: 768px) {
            .grid-info { grid-template-columns: repeat(2, 1fr); }
            .stat-value { font-size: 22px; }
        }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>
                <span class="status-dot"></span>
                Grid Bot Dashboard
            </h1>
            <span id="mode-badge" style="padding: 5px 15px; border-radius: 20px; font-size: 12px;"></span>
        </header>
        
        <div class="stats-row" id="stats-row"></div>
        
        <div class="grids-section">
            <h2>Active Grids</h2>
            <div id="grids-container"></div>
        </div>
        
        <div class="history-section">
            <h2>Recent Orders</h2>
            <div id="orders-container"></div>
        </div>
        
        <div class="history-section">
            <h2>🔄 Exchange Sync (Bitget vs Bot State)</h2>
            <div id="exchange-compare" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                <div class="stat-card">
                    <div class="stat-label">📊 Real Exchange (Bitget)</div>
                    <div id="exchange-real" style="margin-top: 10px;">Loading...</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📝 Bot State (Local)</div>
                    <div id="exchange-local" style="margin-top: 10px;">Loading...</div>
                </div>
            </div>
            <div id="exchange-diff" style="margin-top: 15px;"></div>
        </div>
        
        <div class="history-section">
            <h2>Grid History</h2>
            <div class="history-grid" id="history-container"></div>
        </div>
        
        <div class="refresh-info">
            Auto-refresh every 5 seconds | Last update: <span id="last-update">-</span>
        </div>
    </div>

    <script>
        async function fetchData(endpoint) {
            try {
                const response = await fetch(endpoint);
                return await response.json();
            } catch (e) {
                console.error('Fetch error:', e);
                return null;
            }
        }
        
        function formatNumber(n, decimals = 2) {
            if (n === null || n === undefined) return '-';
            return Number(n).toLocaleString(undefined, {
                minimumFractionDigits: decimals,
                maximumFractionDigits: decimals
            });
        }
        
        function formatPnl(n) {
            if (n === null || n === undefined) return '-';
            const sign = n >= 0 ? '+' : '';
            const cls = n >= 0 ? 'positive' : 'negative';
            return `<span class="${cls}">${sign}$${formatNumber(n)}</span>`;
        }
        
        async function updateDashboard() {
            // Fetch status
            const status = await fetchData('/api/status');
            if (!status) return;
            
            // Update mode badge
            const modeBadge = document.getElementById('mode-badge');
            const isPaper = Object.values(status.pairs).some(p => true); // Would need actual mode info
            modeBadge.textContent = 'PAPER';
            modeBadge.style.background = 'rgba(255,193,7,0.2)';
            modeBadge.style.color = '#ffc107';
            
            // Calculate totals
            let totalPnl = 0;
            let realizedPnl = 0;
            let unrealizedPnl = 0;
            let totalTrades = 0;
            let activeGrids = 0;
            
            for (const [pair, data] of Object.entries(status.pairs)) {
                if (data.grid) {
                    realizedPnl += data.grid.realized_pnl || 0;
                    unrealizedPnl += data.grid.unrealized_pnl || 0;
                    totalPnl += data.grid.realized_pnl + data.grid.unrealized_pnl;
                    totalTrades += data.grid.total_trades;
                    if (data.grid.status === 'active') activeGrids++;
                }
            }
            
            // Update stats
            document.getElementById('stats-row').innerHTML = `
                <div class="stat-card">
                    <div class="stat-label">Active Grids</div>
                    <div class="stat-value">${activeGrids}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Realized P&L</div>
                    <div class="stat-value ${realizedPnl >= 0 ? 'positive' : 'negative'}">
                        ${realizedPnl >= 0 ? '+' : ''}$${formatNumber(realizedPnl)}
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Unrealized P&L</div>
                    <div class="stat-value ${unrealizedPnl >= 0 ? 'positive' : 'negative'}">
                        ${unrealizedPnl >= 0 ? '+' : ''}$${formatNumber(unrealizedPnl)}
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total P&L</div>
                    <div class="stat-value ${totalPnl >= 0 ? 'positive' : 'negative'}">
                        ${totalPnl >= 0 ? '+' : ''}$${formatNumber(totalPnl)}
                    </div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Total Trades</div>
                    <div class="stat-value">${totalTrades}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Last Update</div>
                    <div class="stat-value" style="font-size: 16px;">${new Date().toLocaleTimeString()}</div>
                </div>
            `;
            
            // Fetch grids
            const grids = await fetchData('/api/grids');
            const gridsContainer = document.getElementById('grids-container');
            
            if (!grids || grids.length === 0) {
                gridsContainer.innerHTML = '<p style="color:#666;">No active grids</p>';
            } else {
                gridsContainer.innerHTML = grids.map(grid => `
                    <div class="grid-card">
                        <div class="grid-header">
                            <span class="pair-name">${grid.pair}</span>
                            <span class="grid-status ${grid.status}">${grid.status}</span>
                        </div>
                        <div class="grid-info">
                            <div class="info-item">
                                <div class="info-label">Mode</div>
                                <div class="info-value">${grid.mode.toUpperCase()}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Range</div>
                                <div class="info-value">$${formatNumber(grid.lower)} - $${formatNumber(grid.upper)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Levels</div>
                                <div class="info-value">${grid.levels}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Capital</div>
                                <div class="info-value">$${formatNumber(grid.capital)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Position</div>
                                <div class="info-value">${formatNumber(grid.net_position, 6)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Realized P&L</div>
                                <div class="info-value">${formatPnl(grid.realized_pnl)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Unrealized P&L</div>
                                <div class="info-value">${formatPnl(grid.unrealized_pnl)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Total P&L</div>
                                <div class="info-value">${formatPnl(grid.total_pnl)}</div>
                            </div>
                            <div class="info-item">
                                <div class="info-label">Trades</div>
                                <div class="info-value">${grid.total_trades}</div>
                            </div>
                        </div>
                        <div class="levels-visual" id="levels-${grid.pair}">
                            <div class="info-label">Grid Levels</div>
                            <div class="levels-container" id="levels-bars-${grid.pair}"></div>
                        </div>
                        <div class="actions">
                            <button class="btn-stop" onclick="stopGrid('${grid.pair}')">Stop Grid</button>
                            <button class="btn-cancel" onclick="cancelOrders('${grid.pair}')">Cancel Orders</button>
                            <button class="btn-start" onclick="rebuildGrid('${grid.pair}')">Rebuild Grid</button>
                        </div>
                    </div>
                `).join('');
                
                // Load levels for each grid
                for (const grid of grids) {
                    loadLevels(grid.pair);
                }
            }
            
            // Fetch history
            const history = await fetchData('/api/history?limit=50');
            const historyContainer = document.getElementById('history-container');
            
            if (history && history.length > 0) {
                historyContainer.innerHTML = history.map(h => `
                    <div class="history-card">
                        <div style="display:flex; justify-content:space-between; margin-bottom:10px;">
                            <strong>${h.pair}</strong>
                            <span class="grid-status ${h.status}">${h.status}</span>
                        </div>
                        <div style="font-size:12px; color:#888;">
                            Range: ${h.range}<br>
                            P&L: ${formatPnl(h.realized_pnl)}<br>
                            Trades: ${h.total_trades}<br>
                            ${h.stop_reason ? 'Stop: ' + h.stop_reason : ''}
                        </div>
                    </div>
                `).join('');
            }
            
            document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
            
            // Fetch exchange compare data
            await updateExchangeCompare();
        }
        
        async function updateExchangeCompare() {
            const compare = await fetchData('/api/exchange/compare');
            if (!compare) {
                document.getElementById('exchange-real').innerHTML = '<span style="color:#ff6b6b;">Failed to load</span>';
                return;
            }
            
            // Real exchange
            const realDiv = document.getElementById('exchange-real');
            if (compare.real.positions.length === 0) {
                realDiv.innerHTML = `
                    <div style="color:#00d4aa;">✓ No open positions</div>
                    <div style="margin-top:10px; font-size:12px;">
                        Balance: $${formatNumber(compare.real.balance.total || 0)}<br>
                        Available: $${formatNumber(compare.real.balance.available || 0)}
                    </div>
                `;
            } else {
                realDiv.innerHTML = `
                    <div style="color:#ffc107;">⚠ ${compare.real.positions} positions</div>
                    <div style="margin-top:10px; font-size:12px;">
                        ${compare.real.position_details.map(p => `
                            <div style="margin:5px 0; padding:5px; background:rgba(0,0,0,0.2); border-radius:4px;">
                                <strong>${p.pair}</strong> ${p.side.toUpperCase()}<br>
                                Size: ${p.size} @ $${formatNumber(p.entry)}<br>
                                PnL: ${formatPnl(p.pnl)}
                            </div>
                        `).join('')}
                    </div>
                    <div style="margin-top:10px; font-size:12px;">
                        Balance: $${formatNumber(compare.real.balance.total || 0)}
                    </div>
                `;
            }
            
            // Local bot state
            const localDiv = document.getElementById('exchange-local');
            if (compare.local.positions === 0) {
                localDiv.innerHTML = `
                    <div style="color:#00d4aa;">✓ No open positions</div>
                    <div style="margin-top:10px; font-size:12px;">
                        Balance: $${formatNumber(compare.local.balance)}
                    </div>
                `;
            } else {
                localDiv.innerHTML = `
                    <div style="color:#ffc107;">📝 ${compare.local.positions} positions</div>
                    <div style="margin-top:10px; font-size:12px;">
                        ${compare.local.position_details.map(p => `
                            <div style="margin:5px 0; padding:5px; background:rgba(0,0,0,0.2); border-radius:4px;">
                                <strong>${p.pair}</strong> ${p.side.toUpperCase()}<br>
                                Size: ${p.size} @ $${formatNumber(p.entry)}<br>
                                PnL: ${formatPnl(p.pnl)}
                            </div>
                        `).join('')}
                    </div>
                    <div style="margin-top:10px; font-size:12px;">
                        Balance: $${formatNumber(compare.local.balance)}
                    </div>
                `;
            }
            
            // Differences
            const diffDiv = document.getElementById('exchange-diff');
            if (compare.differences && compare.differences.length > 0) {
                diffDiv.innerHTML = `
                    <div class="stat-card" style="background:rgba(255,107,107,0.1); border-color:rgba(255,107,107,0.3);">
                        <div style="color:#ff6b6b; font-weight:bold;">⚠️ Differences Found</div>
                        ${compare.differences.map(d => `
                            <div style="margin-top:8px; font-size:13px;">• ${d.message}</div>
                        `).join('')}
                    </div>
                `;
            } else if (compare.real.positions === 0 && compare.local.positions === 0) {
                diffDiv.innerHTML = `
                    <div style="color:#00d4aa; text-align:center; padding:10px;">
                        ✓ Exchange and Bot State are in sync (both empty)
                    </div>
                `;
            } else if (compare.real.positions > 0 && compare.local.positions > 0 && compare.differences.length === 0) {
                diffDiv.innerHTML = `
                    <div style="color:#00d4aa; text-align:center; padding:10px;">
                        ✓ Exchange and Bot State are in sync
                    </div>
                `;
            } else {
                diffDiv.innerHTML = '';
            }
        }
        
        async function loadLevels(pair) {
            const levels = await fetchData(`/api/levels/${pair}`);
            if (!levels || !levels.levels) return;
            
            const container = document.getElementById(`levels-bars-${pair}`);
            if (!container) return;
            
            const maxTrades = Math.max(...levels.levels.map(l => l.total_buys + l.total_sells), 1);
            
            container.innerHTML = levels.levels.map(level => {
                const height = Math.max(10, ((level.total_buys + level.total_sells) / maxTrades) * 100);
                let cls = level.buy_status === 'pending' ? 'buy' : (level.sell_status === 'pending' ? 'sell' : '');
                if (level.is_current) cls = 'current';
                return `<div class="level-bar ${cls}" style="height:${height}%" title="Level ${level.index}: $${level.price.toFixed(2)}"></div>`;
            }).join('');
        }
        
        async function stopGrid(pair) {
            if (!confirm(`Stop grid for ${pair}? This will close the position.`)) return;
            await fetch(`/api/grid/${pair}/stop`, { method: 'POST' });
            updateDashboard();
        }
        
        async function cancelOrders(pair) {
            if (!confirm(`Cancel all orders for ${pair}?`)) return;
            await fetch(`/api/grid/${pair}/cancel_orders`, { method: 'POST' });
            updateDashboard();
        }
        
        async function rebuildGrid(pair) {
            if (!confirm(`Rebuild grid for ${pair}? This will cancel all orders and create new grid based on current price.`)) return;
            const result = await fetch(`/api/grid/${pair}/rebuild`, { method: 'POST' });
            const data = await result.json();
            if (data.success) {
                alert(`Grid rebuilt! New range: $${data.new_range[0].toFixed(2)} - $${data.new_range[1].toFixed(2)}`);
            } else {
                alert(`Error: ${data.error}`);
            }
            updateDashboard();
        }
        
        // Initial load and auto-refresh
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>'''
