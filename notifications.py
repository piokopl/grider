"""
Notification System for Grid Bot
Supports Discord and Telegram
"""

import asyncio
import aiohttp
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging


class NotificationType(Enum):
    TRADE = "trade"
    GRID_REBUILD = "grid_rebuild"
    OPTIMIZATION = "optimization"
    ERROR = "error"
    WARNING = "warning"
    DAILY_SUMMARY = "daily_summary"
    RISK_ALERT = "risk_alert"
    PUMP_DUMP = "pump_dump"
    FUNDING = "funding"


@dataclass
class NotificationConfig:
    # Discord
    discord_enabled: bool = False
    discord_webhook_url: str = ""
    
    # Telegram
    telegram_enabled: bool = False
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    
    # What to notify
    notify_trades: bool = True
    notify_rebuilds: bool = True
    notify_optimizations: bool = True
    notify_errors: bool = True
    notify_daily_summary: bool = True
    notify_risk_alerts: bool = True
    
    # Rate limiting
    min_interval_seconds: int = 5  # Min time between notifications
    batch_trades: bool = True  # Batch multiple trades into one notification
    batch_interval_seconds: int = 60


class NotificationManager:
    def __init__(self, config: NotificationConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._last_notification_time: Dict[str, datetime] = {}
        self._trade_buffer: List[Dict] = []
        self._buffer_task: Optional[asyncio.Task] = None
        
    async def start(self):
        """Start notification manager"""
        if self.config.batch_trades:
            self._buffer_task = asyncio.create_task(self._flush_buffer_loop())
        self.logger.info("Notification manager started")
    
    async def stop(self):
        """Stop notification manager"""
        if self._buffer_task:
            self._buffer_task.cancel()
            try:
                await self._buffer_task
            except asyncio.CancelledError:
                pass
        # Flush remaining
        await self._flush_trade_buffer()
    
    async def notify(self, notification_type: NotificationType, 
                    title: str, message: str, 
                    data: Optional[Dict] = None,
                    color: int = 0x00d4aa):
        """Send notification"""
        # Check if this type is enabled
        if not self._should_notify(notification_type):
            return
        
        # Rate limiting
        rate_key = f"{notification_type.value}"
        if not self._check_rate_limit(rate_key):
            return
        
        # Send to enabled channels
        tasks = []
        
        if self.config.discord_enabled:
            tasks.append(self._send_discord(title, message, data, color))
        
        if self.config.telegram_enabled:
            tasks.append(self._send_telegram(title, message, data))
        
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def notify_trade(self, pair: str, side: str, price: float, 
                          amount: float, profit: Optional[float] = None):
        """Notify about a trade (buffered)"""
        if not self.config.notify_trades:
            return
        
        trade_info = {
            "pair": pair,
            "side": side,
            "price": price,
            "amount": amount,
            "profit": profit,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        if self.config.batch_trades:
            self._trade_buffer.append(trade_info)
        else:
            emoji = "🟢" if side == "sell" and profit and profit > 0 else "🔵" if side == "buy" else "🔴"
            title = f"{emoji} {pair} {side.upper()}"
            msg = f"Price: ${price:.4f}\nAmount: {amount:.6f}"
            if profit is not None:
                msg += f"\nProfit: ${profit:.2f}"
            
            await self.notify(NotificationType.TRADE, title, msg)
    
    async def notify_grid_rebuild(self, pair: str, reason: str,
                                  old_range: tuple, new_range: tuple):
        """Notify about grid rebuild"""
        if not self.config.notify_rebuilds:
            return
        
        title = f"🔄 {pair} Grid Rebuilt"
        msg = (f"Reason: {reason}\n"
               f"Old: ${old_range[0]:.2f} - ${old_range[1]:.2f}\n"
               f"New: ${new_range[0]:.2f} - ${new_range[1]:.2f}")
        
        await self.notify(NotificationType.GRID_REBUILD, title, msg, color=0xffc107)
    
    async def notify_optimization(self, pair: str, changes: List[str],
                                  confidence: float):
        """Notify about optimization changes"""
        if not self.config.notify_optimizations:
            return
        
        title = f"🔧 {pair} Optimized"
        msg = (f"Changes:\n" + "\n".join(f"• {c}" for c in changes) +
               f"\nConfidence: {confidence:.0f}%")
        
        await self.notify(NotificationType.OPTIMIZATION, title, msg, color=0x9b59b6)
    
    async def notify_error(self, context: str, error: str):
        """Notify about errors"""
        if not self.config.notify_errors:
            return
        
        title = f"❌ Error: {context}"
        await self.notify(NotificationType.ERROR, title, error[:500], color=0xff0000)
    
    async def notify_risk_alert(self, alert_type: str, pair: str, 
                                details: str, severity: str = "warning"):
        """Notify about risk alerts"""
        if not self.config.notify_risk_alerts:
            return
        
        emoji = "🚨" if severity == "critical" else "⚠️"
        color = 0xff0000 if severity == "critical" else 0xff6b6b
        
        title = f"{emoji} Risk Alert: {alert_type}"
        msg = f"Pair: {pair}\n{details}"
        
        await self.notify(NotificationType.RISK_ALERT, title, msg, color=color)
    
    async def notify_daily_summary(self, summary: Dict):
        """Send daily summary"""
        if not self.config.notify_daily_summary:
            return
        
        title = "📊 Daily Summary"
        
        total_pnl = summary.get("total_pnl", 0)
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        
        msg = (f"{pnl_emoji} Total P&L: ${total_pnl:.2f}\n"
               f"📊 Total Trades: {summary.get('total_trades', 0)}\n"
               f"✅ Win Rate: {summary.get('win_rate', 0):.1f}%\n"
               f"🔄 Rebuilds: {summary.get('rebuilds', 0)}\n"
               f"⚡ Active Grids: {summary.get('active_grids', 0)}")
        
        color = 0x00d4aa if total_pnl >= 0 else 0xff6b6b
        await self.notify(NotificationType.DAILY_SUMMARY, title, msg, color=color)
    
    async def notify_pump_dump(self, pair: str, change_percent: float, 
                              timeframe: str, action: str):
        """Notify about pump/dump detection"""
        emoji = "🚀" if change_percent > 0 else "💥"
        direction = "PUMP" if change_percent > 0 else "DUMP"
        
        title = f"{emoji} {direction} Detected: {pair}"
        msg = (f"Change: {change_percent:+.2f}% in {timeframe}\n"
               f"Action: {action}")
        
        await self.notify(NotificationType.PUMP_DUMP, title, msg, color=0xff6b6b)
    
    async def notify_funding(self, pair: str, rate: float, 
                            impact_per_day: float, action: str):
        """Notify about significant funding rate"""
        emoji = "💸" if rate < 0 else "💰"
        
        title = f"{emoji} Funding Alert: {pair}"
        msg = (f"Rate: {rate:.4f}%\n"
               f"Daily Impact: ${impact_per_day:.2f}\n"
               f"Action: {action}")
        
        color = 0x00d4aa if rate < 0 else 0xff6b6b  # Green if we receive, red if we pay
        await self.notify(NotificationType.FUNDING, title, msg, color=color)
    
    def _should_notify(self, notification_type: NotificationType) -> bool:
        """Check if notification type is enabled"""
        if notification_type == NotificationType.TRADE:
            return self.config.notify_trades
        elif notification_type == NotificationType.GRID_REBUILD:
            return self.config.notify_rebuilds
        elif notification_type == NotificationType.OPTIMIZATION:
            return self.config.notify_optimizations
        elif notification_type in [NotificationType.ERROR, NotificationType.WARNING]:
            return self.config.notify_errors
        elif notification_type == NotificationType.DAILY_SUMMARY:
            return self.config.notify_daily_summary
        elif notification_type in [NotificationType.RISK_ALERT, NotificationType.PUMP_DUMP, 
                                   NotificationType.FUNDING]:
            return self.config.notify_risk_alerts
        return True
    
    def _check_rate_limit(self, key: str) -> bool:
        """Check rate limiting"""
        now = datetime.utcnow()
        last = self._last_notification_time.get(key)
        
        if last:
            elapsed = (now - last).total_seconds()
            if elapsed < self.config.min_interval_seconds:
                return False
        
        self._last_notification_time[key] = now
        return True
    
    async def _flush_buffer_loop(self):
        """Periodically flush trade buffer"""
        while True:
            await asyncio.sleep(self.config.batch_interval_seconds)
            await self._flush_trade_buffer()
    
    async def _flush_trade_buffer(self):
        """Send batched trade notifications"""
        if not self._trade_buffer:
            return
        
        trades = self._trade_buffer.copy()
        self._trade_buffer.clear()
        
        # Group by pair
        by_pair: Dict[str, List] = {}
        for t in trades:
            pair = t["pair"]
            if pair not in by_pair:
                by_pair[pair] = []
            by_pair[pair].append(t)
        
        # Create summary
        total_profit = sum(t.get("profit", 0) or 0 for t in trades)
        
        title = f"📊 {len(trades)} Trades Executed"
        
        lines = []
        for pair, pair_trades in by_pair.items():
            buys = sum(1 for t in pair_trades if t["side"] == "buy")
            sells = sum(1 for t in pair_trades if t["side"] == "sell")
            pair_profit = sum(t.get("profit", 0) or 0 for t in pair_trades)
            lines.append(f"{pair}: {buys}B/{sells}S = ${pair_profit:.2f}")
        
        msg = "\n".join(lines)
        if total_profit != 0:
            msg += f"\n\n💰 Total: ${total_profit:.2f}"
        
        color = 0x00d4aa if total_profit >= 0 else 0xff6b6b
        await self.notify(NotificationType.TRADE, title, msg, color=color)
    
    async def _send_discord(self, title: str, message: str, 
                           data: Optional[Dict], color: int):
        """Send Discord webhook"""
        if not self.config.discord_webhook_url:
            return
        
        embed = {
            "title": title,
            "description": message,
            "color": color,
            "timestamp": datetime.utcnow().isoformat(),
            "footer": {"text": "Grid Bot"}
        }
        
        if data:
            embed["fields"] = [
                {"name": k, "value": str(v), "inline": True}
                for k, v in list(data.items())[:5]
            ]
        
        payload = {"embeds": [embed]}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.discord_webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status not in [200, 204]:
                        self.logger.warning(f"Discord webhook failed: {resp.status}")
        except Exception as e:
            self.logger.warning(f"Discord notification error: {e}")
    
    async def _send_telegram(self, title: str, message: str, 
                            data: Optional[Dict]):
        """Send Telegram message"""
        if not self.config.telegram_bot_token or not self.config.telegram_chat_id:
            return
        
        text = f"*{title}*\n\n{message}"
        
        if data:
            text += "\n\n" + "\n".join(f"• {k}: `{v}`" for k, v in list(data.items())[:5])
        
        url = f"https://api.telegram.org/bot{self.config.telegram_bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.telegram_chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url, json=payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        self.logger.warning(f"Telegram send failed: {resp.status}")
        except Exception as e:
            self.logger.warning(f"Telegram notification error: {e}")
