"""
Advanced Statistics and Analytics
Comprehensive performance metrics and analysis
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import math
import json


@dataclass
class TradeRecord:
    timestamp: datetime
    pair: str
    side: str
    price: float
    amount: float
    pnl: float
    fees: float
    position_after: float
    equity_after: float


@dataclass
class DailyStats:
    date: str
    starting_equity: float
    ending_equity: float
    pnl: float
    pnl_percent: float
    trades: int
    winning_trades: int
    losing_trades: int
    volume: float
    fees: float
    max_equity: float
    min_equity: float


@dataclass
class PairStats:
    pair: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    total_fees: float
    avg_trade_pnl: float
    best_trade: float
    worst_trade: float
    avg_hold_time_minutes: float
    profit_factor: float
    win_rate: float


class StatisticsEngine:
    """
    Comprehensive statistics tracking and analysis
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._trades: List[TradeRecord] = []
        self._daily_snapshots: Dict[str, DailyStats] = {}
        self._equity_history: List[Tuple[datetime, float]] = []
        self._peak_equity: float = 0
        self._initial_equity: float = 0
    
    def initialize(self, initial_equity: float):
        """Initialize with starting equity"""
        self._initial_equity = initial_equity
        self._peak_equity = initial_equity
        self._equity_history.append((datetime.utcnow(), initial_equity))
    
    def record_trade(self, pair: str, side: str, price: float, amount: float,
                    pnl: float, fees: float, position_after: float, equity_after: float):
        """Record a trade"""
        trade = TradeRecord(
            timestamp=datetime.utcnow(),
            pair=pair,
            side=side,
            price=price,
            amount=amount,
            pnl=pnl,
            fees=fees,
            position_after=position_after,
            equity_after=equity_after
        )
        self._trades.append(trade)
        
        # Update equity tracking
        self._equity_history.append((trade.timestamp, equity_after))
        if equity_after > self._peak_equity:
            self._peak_equity = equity_after
        
        # Update daily stats
        self._update_daily_stats(trade)
    
    def record_equity_snapshot(self, equity: float):
        """Record periodic equity snapshot"""
        now = datetime.utcnow()
        self._equity_history.append((now, equity))
        if equity > self._peak_equity:
            self._peak_equity = equity
    
    def _update_daily_stats(self, trade: TradeRecord):
        """Update daily statistics"""
        date_key = trade.timestamp.date().isoformat()
        
        if date_key not in self._daily_snapshots:
            # Get previous day's ending equity or use initial
            prev_equity = self._initial_equity
            if self._daily_snapshots:
                last_day = max(self._daily_snapshots.keys())
                prev_equity = self._daily_snapshots[last_day].ending_equity
            
            self._daily_snapshots[date_key] = DailyStats(
                date=date_key,
                starting_equity=prev_equity,
                ending_equity=trade.equity_after,
                pnl=0,
                pnl_percent=0,
                trades=0,
                winning_trades=0,
                losing_trades=0,
                volume=0,
                fees=0,
                max_equity=trade.equity_after,
                min_equity=trade.equity_after
            )
        
        daily = self._daily_snapshots[date_key]
        daily.trades += 1
        daily.pnl += trade.pnl
        daily.fees += trade.fees
        daily.volume += trade.price * trade.amount
        daily.ending_equity = trade.equity_after
        daily.max_equity = max(daily.max_equity, trade.equity_after)
        daily.min_equity = min(daily.min_equity, trade.equity_after)
        
        if daily.starting_equity > 0:
            daily.pnl_percent = (daily.ending_equity - daily.starting_equity) / daily.starting_equity * 100
        
        if trade.pnl > 0:
            daily.winning_trades += 1
        elif trade.pnl < 0:
            daily.losing_trades += 1
    
    # === PERFORMANCE METRICS ===
    
    def get_total_pnl(self) -> float:
        """Get total P&L"""
        return sum(t.pnl for t in self._trades)
    
    def get_total_pnl_percent(self) -> float:
        """Get total P&L as percentage"""
        if self._initial_equity == 0:
            return 0
        return self.get_total_pnl() / self._initial_equity * 100
    
    def get_win_rate(self) -> float:
        """Get overall win rate"""
        if not self._trades:
            return 0
        winning = sum(1 for t in self._trades if t.pnl > 0)
        return winning / len(self._trades) * 100
    
    def get_profit_factor(self) -> float:
        """Gross profit / Gross loss"""
        gross_profit = sum(t.pnl for t in self._trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in self._trades if t.pnl < 0))
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0
        return gross_profit / gross_loss
    
    def get_avg_trade_pnl(self) -> float:
        """Average P&L per trade"""
        if not self._trades:
            return 0
        return sum(t.pnl for t in self._trades) / len(self._trades)
    
    def get_expectancy(self) -> float:
        """Expected value per trade"""
        if not self._trades:
            return 0
        win_rate = self.get_win_rate() / 100
        avg_win = self._get_avg_win()
        avg_loss = abs(self._get_avg_loss())
        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    def _get_avg_win(self) -> float:
        """Average winning trade"""
        wins = [t.pnl for t in self._trades if t.pnl > 0]
        return sum(wins) / len(wins) if wins else 0
    
    def _get_avg_loss(self) -> float:
        """Average losing trade"""
        losses = [t.pnl for t in self._trades if t.pnl < 0]
        return sum(losses) / len(losses) if losses else 0
    
    # === RISK METRICS ===
    
    def get_max_drawdown(self) -> Tuple[float, float]:
        """Get maximum drawdown (USD and %)"""
        if not self._equity_history:
            return 0, 0
        
        peak = self._equity_history[0][1]
        max_dd = 0
        max_dd_pct = 0
        
        for _, equity in self._equity_history:
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd:
                max_dd = dd
                max_dd_pct = dd_pct
        
        return max_dd, max_dd_pct
    
    def get_sharpe_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sharpe ratio (annualized)"""
        returns = self._get_daily_returns()
        if len(returns) < 2:
            return 0
        
        avg_return = sum(returns) / len(returns)
        std_return = self._std(returns)
        
        if std_return == 0:
            return 0
        
        # Annualize
        annual_return = avg_return * 365
        annual_std = std_return * math.sqrt(365)
        
        return (annual_return - risk_free_rate) / annual_std
    
    def get_sortino_ratio(self, risk_free_rate: float = 0.05) -> float:
        """Calculate Sortino ratio (only downside volatility)"""
        returns = self._get_daily_returns()
        if len(returns) < 2:
            return 0
        
        avg_return = sum(returns) / len(returns)
        negative_returns = [r for r in returns if r < 0]
        
        if not negative_returns:
            return float('inf') if avg_return > 0 else 0
        
        downside_std = self._std(negative_returns)
        if downside_std == 0:
            return 0
        
        annual_return = avg_return * 365
        annual_downside_std = downside_std * math.sqrt(365)
        
        return (annual_return - risk_free_rate) / annual_downside_std
    
    def get_calmar_ratio(self) -> float:
        """Annual return / Max drawdown"""
        _, max_dd_pct = self.get_max_drawdown()
        if max_dd_pct == 0:
            return 0
        
        # Calculate annual return
        if not self._equity_history or len(self._equity_history) < 2:
            return 0
        
        first = self._equity_history[0]
        last = self._equity_history[-1]
        days = (last[0] - first[0]).total_seconds() / 86400
        
        if days == 0:
            return 0
        
        total_return = (last[1] - first[1]) / first[1] * 100 if first[1] > 0 else 0
        annual_return = total_return * (365 / days)
        
        return annual_return / max_dd_pct
    
    def _get_daily_returns(self) -> List[float]:
        """Get list of daily returns"""
        returns = []
        for date, stats in sorted(self._daily_snapshots.items()):
            if stats.starting_equity > 0:
                returns.append((stats.ending_equity - stats.starting_equity) / stats.starting_equity)
        return returns
    
    def _std(self, values: List[float]) -> float:
        """Calculate standard deviation"""
        if len(values) < 2:
            return 0
        avg = sum(values) / len(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        return math.sqrt(variance)
    
    # === STREAK ANALYSIS ===
    
    def get_max_consecutive_wins(self) -> int:
        """Maximum consecutive winning trades"""
        return self._get_max_streak(True)
    
    def get_max_consecutive_losses(self) -> int:
        """Maximum consecutive losing trades"""
        return self._get_max_streak(False)
    
    def _get_max_streak(self, winning: bool) -> int:
        """Get maximum streak of wins or losses"""
        max_streak = 0
        current_streak = 0
        
        for trade in self._trades:
            is_win = trade.pnl > 0
            if is_win == winning:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
        
        return max_streak
    
    def get_current_streak(self) -> Tuple[str, int]:
        """Get current streak (type, count)"""
        if not self._trades:
            return "none", 0
        
        # Count from end
        streak = 0
        streak_type = "win" if self._trades[-1].pnl > 0 else "loss"
        
        for trade in reversed(self._trades):
            is_win = trade.pnl > 0
            current_type = "win" if is_win else "loss"
            
            if current_type == streak_type:
                streak += 1
            else:
                break
        
        return streak_type, streak
    
    # === TIME ANALYSIS ===
    
    def get_best_trading_hours(self) -> Dict[int, float]:
        """Get P&L by hour of day"""
        by_hour: Dict[int, List[float]] = {h: [] for h in range(24)}
        
        for trade in self._trades:
            hour = trade.timestamp.hour
            by_hour[hour].append(trade.pnl)
        
        return {h: sum(pnls) for h, pnls in by_hour.items() if pnls}
    
    def get_best_trading_days(self) -> Dict[str, float]:
        """Get P&L by day of week"""
        days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        by_day: Dict[str, List[float]] = {d: [] for d in days}
        
        for trade in self._trades:
            day = days[trade.timestamp.weekday()]
            by_day[day].append(trade.pnl)
        
        return {d: sum(pnls) for d, pnls in by_day.items() if pnls}
    
    # === PAIR ANALYSIS ===
    
    def get_pair_stats(self) -> Dict[str, PairStats]:
        """Get statistics per pair"""
        by_pair: Dict[str, List[TradeRecord]] = {}
        
        for trade in self._trades:
            if trade.pair not in by_pair:
                by_pair[trade.pair] = []
            by_pair[trade.pair].append(trade)
        
        result = {}
        for pair, trades in by_pair.items():
            wins = [t for t in trades if t.pnl > 0]
            losses = [t for t in trades if t.pnl < 0]
            
            total_pnl = sum(t.pnl for t in trades)
            total_fees = sum(t.fees for t in trades)
            
            gross_profit = sum(t.pnl for t in wins)
            gross_loss = abs(sum(t.pnl for t in losses))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            result[pair] = PairStats(
                pair=pair,
                total_trades=len(trades),
                winning_trades=len(wins),
                losing_trades=len(losses),
                total_pnl=total_pnl,
                total_fees=total_fees,
                avg_trade_pnl=total_pnl / len(trades) if trades else 0,
                best_trade=max((t.pnl for t in trades), default=0),
                worst_trade=min((t.pnl for t in trades), default=0),
                avg_hold_time_minutes=0,  # Would need entry/exit pairs
                profit_factor=profit_factor,
                win_rate=len(wins) / len(trades) * 100 if trades else 0
            )
        
        return result
    
    # === REPORTS ===
    
    def get_full_report(self) -> Dict:
        """Get comprehensive statistics report"""
        max_dd, max_dd_pct = self.get_max_drawdown()
        streak_type, streak_count = self.get_current_streak()
        
        return {
            "overview": {
                "total_trades": len(self._trades),
                "total_pnl": round(self.get_total_pnl(), 2),
                "total_pnl_percent": round(self.get_total_pnl_percent(), 2),
                "win_rate": round(self.get_win_rate(), 1),
                "profit_factor": round(self.get_profit_factor(), 2),
                "avg_trade_pnl": round(self.get_avg_trade_pnl(), 2),
                "expectancy": round(self.get_expectancy(), 2)
            },
            "risk": {
                "max_drawdown_usd": round(max_dd, 2),
                "max_drawdown_percent": round(max_dd_pct, 2),
                "sharpe_ratio": round(self.get_sharpe_ratio(), 2),
                "sortino_ratio": round(self.get_sortino_ratio(), 2),
                "calmar_ratio": round(self.get_calmar_ratio(), 2)
            },
            "streaks": {
                "max_consecutive_wins": self.get_max_consecutive_wins(),
                "max_consecutive_losses": self.get_max_consecutive_losses(),
                "current_streak_type": streak_type,
                "current_streak_count": streak_count
            },
            "by_pair": {
                pair: {
                    "trades": s.total_trades,
                    "pnl": round(s.total_pnl, 2),
                    "win_rate": round(s.win_rate, 1)
                }
                for pair, s in self.get_pair_stats().items()
            },
            "by_hour": {h: round(p, 2) for h, p in self.get_best_trading_hours().items()},
            "by_day": {d: round(p, 2) for d, p in self.get_best_trading_days().items()},
            "daily": [
                {
                    "date": stats.date,
                    "pnl": round(stats.pnl, 2),
                    "trades": stats.trades,
                    "win_rate": round(stats.winning_trades / stats.trades * 100, 1) if stats.trades > 0 else 0
                }
                for stats in sorted(self._daily_snapshots.values(), key=lambda x: x.date)[-30:]
            ]
        }
    
    def get_trade_heatmap(self) -> Dict[str, Dict[int, int]]:
        """Get trade count heatmap by day and hour"""
        heatmap: Dict[str, Dict[int, int]] = {}
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        
        for day in days:
            heatmap[day] = {h: 0 for h in range(24)}
        
        for trade in self._trades:
            day = days[trade.timestamp.weekday()]
            hour = trade.timestamp.hour
            heatmap[day][hour] += 1
        
        return heatmap
    
    def export_trades(self) -> List[Dict]:
        """Export all trades as list of dicts"""
        return [
            {
                "timestamp": t.timestamp.isoformat(),
                "pair": t.pair,
                "side": t.side,
                "price": t.price,
                "amount": t.amount,
                "pnl": t.pnl,
                "fees": t.fees,
                "position_after": t.position_after,
                "equity_after": t.equity_after
            }
            for t in self._trades
        ]
