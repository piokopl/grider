"""
Risk Management System for Grid Bot
Handles drawdown limits, position limits, correlation, and safety controls
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import math


class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskConfig:
    # Drawdown limits
    max_daily_loss_percent: float = 5.0      # Max daily loss as % of total capital
    max_daily_loss_usd: float = 500.0        # Max daily loss in USD
    max_total_drawdown_percent: float = 15.0 # Max total drawdown
    
    # Position limits
    max_position_per_pair_percent: float = 20.0  # Max position size per pair
    max_total_exposure_percent: float = 80.0     # Max total exposure
    max_correlated_exposure_percent: float = 40.0  # Max exposure on correlated pairs
    
    # Trade limits
    max_trades_per_hour: int = 50            # Max trades per hour (all pairs)
    max_consecutive_losses: int = 5          # Max consecutive losing trades
    
    # Pump/dump detection
    pump_dump_threshold_percent: float = 5.0  # % move in short time
    pump_dump_timeframe_minutes: int = 5      # Timeframe to check
    pause_on_pump_dump: bool = True           # Pause trading on detection
    pump_dump_cooldown_minutes: int = 30      # How long to pause
    
    # Funding rate
    max_funding_rate_percent: float = 0.1    # Max acceptable funding rate
    funding_check_interval_minutes: int = 60  # How often to check funding
    
    # Auto actions
    auto_stop_on_critical: bool = True       # Auto-stop bot on critical risk
    reduce_position_on_high: bool = True     # Reduce position on high risk
    
    # Correlation groups (pairs that tend to move together)
    correlation_groups: List[List[str]] = field(default_factory=lambda: [
        ["BTCUSDT", "ETHUSDT"],  # Major cryptos
        ["SOLUSDT", "AVAXUSDT", "NEARUSDT", "SUIUSDT"],  # L1s
        ["LINKUSDT", "DOTUSDT", "ATOMUSDT"],  # Infrastructure
        ["ADAUSDT", "ALGOUSDT", "IOTAUSDT"],  # Smart contracts
    ])


@dataclass 
class RiskState:
    """Current risk state"""
    daily_pnl: float = 0.0
    daily_trades: int = 0
    daily_start_equity: float = 0.0
    consecutive_losses: int = 0
    current_drawdown_percent: float = 0.0
    peak_equity: float = 0.0
    total_exposure: float = 0.0
    last_reset: str = ""
    paused_pairs: Dict[str, str] = field(default_factory=dict)  # pair -> reason
    risk_level: RiskLevel = RiskLevel.LOW


class RiskManager:
    """
    Manages risk across all trading pairs
    """
    
    def __init__(self, config: RiskConfig, logger: logging.Logger,
                 notification_manager=None):
        self.config = config
        self.logger = logger
        self.notifications = notification_manager
        
        self.state = RiskState()
        self._trade_history: List[Dict] = []  # Recent trades for analysis
        self._price_history: Dict[str, List[Tuple[datetime, float]]] = {}
        self._funding_rates: Dict[str, float] = {}
        self._last_funding_check: Optional[datetime] = None
    
    async def initialize(self, initial_equity: float):
        """Initialize risk manager with starting equity"""
        self.state.daily_start_equity = initial_equity
        self.state.peak_equity = initial_equity
        self.state.last_reset = datetime.utcnow().date().isoformat()
        self.logger.info(f"Risk manager initialized with equity: ${initial_equity:.2f}")
    
    async def check_daily_reset(self, current_equity: float):
        """Check if daily stats need reset"""
        today = datetime.utcnow().date().isoformat()
        if self.state.last_reset != today:
            self.logger.info(f"Daily reset - Previous P&L: ${self.state.daily_pnl:.2f}")
            self.state.daily_pnl = 0.0
            self.state.daily_trades = 0
            self.state.daily_start_equity = current_equity
            self.state.consecutive_losses = 0
            self.state.last_reset = today
            self._trade_history.clear()
    
    def update_equity(self, current_equity: float):
        """Update equity tracking"""
        # Update peak
        if current_equity > self.state.peak_equity:
            self.state.peak_equity = current_equity
        
        # Calculate drawdown
        if self.state.peak_equity > 0:
            self.state.current_drawdown_percent = (
                (self.state.peak_equity - current_equity) / self.state.peak_equity * 100
            )
    
    def record_trade(self, pair: str, side: str, pnl: float, 
                    position_value: float):
        """Record a trade for risk tracking"""
        self.state.daily_trades += 1
        self.state.daily_pnl += pnl
        
        # Track consecutive losses
        if pnl < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0
        
        # Add to history
        self._trade_history.append({
            "pair": pair,
            "side": side,
            "pnl": pnl,
            "timestamp": datetime.utcnow()
        })
        
        # Keep only last 100 trades
        if len(self._trade_history) > 100:
            self._trade_history = self._trade_history[-100:]
    
    def record_price(self, pair: str, price: float):
        """Record price for pump/dump detection"""
        now = datetime.utcnow()
        
        if pair not in self._price_history:
            self._price_history[pair] = []
        
        self._price_history[pair].append((now, price))
        
        # Keep only recent history
        cutoff = now - timedelta(minutes=60)
        self._price_history[pair] = [
            (t, p) for t, p in self._price_history[pair] if t > cutoff
        ]
    
    async def check_risk(self, pair: str, current_equity: float,
                        position_values: Dict[str, float]) -> Tuple[bool, str, RiskLevel]:
        """
        Check if trading should be allowed
        Returns: (allowed, reason, risk_level)
        """
        await self.check_daily_reset(current_equity)
        self.update_equity(current_equity)
        
        # Calculate total exposure
        self.state.total_exposure = sum(abs(v) for v in position_values.values())
        
        reasons = []
        risk_level = RiskLevel.LOW
        
        # Check if pair is paused
        if pair in self.state.paused_pairs:
            return False, f"Pair paused: {self.state.paused_pairs[pair]}", RiskLevel.HIGH
        
        # === CRITICAL CHECKS ===
        
        # Max drawdown
        if self.state.current_drawdown_percent >= self.config.max_total_drawdown_percent:
            reasons.append(f"Max drawdown reached: {self.state.current_drawdown_percent:.1f}%")
            risk_level = RiskLevel.CRITICAL
        
        # Daily loss USD
        if self.state.daily_pnl <= -self.config.max_daily_loss_usd:
            reasons.append(f"Daily loss limit: ${abs(self.state.daily_pnl):.2f}")
            risk_level = RiskLevel.CRITICAL
        
        # Daily loss percent
        daily_loss_percent = 0
        if self.state.daily_start_equity > 0:
            daily_loss_percent = abs(self.state.daily_pnl) / self.state.daily_start_equity * 100
            if self.state.daily_pnl < 0 and daily_loss_percent >= self.config.max_daily_loss_percent:
                reasons.append(f"Daily loss %: {daily_loss_percent:.1f}%")
                risk_level = RiskLevel.CRITICAL
        
        # Consecutive losses
        if self.state.consecutive_losses >= self.config.max_consecutive_losses:
            reasons.append(f"Consecutive losses: {self.state.consecutive_losses}")
            risk_level = max(risk_level, RiskLevel.HIGH)
        
        # === HIGH RISK CHECKS ===
        
        # Total exposure
        exposure_percent = self.state.total_exposure / current_equity * 100 if current_equity > 0 else 0
        if exposure_percent >= self.config.max_total_exposure_percent:
            reasons.append(f"Max exposure: {exposure_percent:.1f}%")
            risk_level = max(risk_level, RiskLevel.HIGH)
        
        # Per-pair position limit
        pair_position = abs(position_values.get(pair, 0))
        pair_percent = pair_position / current_equity * 100 if current_equity > 0 else 0
        if pair_percent >= self.config.max_position_per_pair_percent:
            reasons.append(f"{pair} position too large: {pair_percent:.1f}%")
            risk_level = max(risk_level, RiskLevel.HIGH)
        
        # Correlated exposure
        correlated_exposure = self._calculate_correlated_exposure(pair, position_values)
        correlated_percent = correlated_exposure / current_equity * 100 if current_equity > 0 else 0
        if correlated_percent >= self.config.max_correlated_exposure_percent:
            reasons.append(f"Correlated exposure too high: {correlated_percent:.1f}%")
            risk_level = max(risk_level, RiskLevel.MEDIUM)
        
        # Trade frequency
        trades_last_hour = sum(
            1 for t in self._trade_history
            if t["timestamp"] > datetime.utcnow() - timedelta(hours=1)
        )
        if trades_last_hour >= self.config.max_trades_per_hour:
            reasons.append(f"Trade frequency limit: {trades_last_hour}/hr")
            risk_level = max(risk_level, RiskLevel.MEDIUM)
        
        # Update state
        self.state.risk_level = risk_level
        
        # Determine if allowed
        if risk_level == RiskLevel.CRITICAL:
            if self.config.auto_stop_on_critical and self.notifications:
                await self.notifications.notify_risk_alert(
                    "CRITICAL", pair, "\n".join(reasons), "critical"
                )
            return False, "; ".join(reasons), risk_level
        
        if risk_level == RiskLevel.HIGH and self.config.reduce_position_on_high:
            if self.notifications:
                await self.notifications.notify_risk_alert(
                    "HIGH RISK", pair, "\n".join(reasons), "warning"
                )
            # Allow but with reduced size
            return True, f"Reduced size: {'; '.join(reasons)}", risk_level
        
        return True, "", risk_level
    
    async def check_pump_dump(self, pair: str, current_price: float) -> Tuple[bool, float]:
        """
        Check for pump/dump pattern
        Returns: (detected, change_percent)
        """
        self.record_price(pair, current_price)
        
        if pair not in self._price_history:
            return False, 0
        
        history = self._price_history[pair]
        if len(history) < 2:
            return False, 0
        
        # Check price change in timeframe
        cutoff = datetime.utcnow() - timedelta(minutes=self.config.pump_dump_timeframe_minutes)
        old_prices = [p for t, p in history if t <= cutoff]
        
        if not old_prices:
            return False, 0
        
        old_price = old_prices[-1]  # Most recent price before cutoff
        change_percent = (current_price - old_price) / old_price * 100
        
        if abs(change_percent) >= self.config.pump_dump_threshold_percent:
            if self.config.pause_on_pump_dump:
                direction = "pump" if change_percent > 0 else "dump"
                self.pause_pair(pair, f"{direction} detected: {change_percent:.2f}%",
                               self.config.pump_dump_cooldown_minutes)
                
                if self.notifications:
                    await self.notifications.notify_pump_dump(
                        pair, change_percent,
                        f"{self.config.pump_dump_timeframe_minutes}min",
                        f"Paused for {self.config.pump_dump_cooldown_minutes}min"
                    )
            
            return True, change_percent
        
        return False, change_percent
    
    def update_funding_rate(self, pair: str, rate: float):
        """Update funding rate for a pair"""
        self._funding_rates[pair] = rate
    
    async def check_funding_rate(self, pair: str, position_value: float) -> Tuple[bool, str]:
        """
        Check if funding rate is acceptable
        Returns: (acceptable, action)
        """
        rate = self._funding_rates.get(pair, 0)
        
        if abs(rate) >= self.config.max_funding_rate_percent:
            # Calculate impact
            daily_impact = position_value * rate * 3  # 3 funding periods per day
            
            action = "consider closing" if rate > 0 else "favorable rate"
            
            if rate > 0 and self.notifications:
                await self.notifications.notify_funding(
                    pair, rate, daily_impact, action
                )
            
            if rate > 0:  # We pay funding
                return False, f"High funding: {rate:.4f}%"
        
        return True, ""
    
    def pause_pair(self, pair: str, reason: str, duration_minutes: int = 30):
        """Pause trading on a pair"""
        self.state.paused_pairs[pair] = reason
        self.logger.warning(f"Paused {pair}: {reason}")
        
        # Schedule unpause
        asyncio.get_event_loop().call_later(
            duration_minutes * 60,
            lambda: self.unpause_pair(pair)
        )
    
    def unpause_pair(self, pair: str):
        """Unpause trading on a pair"""
        if pair in self.state.paused_pairs:
            del self.state.paused_pairs[pair]
            self.logger.info(f"Unpaused {pair}")
    
    def _calculate_correlated_exposure(self, pair: str, 
                                       position_values: Dict[str, float]) -> float:
        """Calculate total exposure on correlated pairs"""
        # Find correlation group for this pair
        correlated_pairs = {pair}
        
        for group in self.config.correlation_groups:
            if pair in group:
                correlated_pairs.update(group)
                break
        
        # Sum exposure
        return sum(
            abs(position_values.get(p, 0))
            for p in correlated_pairs
        )
    
    def get_position_size_multiplier(self) -> float:
        """
        Get position size multiplier based on risk level
        Returns 0-1 where 1 is full size
        """
        if self.state.risk_level == RiskLevel.CRITICAL:
            return 0.0
        elif self.state.risk_level == RiskLevel.HIGH:
            return 0.5
        elif self.state.risk_level == RiskLevel.MEDIUM:
            return 0.75
        return 1.0
    
    def get_risk_summary(self) -> Dict:
        """Get current risk summary"""
        return {
            "risk_level": self.state.risk_level.value,
            "daily_pnl": round(self.state.daily_pnl, 2),
            "daily_trades": self.state.daily_trades,
            "consecutive_losses": self.state.consecutive_losses,
            "current_drawdown_percent": round(self.state.current_drawdown_percent, 2),
            "total_exposure": round(self.state.total_exposure, 2),
            "paused_pairs": list(self.state.paused_pairs.keys()),
            "position_multiplier": self.get_position_size_multiplier()
        }
