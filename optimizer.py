"""
Grid Bot Optimizer
Automatic strategy adjustment and parameter optimization
"""

import math
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

from database import Database, Grid, GridLevel


@dataclass
class MarketCondition:
    """Current market analysis"""
    trend: str  # bullish / bearish / sideways
    trend_strength: float  # 0-100
    volatility: float  # % daily volatility
    volatility_trend: str  # increasing / decreasing / stable
    volume_trend: str  # increasing / decreasing / stable
    recommended_mode: str  # neutral / long / short
    recommended_range: float  # suggested range %
    recommended_levels: int  # suggested grid levels
    confidence: float  # 0-100


@dataclass
class GridPerformance:
    """Grid performance metrics"""
    grid_id: int
    pair: str
    duration_hours: float
    total_trades: int
    trades_per_hour: float
    win_rate: float  # % of profitable trades
    realized_pnl: float
    pnl_per_trade: float
    pnl_per_hour: float
    max_drawdown: float
    efficiency: float  # trades vs levels ratio
    range_utilization: float  # % of range actually used
    

@dataclass
class OptimizationResult:
    """Result of optimization analysis"""
    pair: str
    current_params: Dict
    suggested_params: Dict
    changes: List[str]
    expected_improvement: float  # % improvement estimate
    confidence: float
    reason: str


class GridOptimizer:
    """
    Automatic grid optimization based on:
    1. Market conditions (trend, volatility)
    2. Historical performance
    3. Risk management
    """
    
    # Optimization thresholds
    MIN_TRADES_FOR_ANALYSIS = 5
    MIN_HOURS_FOR_ANALYSIS = 2
    POOR_PERFORMANCE_THRESHOLD = -0.5  # % per hour
    GOOD_PERFORMANCE_THRESHOLD = 0.5   # % per hour
    
    # Parameter bounds
    MIN_GRID_LEVELS = 8
    MAX_GRID_LEVELS = 40
    MIN_RANGE_PERCENT = 3
    MAX_RANGE_PERCENT = 35
    MIN_PROFIT_PERCENT = 0.2
    MAX_PROFIT_PERCENT = 1.5
    
    def __init__(self, db: Database, logger: logging.Logger):
        self.db = db
        self.logger = logger
    
    def analyze_market(self, klines: List[Dict], pair: str) -> MarketCondition:
        """Analyze market conditions from kline data"""
        if not klines or len(klines) < 20:
            return self._default_market_condition()
        
        closes = [float(k["close"]) for k in klines]
        highs = [float(k["high"]) for k in klines]
        lows = [float(k["low"]) for k in klines]
        volumes = [float(k.get("volume", 0)) for k in klines]
        
        # Calculate trend
        trend, trend_strength = self._calculate_trend(closes)
        
        # Calculate volatility
        volatility = self._calculate_volatility(closes)
        volatility_trend = self._calculate_volatility_trend(closes)
        
        # Volume trend
        volume_trend = self._calculate_volume_trend(volumes)
        
        # Determine recommended mode
        recommended_mode = self._recommend_mode(trend, trend_strength, volatility)
        
        # Recommend range based on volatility
        recommended_range = self._recommend_range(volatility, trend_strength)
        
        # Recommend levels based on range and volatility
        recommended_levels = self._recommend_levels(recommended_range, volatility)
        
        # Confidence based on data quality
        confidence = min(100, len(klines) / 2)
        
        return MarketCondition(
            trend=trend,
            trend_strength=trend_strength,
            volatility=volatility,
            volatility_trend=volatility_trend,
            volume_trend=volume_trend,
            recommended_mode=recommended_mode,
            recommended_range=recommended_range,
            recommended_levels=recommended_levels,
            confidence=confidence
        )
    
    def _calculate_trend(self, closes: List[float]) -> Tuple[str, float]:
        """Calculate trend direction and strength"""
        if len(closes) < 20:
            return "sideways", 50
        
        # EMA 10 vs EMA 20
        ema10 = self._ema(closes, 10)
        ema20 = self._ema(closes, 20)
        
        if not ema10 or not ema20:
            return "sideways", 50
        
        # Current position
        current_price = closes[-1]
        ema10_val = ema10[-1]
        ema20_val = ema20[-1]
        
        # Calculate strength as % distance between EMAs
        ema_diff_percent = abs(ema10_val - ema20_val) / ema20_val * 100
        
        # Price position relative to EMAs
        above_both = current_price > ema10_val and current_price > ema20_val
        below_both = current_price < ema10_val and current_price < ema20_val
        ema10_above_20 = ema10_val > ema20_val
        
        if above_both and ema10_above_20:
            trend = "bullish"
            strength = min(100, 50 + ema_diff_percent * 10)
        elif below_both and not ema10_above_20:
            trend = "bearish"
            strength = min(100, 50 + ema_diff_percent * 10)
        else:
            trend = "sideways"
            strength = max(0, 50 - ema_diff_percent * 10)
        
        return trend, strength
    
    def _calculate_volatility(self, closes: List[float]) -> float:
        """Calculate price volatility as daily %"""
        if len(closes) < 10:
            return 5.0
        
        # Calculate returns
        returns = [(closes[i] - closes[i-1]) / closes[i-1] * 100 
                  for i in range(1, len(closes))]
        
        # Standard deviation of returns
        avg = sum(returns) / len(returns)
        variance = sum((r - avg) ** 2 for r in returns) / len(returns)
        std = math.sqrt(variance)
        
        # Annualize (assuming hourly data, ~24 periods per day)
        daily_vol = std * math.sqrt(24)
        
        return round(daily_vol, 2)
    
    def _calculate_volatility_trend(self, closes: List[float]) -> str:
        """Determine if volatility is increasing or decreasing"""
        if len(closes) < 40:
            return "stable"
        
        # Compare recent vs older volatility
        recent = closes[-20:]
        older = closes[-40:-20]
        
        recent_vol = self._calculate_volatility(recent)
        older_vol = self._calculate_volatility(older)
        
        change = (recent_vol - older_vol) / older_vol * 100 if older_vol > 0 else 0
        
        if change > 20:
            return "increasing"
        elif change < -20:
            return "decreasing"
        return "stable"
    
    def _calculate_volume_trend(self, volumes: List[float]) -> str:
        """Determine volume trend"""
        if not volumes or len(volumes) < 20:
            return "stable"
        
        recent_avg = sum(volumes[-10:]) / 10
        older_avg = sum(volumes[-20:-10]) / 10
        
        if older_avg == 0:
            return "stable"
        
        change = (recent_avg - older_avg) / older_avg * 100
        
        if change > 30:
            return "increasing"
        elif change < -30:
            return "decreasing"
        return "stable"
    
    def _recommend_mode(self, trend: str, strength: float, volatility: float) -> str:
        """Recommend grid mode based on market conditions"""
        # Strong trend = follow it
        if strength > 70:
            if trend == "bullish":
                return "long"
            elif trend == "bearish":
                return "short"
        
        # Sideways or weak trend = neutral grid works best
        return "neutral"
    
    def _recommend_range(self, volatility: float, trend_strength: float) -> float:
        """Recommend grid range based on volatility"""
        # Base range on volatility
        base_range = volatility * 2.5
        
        # Wider range for strong trends (price moves more)
        if trend_strength > 70:
            base_range *= 1.3
        
        # Clamp to bounds
        return max(self.MIN_RANGE_PERCENT, 
                   min(self.MAX_RANGE_PERCENT, base_range))
    
    def _recommend_levels(self, range_percent: float, volatility: float) -> int:
        """Recommend number of grid levels"""
        # More levels for wider ranges
        # Target: each level captures ~0.3-0.5% move
        target_level_spacing = 0.4
        
        levels = int(range_percent / target_level_spacing)
        
        # Fewer levels for high volatility (bigger moves)
        if volatility > 10:
            levels = int(levels * 0.8)
        
        return max(self.MIN_GRID_LEVELS, 
                   min(self.MAX_GRID_LEVELS, levels))
    
    def _ema(self, data: List[float], period: int) -> List[float]:
        """Calculate EMA"""
        if len(data) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema = [sum(data[:period]) / period]
        
        for price in data[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def _default_market_condition(self) -> MarketCondition:
        """Default market condition when data is insufficient"""
        return MarketCondition(
            trend="sideways",
            trend_strength=50,
            volatility=5.0,
            volatility_trend="stable",
            volume_trend="stable",
            recommended_mode="neutral",
            recommended_range=10,
            recommended_levels=15,
            confidence=20
        )
    
    def analyze_grid_performance(self, grid: Grid) -> Optional[GridPerformance]:
        """Analyze performance of a grid"""
        if not grid:
            return None
        
        # Calculate duration
        created = datetime.fromisoformat(grid.created_at)
        if grid.stopped_at:
            ended = datetime.fromisoformat(grid.stopped_at)
        else:
            ended = datetime.utcnow()
        
        duration_hours = (ended - created).total_seconds() / 3600
        
        if duration_hours < self.MIN_HOURS_FOR_ANALYSIS:
            return None
        
        # Get levels for analysis
        levels = self.db.get_grid_levels(grid.id)
        
        # Calculate metrics
        total_trades = grid.total_trades
        trades_per_hour = total_trades / duration_hours if duration_hours > 0 else 0
        
        # Win rate (approximated from grid_profits)
        win_rate = (grid.grid_profits / total_trades * 100) if total_trades > 0 else 0
        
        # P&L metrics
        pnl_per_trade = grid.realized_pnl / total_trades if total_trades > 0 else 0
        pnl_per_hour = grid.realized_pnl / duration_hours if duration_hours > 0 else 0
        
        # Efficiency: how many levels generated trades
        active_levels = sum(1 for l in levels if l.total_buys > 0 or l.total_sells > 0)
        efficiency = active_levels / len(levels) * 100 if levels else 0
        
        # Range utilization
        if levels:
            prices_traded = [l.price for l in levels if l.total_buys > 0 or l.total_sells > 0]
            if prices_traded:
                traded_range = max(prices_traded) - min(prices_traded)
                total_range = grid.upper_price - grid.lower_price
                range_utilization = traded_range / total_range * 100 if total_range > 0 else 0
            else:
                range_utilization = 0
        else:
            range_utilization = 0
        
        # Max drawdown (simplified - would need trade history for accurate)
        max_drawdown = min(0, grid.unrealized_pnl) if grid.net_position != 0 else 0
        
        return GridPerformance(
            grid_id=grid.id,
            pair=grid.pair,
            duration_hours=duration_hours,
            total_trades=total_trades,
            trades_per_hour=trades_per_hour,
            win_rate=win_rate,
            realized_pnl=grid.realized_pnl,
            pnl_per_trade=pnl_per_trade,
            pnl_per_hour=pnl_per_hour,
            max_drawdown=max_drawdown,
            efficiency=efficiency,
            range_utilization=range_utilization
        )
    
    def optimize_grid(self, pair: str, current_config: Dict, 
                      market: MarketCondition,
                      performance: Optional[GridPerformance] = None) -> OptimizationResult:
        """Generate optimization suggestions for a grid"""
        changes = []
        suggested = current_config.copy()
        reasons = []
        
        # === 1. Optimize Grid Mode ===
        if market.confidence > 50:
            if current_config.get("grid_mode") != market.recommended_mode:
                if market.trend_strength > 65:
                    suggested["grid_mode"] = market.recommended_mode
                    changes.append(f"grid_mode: {current_config.get('grid_mode')} → {market.recommended_mode}")
                    reasons.append(f"{market.trend} trend detected (strength: {market.trend_strength:.0f}%)")
        
        # === 2. Optimize Range ===
        current_range = current_config.get("range_percent", 10)
        
        # Adjust based on volatility
        optimal_range = market.recommended_range
        
        # Adjust based on performance
        if performance:
            if performance.range_utilization < 30:
                # Range too wide, price not reaching edges
                optimal_range = current_range * 0.75
                reasons.append(f"low range utilization ({performance.range_utilization:.0f}%)")
            elif performance.range_utilization > 90:
                # Range too tight, price hitting edges
                optimal_range = current_range * 1.25
                reasons.append(f"high range utilization ({performance.range_utilization:.0f}%)")
        
        optimal_range = max(self.MIN_RANGE_PERCENT, 
                           min(self.MAX_RANGE_PERCENT, optimal_range))
        
        if abs(optimal_range - current_range) > 2:
            suggested["range_percent"] = round(optimal_range, 1)
            changes.append(f"range_percent: {current_range} → {optimal_range:.1f}")
        
        # === 3. Optimize Grid Levels ===
        current_levels = current_config.get("grid_levels", 15)
        optimal_levels = market.recommended_levels
        
        # Adjust based on performance
        if performance:
            if performance.trades_per_hour < 0.5 and performance.efficiency < 30:
                # Too few trades, reduce levels for larger spacing
                optimal_levels = int(current_levels * 0.8)
                reasons.append(f"low trade frequency ({performance.trades_per_hour:.2f}/hr)")
            elif performance.trades_per_hour > 5:
                # Very active, could add more levels
                optimal_levels = int(current_levels * 1.2)
                reasons.append(f"high trade frequency ({performance.trades_per_hour:.2f}/hr)")
        
        optimal_levels = max(self.MIN_GRID_LEVELS, 
                            min(self.MAX_GRID_LEVELS, optimal_levels))
        
        if abs(optimal_levels - current_levels) >= 3:
            suggested["grid_levels"] = optimal_levels
            changes.append(f"grid_levels: {current_levels} → {optimal_levels}")
        
        # === 4. Optimize Min Profit ===
        current_min_profit = current_config.get("min_profit_percent", 0.5)
        
        # Higher volatility = can require higher min profit
        optimal_min_profit = 0.3 + (market.volatility * 0.03)
        
        # Adjust based on performance
        if performance and performance.total_trades >= self.MIN_TRADES_FOR_ANALYSIS:
            if performance.win_rate < 60:
                # Low win rate, increase min profit requirement
                optimal_min_profit *= 1.2
                reasons.append(f"low win rate ({performance.win_rate:.0f}%)")
            elif performance.win_rate > 90 and performance.pnl_per_trade < 0.1:
                # High win rate but low profit per trade
                optimal_min_profit *= 1.1
        
        optimal_min_profit = max(self.MIN_PROFIT_PERCENT, 
                                min(self.MAX_PROFIT_PERCENT, optimal_min_profit))
        
        if abs(optimal_min_profit - current_min_profit) > 0.1:
            suggested["min_profit_percent"] = round(optimal_min_profit, 2)
            changes.append(f"min_profit_percent: {current_min_profit} → {optimal_min_profit:.2f}")
        
        # === 5. Optimize Spacing ===
        # Use geometric for high volatility, arithmetic for low
        current_spacing = current_config.get("grid_spacing", "arithmetic")
        optimal_spacing = "geometric" if market.volatility > 8 else "arithmetic"
        
        if current_spacing != optimal_spacing:
            suggested["grid_spacing"] = optimal_spacing
            changes.append(f"grid_spacing: {current_spacing} → {optimal_spacing}")
            reasons.append(f"volatility {market.volatility:.1f}%")
        
        # Calculate expected improvement
        expected_improvement = 0
        if changes:
            # Rough estimate based on number and type of changes
            expected_improvement = min(50, len(changes) * 10)
            if performance and performance.pnl_per_hour < 0:
                expected_improvement += 20  # Bigger improvement potential if currently losing
        
        # Confidence based on data quality
        confidence = market.confidence
        if performance:
            if performance.total_trades >= 20:
                confidence = min(100, confidence + 20)
            if performance.duration_hours >= 24:
                confidence = min(100, confidence + 10)
        
        return OptimizationResult(
            pair=pair,
            current_params=current_config,
            suggested_params=suggested,
            changes=changes,
            expected_improvement=expected_improvement,
            confidence=confidence,
            reason="; ".join(reasons) if reasons else "market conditions"
        )
    
    def should_apply_optimization(self, result: OptimizationResult, 
                                  min_confidence: float = 60) -> bool:
        """Determine if optimization should be applied"""
        if not result.changes:
            return False
        
        if result.confidence < min_confidence:
            return False
        
        # Don't optimize if only minor changes
        if result.expected_improvement < 10:
            return False
        
        return True
    
    def get_optimization_summary(self, results: List[OptimizationResult]) -> Dict:
        """Get summary of all optimization results"""
        total_pairs = len(results)
        pairs_to_optimize = sum(1 for r in results if self.should_apply_optimization(r))
        
        all_changes = []
        for r in results:
            for change in r.changes:
                all_changes.append(f"{r.pair}: {change}")
        
        avg_improvement = sum(r.expected_improvement for r in results) / total_pairs if total_pairs > 0 else 0
        avg_confidence = sum(r.confidence for r in results) / total_pairs if total_pairs > 0 else 0
        
        return {
            "total_pairs": total_pairs,
            "pairs_to_optimize": pairs_to_optimize,
            "all_changes": all_changes,
            "avg_expected_improvement": round(avg_improvement, 1),
            "avg_confidence": round(avg_confidence, 1)
        }


class AutoPilot:
    """
    Automatic strategy management
    Runs continuously and applies optimizations
    """
    
    def __init__(self, db: Database, optimizer: GridOptimizer, logger: logging.Logger):
        self.db = db
        self.optimizer = optimizer
        self.logger = logger
        self._optimization_history: Dict[str, List[Dict]] = {}
    
    def record_optimization(self, pair: str, result: OptimizationResult, applied: bool):
        """Record optimization for tracking"""
        if pair not in self._optimization_history:
            self._optimization_history[pair] = []
        
        self._optimization_history[pair].append({
            "timestamp": datetime.utcnow().isoformat(),
            "changes": result.changes,
            "expected_improvement": result.expected_improvement,
            "confidence": result.confidence,
            "applied": applied
        })
        
        # Keep only last 50 records per pair
        if len(self._optimization_history[pair]) > 50:
            self._optimization_history[pair] = self._optimization_history[pair][-50:]
    
    def get_pair_history(self, pair: str) -> List[Dict]:
        """Get optimization history for a pair"""
        return self._optimization_history.get(pair, [])
    
    def calculate_capital_allocation(self, pairs: List[str], 
                                     total_capital: float,
                                     performances: Dict[str, GridPerformance]) -> Dict[str, float]:
        """
        Calculate optimal capital allocation based on performance
        Better performing pairs get more capital
        """
        allocations = {}
        
        # Calculate scores for each pair
        scores = {}
        for pair in pairs:
            perf = performances.get(pair)
            if perf and perf.total_trades >= 5:
                # Score based on PnL per hour and win rate
                pnl_score = max(0, perf.pnl_per_hour + 1) * 10  # Normalize around 0
                win_score = perf.win_rate / 100
                efficiency_score = perf.efficiency / 100
                
                scores[pair] = (pnl_score * 0.5 + win_score * 0.3 + efficiency_score * 0.2)
            else:
                # No data = neutral score
                scores[pair] = 1.0
        
        # Normalize scores to sum to 1
        total_score = sum(scores.values())
        if total_score > 0:
            for pair in pairs:
                base_allocation = total_capital / len(pairs)  # Equal base
                performance_adjustment = (scores[pair] / total_score) * total_capital
                
                # Blend: 50% equal, 50% performance-based
                allocations[pair] = (base_allocation * 0.5 + performance_adjustment * 0.5)
        else:
            # Equal allocation if no scores
            for pair in pairs:
                allocations[pair] = total_capital / len(pairs)
        
        return allocations
    
    def suggest_new_pairs(self, current_pairs: List[str], 
                          available_pairs: List[str],
                          market_conditions: Dict[str, MarketCondition]) -> List[str]:
        """Suggest new pairs to add based on market conditions"""
        suggestions = []
        
        for pair in available_pairs:
            if pair in current_pairs:
                continue
            
            condition = market_conditions.get(pair)
            if not condition:
                continue
            
            # Good candidates: sideways market with decent volatility
            if (condition.trend == "sideways" and 
                condition.trend_strength < 60 and
                3 < condition.volatility < 15):
                suggestions.append(pair)
        
        return suggestions[:5]  # Max 5 suggestions
