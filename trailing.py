"""
Trailing Grid System
Grid that follows price instead of waiting for rebuild
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import math


@dataclass
class TrailingConfig:
    enabled: bool = True
    
    # When to move grid
    trigger_percent: float = 2.0      # Move when price is X% from center
    move_percent: float = 1.0         # How much to move (% of range)
    
    # Constraints
    min_move_interval_minutes: int = 5   # Min time between moves
    max_moves_per_hour: int = 10         # Max moves per hour
    
    # Mode
    follow_trend: bool = True         # Move in trend direction only
    symmetric: bool = False           # Move both bounds equally
    
    # Position handling
    cancel_far_orders: bool = True    # Cancel orders far from new range
    far_order_threshold_percent: float = 3.0  # What's "far"


class TrailingGrid:
    """
    Manages trailing grid that follows price movements
    """
    
    def __init__(self, config: TrailingConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        
        self._last_move_time: Dict[str, datetime] = {}
        self._moves_this_hour: Dict[str, List[datetime]] = {}
        self._grid_centers: Dict[str, float] = {}
    
    def initialize_grid(self, pair: str, lower: float, upper: float):
        """Initialize tracking for a grid"""
        center = (lower + upper) / 2
        self._grid_centers[pair] = center
        self._moves_this_hour[pair] = []
    
    def should_trail(self, pair: str, current_price: float,
                    lower: float, upper: float,
                    trend: Optional[str] = None) -> Tuple[bool, float, float]:
        """
        Check if grid should trail
        Returns: (should_move, new_lower, new_upper)
        """
        if not self.config.enabled:
            return False, lower, upper
        
        # Check rate limits
        if not self._can_move(pair):
            return False, lower, upper
        
        center = (lower + upper) / 2
        grid_range = upper - lower
        
        # Calculate distance from center
        distance_percent = (current_price - center) / center * 100
        
        # Check if trigger is met
        if abs(distance_percent) < self.config.trigger_percent:
            return False, lower, upper
        
        # Check trend filter
        if self.config.follow_trend and trend:
            if trend == "bullish" and distance_percent < 0:
                return False, lower, upper  # Price down but trend up
            if trend == "bearish" and distance_percent > 0:
                return False, lower, upper  # Price up but trend down
        
        # Calculate move amount
        move_amount = grid_range * (self.config.move_percent / 100)
        
        if distance_percent > 0:
            # Price above center - move up
            if self.config.symmetric:
                new_lower = lower + move_amount
                new_upper = upper + move_amount
            else:
                new_upper = upper + move_amount
                new_lower = lower + move_amount * 0.5  # Move lower less
        else:
            # Price below center - move down
            if self.config.symmetric:
                new_lower = lower - move_amount
                new_upper = upper - move_amount
            else:
                new_lower = lower - move_amount
                new_upper = upper - move_amount * 0.5  # Move upper less
        
        self._record_move(pair)
        self._grid_centers[pair] = (new_lower + new_upper) / 2
        
        self.logger.info(f"{pair} Trailing: ${lower:.2f}-${upper:.2f} → "
                        f"${new_lower:.2f}-${new_upper:.2f} "
                        f"(price: ${current_price:.2f})")
        
        return True, new_lower, new_upper
    
    def get_orders_to_cancel(self, pair: str, order_prices: List[float],
                            new_lower: float, new_upper: float) -> List[float]:
        """Get order prices that are too far from new range"""
        if not self.config.cancel_far_orders:
            return []
        
        far_orders = []
        range_size = new_upper - new_lower
        threshold = range_size * (self.config.far_order_threshold_percent / 100)
        
        for price in order_prices:
            if price < new_lower - threshold or price > new_upper + threshold:
                far_orders.append(price)
        
        return far_orders
    
    def _can_move(self, pair: str) -> bool:
        """Check if move is allowed (rate limiting)"""
        now = datetime.utcnow()
        
        # Check min interval
        last_move = self._last_move_time.get(pair)
        if last_move:
            elapsed = (now - last_move).total_seconds() / 60
            if elapsed < self.config.min_move_interval_minutes:
                return False
        
        # Check hourly limit
        hour_ago = now - timedelta(hours=1)
        if pair in self._moves_this_hour:
            self._moves_this_hour[pair] = [
                t for t in self._moves_this_hour[pair] if t > hour_ago
            ]
            if len(self._moves_this_hour[pair]) >= self.config.max_moves_per_hour:
                return False
        
        return True
    
    def _record_move(self, pair: str):
        """Record a grid move"""
        now = datetime.utcnow()
        self._last_move_time[pair] = now
        
        if pair not in self._moves_this_hour:
            self._moves_this_hour[pair] = []
        self._moves_this_hour[pair].append(now)
    
    def get_trailing_stats(self, pair: str) -> Dict:
        """Get trailing statistics for a pair"""
        now = datetime.utcnow()
        hour_ago = now - timedelta(hours=1)
        
        moves_last_hour = 0
        if pair in self._moves_this_hour:
            moves_last_hour = sum(
                1 for t in self._moves_this_hour[pair] if t > hour_ago
            )
        
        return {
            "enabled": self.config.enabled,
            "moves_last_hour": moves_last_hour,
            "max_moves_per_hour": self.config.max_moves_per_hour,
            "last_move": self._last_move_time.get(pair, "never"),
            "current_center": self._grid_centers.get(pair, 0)
        }


class AdaptiveTrailing:
    """
    Advanced trailing that adapts based on market conditions
    """
    
    def __init__(self, base_config: TrailingConfig, logger: logging.Logger):
        self.base_config = base_config
        self.logger = logger
        self._volatility_cache: Dict[str, float] = {}
    
    def update_volatility(self, pair: str, volatility: float):
        """Update volatility for adaptive calculations"""
        self._volatility_cache[pair] = volatility
    
    def get_adaptive_config(self, pair: str) -> TrailingConfig:
        """Get trailing config adapted to current volatility"""
        vol = self._volatility_cache.get(pair, 5.0)
        
        # Create adapted config
        adapted = TrailingConfig(
            enabled=self.base_config.enabled,
            trigger_percent=self.base_config.trigger_percent,
            move_percent=self.base_config.move_percent,
            min_move_interval_minutes=self.base_config.min_move_interval_minutes,
            max_moves_per_hour=self.base_config.max_moves_per_hour,
            follow_trend=self.base_config.follow_trend,
            symmetric=self.base_config.symmetric,
            cancel_far_orders=self.base_config.cancel_far_orders,
            far_order_threshold_percent=self.base_config.far_order_threshold_percent
        )
        
        # High volatility = larger trigger, larger moves, less frequent
        if vol > 10:
            adapted.trigger_percent *= 1.5
            adapted.move_percent *= 1.5
            adapted.min_move_interval_minutes = max(10, adapted.min_move_interval_minutes)
            adapted.max_moves_per_hour = min(5, adapted.max_moves_per_hour)
        
        # Low volatility = smaller trigger, smaller moves, more frequent
        elif vol < 3:
            adapted.trigger_percent *= 0.7
            adapted.move_percent *= 0.7
            adapted.min_move_interval_minutes = max(2, adapted.min_move_interval_minutes - 2)
            adapted.max_moves_per_hour = min(15, adapted.max_moves_per_hour + 5)
        
        return adapted
