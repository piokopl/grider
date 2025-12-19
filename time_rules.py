"""
Time-Based Trading Rules
Adjust parameters based on time of day, day of week, and market sessions
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import pytz


class MarketSession(Enum):
    ASIA = "asia"          # Tokyo, Hong Kong, Singapore
    EUROPE = "europe"      # London, Frankfurt
    US = "us"              # New York
    OVERLAP_EU_US = "eu_us_overlap"
    OVERLAP_ASIA_EU = "asia_eu_overlap"
    WEEKEND = "weekend"
    OFF_HOURS = "off_hours"


@dataclass
class SessionConfig:
    """Configuration for a trading session"""
    enabled: bool = True
    position_multiplier: float = 1.0  # Scale position size
    range_multiplier: float = 1.0     # Scale grid range
    min_profit_multiplier: float = 1.0  # Scale min profit
    max_trades_per_hour: int = 50
    pause_trading: bool = False


@dataclass
class TimeRulesConfig:
    # Enable/disable
    enabled: bool = True
    
    # Timezone for calculations
    timezone: str = "UTC"
    
    # Session definitions (in UTC)
    asia_start: str = "00:00"     # 00:00-08:00 UTC
    asia_end: str = "08:00"
    europe_start: str = "07:00"   # 07:00-16:00 UTC
    europe_end: str = "16:00"
    us_start: str = "13:00"       # 13:00-21:00 UTC
    us_end: str = "21:00"
    
    # Session-specific configs
    sessions: Dict[str, SessionConfig] = field(default_factory=lambda: {
        "asia": SessionConfig(position_multiplier=0.8),
        "europe": SessionConfig(position_multiplier=1.0),
        "us": SessionConfig(position_multiplier=1.0),
        "eu_us_overlap": SessionConfig(position_multiplier=1.2, range_multiplier=1.1),
        "asia_eu_overlap": SessionConfig(position_multiplier=1.0),
        "weekend": SessionConfig(position_multiplier=0.5, range_multiplier=1.3, min_profit_multiplier=1.2),
        "off_hours": SessionConfig(position_multiplier=0.6)
    })
    
    # Weekend settings
    reduce_on_weekend: bool = True
    weekend_position_multiplier: float = 0.5
    
    # Low volume hours
    low_volume_hours: List[int] = field(default_factory=lambda: [2, 3, 4, 5])  # UTC
    low_volume_multiplier: float = 0.7
    
    # Special dates (holidays, events)
    pause_dates: List[str] = field(default_factory=list)  # ["2024-12-25", "2024-01-01"]
    
    # Funding rate times (typically 00:00, 08:00, 16:00 UTC)
    funding_times: List[str] = field(default_factory=lambda: ["00:00", "08:00", "16:00"])
    pause_before_funding_minutes: int = 5
    pause_after_funding_minutes: int = 5


class TimeBasedRules:
    """
    Apply time-based modifications to trading parameters
    """
    
    def __init__(self, config: TimeRulesConfig, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self._tz = pytz.timezone(config.timezone)
    
    def get_current_session(self) -> MarketSession:
        """Determine current market session"""
        now = datetime.now(self._tz)
        current_time = now.time()
        weekday = now.weekday()
        
        # Check weekend
        if weekday >= 5:  # Saturday = 5, Sunday = 6
            return MarketSession.WEEKEND
        
        # Parse session times
        asia_start = self._parse_time(self.config.asia_start)
        asia_end = self._parse_time(self.config.asia_end)
        europe_start = self._parse_time(self.config.europe_start)
        europe_end = self._parse_time(self.config.europe_end)
        us_start = self._parse_time(self.config.us_start)
        us_end = self._parse_time(self.config.us_end)
        
        # Check overlaps first (highest priority)
        if self._in_time_range(current_time, europe_start, asia_end):
            return MarketSession.OVERLAP_ASIA_EU
        
        if self._in_time_range(current_time, us_start, europe_end):
            return MarketSession.OVERLAP_EU_US
        
        # Check individual sessions
        if self._in_time_range(current_time, asia_start, asia_end):
            return MarketSession.ASIA
        
        if self._in_time_range(current_time, europe_start, europe_end):
            return MarketSession.EUROPE
        
        if self._in_time_range(current_time, us_start, us_end):
            return MarketSession.US
        
        return MarketSession.OFF_HOURS
    
    def get_session_config(self, session: Optional[MarketSession] = None) -> SessionConfig:
        """Get configuration for current or specified session"""
        if session is None:
            session = self.get_current_session()
        
        session_key = session.value
        if session_key in self.config.sessions:
            return self.config.sessions[session_key]
        
        return SessionConfig()
    
    def get_modifiers(self) -> Dict[str, float]:
        """
        Get all applicable modifiers for current time
        Returns multipliers for various parameters
        """
        if not self.config.enabled:
            return {
                "position": 1.0,
                "range": 1.0,
                "min_profit": 1.0,
                "max_trades": 50
            }
        
        now = datetime.now(self._tz)
        session = self.get_current_session()
        session_config = self.get_session_config(session)
        
        # Start with session defaults
        position_mult = session_config.position_multiplier
        range_mult = session_config.range_multiplier
        min_profit_mult = session_config.min_profit_multiplier
        max_trades = session_config.max_trades_per_hour
        
        # Apply low volume modifier
        if now.hour in self.config.low_volume_hours:
            position_mult *= self.config.low_volume_multiplier
        
        # Apply weekend modifier (additional)
        if session == MarketSession.WEEKEND and self.config.reduce_on_weekend:
            position_mult *= self.config.weekend_position_multiplier
        
        return {
            "position": position_mult,
            "range": range_mult,
            "min_profit": min_profit_mult,
            "max_trades": max_trades,
            "session": session.value
        }
    
    def should_pause_trading(self) -> Tuple[bool, str]:
        """
        Check if trading should be paused
        Returns: (should_pause, reason)
        """
        if not self.config.enabled:
            return False, ""
        
        now = datetime.now(self._tz)
        
        # Check pause dates
        today = now.date().isoformat()
        if today in self.config.pause_dates:
            return True, f"Pause date: {today}"
        
        # Check session pause
        session = self.get_current_session()
        session_config = self.get_session_config(session)
        if session_config.pause_trading:
            return True, f"Session paused: {session.value}"
        
        # Check funding rate times
        for funding_time in self.config.funding_times:
            ft = self._parse_time(funding_time)
            funding_dt = now.replace(hour=ft.hour, minute=ft.minute, second=0, microsecond=0)
            
            # Check before funding
            before_start = funding_dt - timedelta(minutes=self.config.pause_before_funding_minutes)
            if before_start <= now < funding_dt:
                return True, f"Pause before funding at {funding_time}"
            
            # Check after funding
            after_end = funding_dt + timedelta(minutes=self.config.pause_after_funding_minutes)
            if funding_dt <= now < after_end:
                return True, f"Pause after funding at {funding_time}"
        
        return False, ""
    
    def get_next_event(self) -> Tuple[str, datetime]:
        """Get next significant time event"""
        now = datetime.now(self._tz)
        events = []
        
        # Session changes
        for session_name, start_time in [
            ("asia", self.config.asia_start),
            ("europe", self.config.europe_start),
            ("us", self.config.us_start)
        ]:
            t = self._parse_time(start_time)
            event_dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            if event_dt <= now:
                event_dt += timedelta(days=1)
            events.append((f"{session_name}_start", event_dt))
        
        # Funding times
        for funding_time in self.config.funding_times:
            t = self._parse_time(funding_time)
            event_dt = now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
            if event_dt <= now:
                event_dt += timedelta(days=1)
            events.append((f"funding_{funding_time}", event_dt))
        
        # Weekend
        days_until_weekend = (5 - now.weekday()) % 7
        if days_until_weekend == 0 and now.weekday() < 5:
            days_until_weekend = 5 - now.weekday()
        weekend_start = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_until_weekend)
        events.append(("weekend_start", weekend_start))
        
        # Find nearest
        events.sort(key=lambda x: x[1])
        return events[0] if events else ("none", now)
    
    def apply_modifiers_to_config(self, base_config: Dict) -> Dict:
        """Apply time-based modifiers to a grid config"""
        modifiers = self.get_modifiers()
        modified = base_config.copy()
        
        # Apply position modifier to capital
        if "total_capital" in modified:
            modified["total_capital"] *= modifiers["position"]
        
        # Apply range modifier
        if "range_percent" in modified:
            modified["range_percent"] *= modifiers["range"]
        
        # Apply min profit modifier
        if "min_profit_percent" in modified:
            modified["min_profit_percent"] *= modifiers["min_profit"]
        
        return modified
    
    def _parse_time(self, time_str: str) -> time:
        """Parse time string HH:MM to time object"""
        parts = time_str.split(":")
        return time(int(parts[0]), int(parts[1]))
    
    def _in_time_range(self, current: time, start: time, end: time) -> bool:
        """Check if current time is in range (handles overnight ranges)"""
        if start <= end:
            return start <= current < end
        else:
            # Overnight range (e.g., 22:00 - 06:00)
            return current >= start or current < end
    
    def get_status(self) -> Dict:
        """Get current time rules status"""
        session = self.get_current_session()
        modifiers = self.get_modifiers()
        should_pause, pause_reason = self.should_pause_trading()
        next_event, next_time = self.get_next_event()
        
        return {
            "enabled": self.config.enabled,
            "current_session": session.value,
            "modifiers": modifiers,
            "should_pause": should_pause,
            "pause_reason": pause_reason,
            "next_event": next_event,
            "next_event_time": next_time.isoformat(),
            "current_time": datetime.now(self._tz).isoformat()
        }


class ScheduledActions:
    """
    Schedule specific actions at specific times
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._scheduled: List[Dict] = []
    
    def schedule_action(self, action_type: str, execute_at: datetime,
                       params: Dict = None, repeat: str = None):
        """
        Schedule an action
        
        repeat options: "daily", "hourly", "weekly", None
        """
        self._scheduled.append({
            "type": action_type,
            "execute_at": execute_at,
            "params": params or {},
            "repeat": repeat,
            "executed": False
        })
    
    def get_due_actions(self) -> List[Dict]:
        """Get actions that are due for execution"""
        now = datetime.utcnow()
        due = []
        
        for action in self._scheduled:
            if not action["executed"] and action["execute_at"] <= now:
                due.append(action)
                action["executed"] = True
                
                # Reschedule if repeating
                if action["repeat"]:
                    next_time = self._calculate_next_time(action["execute_at"], action["repeat"])
                    self.schedule_action(
                        action["type"],
                        next_time,
                        action["params"],
                        action["repeat"]
                    )
        
        # Clean up executed non-repeating actions
        self._scheduled = [a for a in self._scheduled if not a["executed"] or a["repeat"]]
        
        return due
    
    def _calculate_next_time(self, current: datetime, repeat: str) -> datetime:
        """Calculate next execution time"""
        if repeat == "hourly":
            return current + timedelta(hours=1)
        elif repeat == "daily":
            return current + timedelta(days=1)
        elif repeat == "weekly":
            return current + timedelta(weeks=1)
        return current
    
    def get_scheduled(self) -> List[Dict]:
        """Get all scheduled actions"""
        return [
            {
                "type": a["type"],
                "execute_at": a["execute_at"].isoformat(),
                "params": a["params"],
                "repeat": a["repeat"]
            }
            for a in self._scheduled
            if not a["executed"]
        ]
