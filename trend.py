from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

@dataclass
class TrendSignal:
    direction: str  # long / short / neutral
    strength: float  # 0.0 - 1.0
    indicator: str
    details: Dict

class TrendDetector:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
    
    def detect(self, klines: List[Dict], indicator: str, params: Dict) -> TrendSignal:
        """Detect trend using specified indicator"""
        if indicator == "ema_cross":
            return self._ema_cross(klines, params)
        elif indicator == "ema_price":
            return self._ema_price(klines, params)
        elif indicator == "adx":
            return self._adx(klines, params)
        elif indicator == "supertrend":
            return self._supertrend(klines, params)
        else:
            raise ValueError(f"Unknown indicator: {indicator}")
    
    def _ema(self, values: List[float], period: int) -> List[float]:
        """Calculate EMA"""
        if len(values) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema = [sum(values[:period]) / period]
        
        for price in values[period:]:
            ema.append((price - ema[-1]) * multiplier + ema[-1])
        
        return ema
    
    def _sma(self, values: List[float], period: int) -> List[float]:
        """Calculate SMA"""
        if len(values) < period:
            return []
        return [sum(values[i:i+period]) / period for i in range(len(values) - period + 1)]
    
    def _atr(self, klines: List[Dict], period: int = 14) -> List[float]:
        """Calculate ATR"""
        if len(klines) < period + 1:
            return []
        
        tr_list = []
        for i in range(1, len(klines)):
            high = klines[i]["high"]
            low = klines[i]["low"]
            prev_close = klines[i-1]["close"]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr)
        
        # First ATR is SMA
        atr = [sum(tr_list[:period]) / period]
        
        # Subsequent ATRs use smoothing
        for tr in tr_list[period:]:
            atr.append((atr[-1] * (period - 1) + tr) / period)
        
        return atr
    
    def _ema_cross(self, klines: List[Dict], params: Dict) -> TrendSignal:
        """EMA crossover strategy"""
        fast_period = params.get("fast", 20)
        slow_period = params.get("slow", 50)
        
        closes = [k["close"] for k in klines]
        
        fast_ema = self._ema(closes, fast_period)
        slow_ema = self._ema(closes, slow_period)
        
        if not fast_ema or not slow_ema:
            return TrendSignal("neutral", 0, "ema_cross", {})
        
        # Align arrays - slow EMA starts later
        offset = slow_period - fast_period
        fast_ema = fast_ema[offset:]
        
        if len(fast_ema) < 2 or len(slow_ema) < 2:
            return TrendSignal("neutral", 0, "ema_cross", {})
        
        current_fast = fast_ema[-1]
        current_slow = slow_ema[-1]
        prev_fast = fast_ema[-2]
        prev_slow = slow_ema[-2]
        
        # Calculate strength based on separation
        separation = abs(current_fast - current_slow) / current_slow
        strength = min(separation * 10, 1.0)
        
        # Determine direction
        if current_fast > current_slow:
            direction = "long"
            # Fresh cross is stronger
            if prev_fast <= prev_slow:
                strength = min(strength + 0.3, 1.0)
        elif current_fast < current_slow:
            direction = "short"
            if prev_fast >= prev_slow:
                strength = min(strength + 0.3, 1.0)
        else:
            direction = "neutral"
            strength = 0
        
        return TrendSignal(
            direction=direction,
            strength=strength,
            indicator="ema_cross",
            details={
                "fast_ema": current_fast,
                "slow_ema": current_slow,
                "fast_period": fast_period,
                "slow_period": slow_period
            }
        )
    
    def _ema_price(self, klines: List[Dict], params: Dict) -> TrendSignal:
        """Price vs EMA strategy"""
        period = params.get("period", 200)
        
        closes = [k["close"] for k in klines]
        ema = self._ema(closes, period)
        
        if not ema:
            return TrendSignal("neutral", 0, "ema_price", {})
        
        current_price = closes[-1]
        current_ema = ema[-1]
        
        # Calculate strength based on distance from EMA
        distance = (current_price - current_ema) / current_ema
        strength = min(abs(distance) * 5, 1.0)
        
        if current_price > current_ema:
            direction = "long"
        elif current_price < current_ema:
            direction = "short"
        else:
            direction = "neutral"
            strength = 0
        
        return TrendSignal(
            direction=direction,
            strength=strength,
            indicator="ema_price",
            details={
                "price": current_price,
                "ema": current_ema,
                "period": period,
                "distance_percent": distance * 100
            }
        )
    
    def _adx(self, klines: List[Dict], params: Dict) -> TrendSignal:
        """ADX + DI strategy"""
        period = params.get("period", 14)
        adx_threshold = params.get("threshold", 25)
        
        if len(klines) < period * 2:
            return TrendSignal("neutral", 0, "adx", {})
        
        # Calculate +DM and -DM
        plus_dm = []
        minus_dm = []
        
        for i in range(1, len(klines)):
            high_diff = klines[i]["high"] - klines[i-1]["high"]
            low_diff = klines[i-1]["low"] - klines[i]["low"]
            
            if high_diff > low_diff and high_diff > 0:
                plus_dm.append(high_diff)
            else:
                plus_dm.append(0)
            
            if low_diff > high_diff and low_diff > 0:
                minus_dm.append(low_diff)
            else:
                minus_dm.append(0)
        
        # Calculate ATR
        atr = self._atr(klines, period)
        if not atr:
            return TrendSignal("neutral", 0, "adx", {})
        
        # Smooth DM values
        def smooth(values: List[float], period: int) -> List[float]:
            if len(values) < period:
                return []
            result = [sum(values[:period])]
            for v in values[period:]:
                result.append(result[-1] - result[-1]/period + v)
            return result
        
        smooth_plus_dm = smooth(plus_dm, period)
        smooth_minus_dm = smooth(minus_dm, period)
        
        if not smooth_plus_dm or not smooth_minus_dm:
            return TrendSignal("neutral", 0, "adx", {})
        
        # Calculate +DI and -DI
        min_len = min(len(smooth_plus_dm), len(smooth_minus_dm), len(atr))
        plus_di = [100 * smooth_plus_dm[i] / atr[i] if atr[i] > 0 else 0 
                   for i in range(min_len)]
        minus_di = [100 * smooth_minus_dm[i] / atr[i] if atr[i] > 0 else 0 
                    for i in range(min_len)]
        
        # Calculate DX and ADX
        dx = []
        for i in range(len(plus_di)):
            di_sum = plus_di[i] + minus_di[i]
            if di_sum > 0:
                dx.append(100 * abs(plus_di[i] - minus_di[i]) / di_sum)
            else:
                dx.append(0)
        
        adx_values = self._ema(dx, period)
        
        if not adx_values:
            return TrendSignal("neutral", 0, "adx", {})
        
        current_adx = adx_values[-1]
        current_plus_di = plus_di[-1]
        current_minus_di = minus_di[-1]
        
        # Determine direction and strength
        if current_adx < adx_threshold:
            direction = "neutral"
            strength = 0
        elif current_plus_di > current_minus_di:
            direction = "long"
            strength = min(current_adx / 50, 1.0)
        else:
            direction = "short"
            strength = min(current_adx / 50, 1.0)
        
        return TrendSignal(
            direction=direction,
            strength=strength,
            indicator="adx",
            details={
                "adx": current_adx,
                "plus_di": current_plus_di,
                "minus_di": current_minus_di,
                "threshold": adx_threshold
            }
        )
    
    def _supertrend(self, klines: List[Dict], params: Dict) -> TrendSignal:
        """Supertrend strategy"""
        period = params.get("period", 10)
        multiplier = params.get("multiplier", 3.0)
        
        if len(klines) < period + 1:
            return TrendSignal("neutral", 0, "supertrend", {})
        
        atr = self._atr(klines, period)
        if not atr:
            return TrendSignal("neutral", 0, "supertrend", {})
        
        # Calculate basic upper and lower bands
        supertrend = []
        direction_list = []
        
        for i in range(len(atr)):
            idx = i + period  # Offset for klines
            if idx >= len(klines):
                break
            
            hl2 = (klines[idx]["high"] + klines[idx]["low"]) / 2
            upper = hl2 + multiplier * atr[i]
            lower = hl2 - multiplier * atr[i]
            
            close = klines[idx]["close"]
            prev_close = klines[idx-1]["close"] if idx > 0 else close
            
            if i == 0:
                st = upper if close <= upper else lower
                direction = -1 if close <= upper else 1
            else:
                prev_st = supertrend[-1]
                prev_dir = direction_list[-1]
                
                if prev_dir == 1:  # Previous was uptrend
                    if close > prev_st:
                        st = max(lower, prev_st)
                        direction = 1
                    else:
                        st = upper
                        direction = -1
                else:  # Previous was downtrend
                    if close < prev_st:
                        st = min(upper, prev_st)
                        direction = -1
                    else:
                        st = lower
                        direction = 1
            
            supertrend.append(st)
            direction_list.append(direction)
        
        if not direction_list:
            return TrendSignal("neutral", 0, "supertrend", {})
        
        current_dir = direction_list[-1]
        current_st = supertrend[-1]
        current_price = klines[-1]["close"]
        
        # Calculate strength based on distance from supertrend
        distance = abs(current_price - current_st) / current_st
        strength = min(distance * 10, 1.0)
        
        if current_dir == 1:
            trend_direction = "long"
        else:
            trend_direction = "short"
        
        return TrendSignal(
            direction=trend_direction,
            strength=strength,
            indicator="supertrend",
            details={
                "supertrend": current_st,
                "price": current_price,
                "period": period,
                "multiplier": multiplier
            }
        )
