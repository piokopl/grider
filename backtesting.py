"""
Backtesting Engine for Grid Bot
Test strategies on historical data
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import logging
import json
import math


@dataclass
class BacktestConfig:
    # Time range
    start_date: str = ""
    end_date: str = ""
    
    # Grid parameters to test
    pair: str = "BTCUSDT"
    initial_capital: float = 1000.0
    leverage: int = 5
    
    grid_mode: str = "neutral"
    grid_levels: int = 15
    range_percent: float = 10.0
    grid_spacing: str = "arithmetic"
    min_profit_percent: float = 0.4
    
    # Fees
    maker_fee: float = 0.02  # %
    taker_fee: float = 0.06  # %
    funding_rate: float = 0.01  # % per 8h (average)
    
    # Options
    enable_trailing: bool = False
    enable_auto_adjust: bool = True
    rebuild_on_exit: bool = True


@dataclass
class BacktestTrade:
    timestamp: str
    side: str  # buy / sell
    price: float
    amount: float
    fee: float
    pnl: float
    position_after: float
    equity_after: float


@dataclass
class BacktestResult:
    config: BacktestConfig
    
    # Performance
    total_pnl: float = 0.0
    total_pnl_percent: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Returns
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    profit_factor: float = 0.0
    
    # Risk
    max_drawdown_percent: float = 0.0
    max_drawdown_usd: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Activity
    avg_trades_per_day: float = 0.0
    total_fees: float = 0.0
    total_funding: float = 0.0
    grid_rebuilds: int = 0
    
    # Time analysis
    duration_days: float = 0.0
    time_in_position_percent: float = 0.0
    longest_drawdown_days: float = 0.0
    
    # Equity curve
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)
    trades: List[BacktestTrade] = field(default_factory=list)


class GridBacktester:
    """
    Backtest grid trading strategies
    """
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    async def run_backtest(self, config: BacktestConfig,
                          price_data: List[Dict]) -> BacktestResult:
        """
        Run backtest on historical price data
        
        price_data format: [{"timestamp": "...", "open": x, "high": x, "low": x, "close": x}, ...]
        """
        result = BacktestResult(config=config)
        
        if not price_data:
            self.logger.error("No price data provided")
            return result
        
        # Initialize state
        equity = config.initial_capital
        position = 0.0
        avg_entry_price = 0.0
        peak_equity = equity
        max_drawdown = 0.0
        
        # Grid state
        current_price = float(price_data[0]["close"])
        lower_price, upper_price = self._calculate_range(current_price, config.range_percent)
        grid_levels = self._create_grid_levels(lower_price, upper_price, config.grid_levels, config.grid_spacing)
        level_states = {level: {"bought": False, "buy_price": 0} for level in grid_levels}
        
        daily_returns = []
        last_equity = equity
        funding_periods = 0
        last_timestamp = None
        
        self.logger.info(f"Starting backtest: {config.pair} from {price_data[0]['timestamp']} to {price_data[-1]['timestamp']}")
        self.logger.info(f"Initial grid: ${lower_price:.2f} - ${upper_price:.2f}, {config.grid_levels} levels")
        
        for candle in price_data:
            timestamp = candle["timestamp"]
            high = float(candle["high"])
            low = float(candle["low"])
            close = float(candle["close"])
            
            # Check for grid rebuild
            if config.enable_auto_adjust and config.rebuild_on_exit:
                if close > upper_price * 1.01 or close < lower_price * 0.99:
                    # Rebuild grid
                    lower_price, upper_price = self._calculate_range(close, config.range_percent)
                    grid_levels = self._create_grid_levels(lower_price, upper_price, config.grid_levels, config.grid_spacing)
                    
                    # Keep position but reset level states
                    new_level_states = {level: {"bought": False, "buy_price": 0} for level in grid_levels}
                    
                    # Mark levels below current price as "bought" if we have position
                    if position > 0:
                        for level in grid_levels:
                            if level < close:
                                new_level_states[level]["bought"] = True
                                new_level_states[level]["buy_price"] = avg_entry_price
                    
                    level_states = new_level_states
                    result.grid_rebuilds += 1
            
            # Process each grid level
            for level in grid_levels:
                level_state = level_states[level]
                
                # Check for BUY (price crosses level going down)
                if not level_state["bought"] and low <= level < close:
                    # Buy triggered
                    buy_price = level
                    amount = (config.initial_capital / config.grid_levels) / buy_price
                    amount *= config.leverage
                    
                    fee = amount * buy_price * (config.taker_fee / 100)
                    
                    # Update position
                    if position == 0:
                        avg_entry_price = buy_price
                    else:
                        avg_entry_price = ((avg_entry_price * position) + (buy_price * amount)) / (position + amount)
                    
                    position += amount
                    equity -= fee
                    
                    level_state["bought"] = True
                    level_state["buy_price"] = buy_price
                    
                    result.trades.append(BacktestTrade(
                        timestamp=timestamp,
                        side="buy",
                        price=buy_price,
                        amount=amount,
                        fee=fee,
                        pnl=-fee,
                        position_after=position,
                        equity_after=equity
                    ))
                    result.total_trades += 1
                    result.total_fees += fee
                
                # Check for SELL (price crosses level going up, and we bought at this level)
                elif level_state["bought"] and high >= level > close:
                    # Calculate minimum profitable sell price
                    buy_price = level_state["buy_price"]
                    min_sell = buy_price * (1 + config.min_profit_percent / 100 + 2 * config.taker_fee / 100)
                    
                    if level >= min_sell:
                        sell_price = level
                        amount = (config.initial_capital / config.grid_levels) / buy_price
                        amount *= config.leverage
                        
                        fee = amount * sell_price * (config.taker_fee / 100)
                        trade_pnl = (sell_price - buy_price) * amount - fee * 2  # Both side fees
                        
                        position -= amount
                        equity += trade_pnl
                        
                        level_state["bought"] = False
                        level_state["buy_price"] = 0
                        
                        result.trades.append(BacktestTrade(
                            timestamp=timestamp,
                            side="sell",
                            price=sell_price,
                            amount=amount,
                            fee=fee,
                            pnl=trade_pnl,
                            position_after=position,
                            equity_after=equity
                        ))
                        result.total_trades += 1
                        result.total_fees += fee
                        
                        if trade_pnl > 0:
                            result.winning_trades += 1
                            result.best_trade = max(result.best_trade, trade_pnl)
                        else:
                            result.losing_trades += 1
                            result.worst_trade = min(result.worst_trade, trade_pnl)
            
            # Calculate unrealized PnL
            if position > 0:
                unrealized = (close - avg_entry_price) * position
            else:
                unrealized = 0
            
            current_equity = equity + unrealized
            
            # Update peak and drawdown
            if current_equity > peak_equity:
                peak_equity = current_equity
            
            drawdown = peak_equity - current_equity
            drawdown_percent = drawdown / peak_equity * 100 if peak_equity > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                result.max_drawdown_usd = drawdown
                result.max_drawdown_percent = drawdown_percent
            
            # Funding rate (every 8 hours)
            if last_timestamp:
                # Simple time check (assuming hourly data)
                funding_periods += 1
                if funding_periods >= 8:
                    if position > 0:
                        funding_cost = position * close * (config.funding_rate / 100)
                        equity -= funding_cost
                        result.total_funding += funding_cost
                    funding_periods = 0
            
            # Daily return tracking
            result.equity_curve.append((timestamp, current_equity))
            
            if last_equity > 0:
                daily_return = (current_equity - last_equity) / last_equity
                daily_returns.append(daily_return)
            
            last_equity = current_equity
            last_timestamp = timestamp
        
        # Final calculations
        final_equity = equity
        if position > 0:
            final_equity += (float(price_data[-1]["close"]) - avg_entry_price) * position
        
        result.total_pnl = final_equity - config.initial_capital
        result.total_pnl_percent = result.total_pnl / config.initial_capital * 100
        
        if result.total_trades > 0:
            result.win_rate = result.winning_trades / result.total_trades * 100
            result.avg_trade_pnl = result.total_pnl / result.total_trades
        
        # Calculate ratios
        if daily_returns:
            result.duration_days = len(price_data) / 24  # Assuming hourly data
            result.avg_trades_per_day = result.total_trades / max(1, result.duration_days)
            
            avg_return = sum(daily_returns) / len(daily_returns)
            std_return = math.sqrt(sum((r - avg_return) ** 2 for r in daily_returns) / len(daily_returns))
            
            if std_return > 0:
                result.sharpe_ratio = (avg_return * 365 - 0.05) / (std_return * math.sqrt(365))  # Annualized
            
            negative_returns = [r for r in daily_returns if r < 0]
            if negative_returns:
                downside_std = math.sqrt(sum(r ** 2 for r in negative_returns) / len(negative_returns))
                if downside_std > 0:
                    result.sortino_ratio = (avg_return * 365 - 0.05) / (downside_std * math.sqrt(365))
            
            if result.max_drawdown_percent > 0:
                annual_return = result.total_pnl_percent * (365 / max(1, result.duration_days))
                result.calmar_ratio = annual_return / result.max_drawdown_percent
        
        # Profit factor
        gross_profit = sum(t.pnl for t in result.trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in result.trades if t.pnl < 0))
        if gross_loss > 0:
            result.profit_factor = gross_profit / gross_loss
        
        self.logger.info(f"Backtest complete: P&L ${result.total_pnl:.2f} ({result.total_pnl_percent:.2f}%), "
                        f"{result.total_trades} trades, {result.win_rate:.1f}% win rate")
        
        return result
    
    def _calculate_range(self, price: float, range_percent: float) -> Tuple[float, float]:
        """Calculate grid range around price"""
        half_range = price * (range_percent / 100) / 2
        return price - half_range, price + half_range
    
    def _create_grid_levels(self, lower: float, upper: float, 
                           num_levels: int, spacing: str) -> List[float]:
        """Create grid levels"""
        levels = []
        
        if spacing == "geometric":
            ratio = (upper / lower) ** (1 / (num_levels - 1))
            for i in range(num_levels):
                levels.append(lower * (ratio ** i))
        else:  # arithmetic
            step = (upper - lower) / (num_levels - 1)
            for i in range(num_levels):
                levels.append(lower + step * i)
        
        return levels
    
    def compare_strategies(self, results: List[BacktestResult]) -> Dict:
        """Compare multiple backtest results"""
        if not results:
            return {}
        
        comparison = {
            "strategies": [],
            "best_pnl": None,
            "best_sharpe": None,
            "best_win_rate": None,
            "lowest_drawdown": None
        }
        
        best_pnl = float('-inf')
        best_sharpe = float('-inf')
        best_win_rate = 0
        lowest_dd = float('inf')
        
        for i, r in enumerate(results):
            strategy_info = {
                "index": i,
                "config_summary": f"{r.config.grid_levels}L, {r.config.range_percent}%R, {r.config.grid_spacing}",
                "total_pnl": round(r.total_pnl, 2),
                "total_pnl_percent": round(r.total_pnl_percent, 2),
                "win_rate": round(r.win_rate, 1),
                "sharpe_ratio": round(r.sharpe_ratio, 2),
                "max_drawdown": round(r.max_drawdown_percent, 2),
                "total_trades": r.total_trades
            }
            comparison["strategies"].append(strategy_info)
            
            if r.total_pnl > best_pnl:
                best_pnl = r.total_pnl
                comparison["best_pnl"] = i
            
            if r.sharpe_ratio > best_sharpe:
                best_sharpe = r.sharpe_ratio
                comparison["best_sharpe"] = i
            
            if r.win_rate > best_win_rate:
                best_win_rate = r.win_rate
                comparison["best_win_rate"] = i
            
            if r.max_drawdown_percent < lowest_dd:
                lowest_dd = r.max_drawdown_percent
                comparison["lowest_drawdown"] = i
        
        return comparison


class ParameterOptimizer:
    """
    Optimize grid parameters using backtesting
    """
    
    def __init__(self, backtester: GridBacktester, logger: logging.Logger):
        self.backtester = backtester
        self.logger = logger
    
    async def optimize(self, base_config: BacktestConfig,
                      price_data: List[Dict],
                      param_ranges: Dict[str, List]) -> Tuple[BacktestConfig, BacktestResult]:
        """
        Find optimal parameters by testing combinations
        
        param_ranges example:
        {
            "grid_levels": [10, 15, 20, 25],
            "range_percent": [8, 10, 12, 15],
            "min_profit_percent": [0.3, 0.4, 0.5]
        }
        """
        best_result = None
        best_config = None
        best_score = float('-inf')
        
        # Generate all combinations
        combinations = self._generate_combinations(param_ranges)
        total = len(combinations)
        
        self.logger.info(f"Testing {total} parameter combinations...")
        
        for i, params in enumerate(combinations):
            # Create config with these params
            config = BacktestConfig(
                start_date=base_config.start_date,
                end_date=base_config.end_date,
                pair=base_config.pair,
                initial_capital=base_config.initial_capital,
                leverage=base_config.leverage,
                grid_mode=base_config.grid_mode,
                grid_levels=params.get("grid_levels", base_config.grid_levels),
                range_percent=params.get("range_percent", base_config.range_percent),
                grid_spacing=params.get("grid_spacing", base_config.grid_spacing),
                min_profit_percent=params.get("min_profit_percent", base_config.min_profit_percent),
                enable_auto_adjust=base_config.enable_auto_adjust,
                rebuild_on_exit=base_config.rebuild_on_exit
            )
            
            # Run backtest
            result = await self.backtester.run_backtest(config, price_data)
            
            # Score (balance of return and risk)
            score = self._calculate_score(result)
            
            if score > best_score:
                best_score = score
                best_result = result
                best_config = config
            
            if (i + 1) % 10 == 0:
                self.logger.info(f"Progress: {i + 1}/{total}, best score: {best_score:.2f}")
        
        self.logger.info(f"Optimization complete. Best config: "
                        f"{best_config.grid_levels}L, {best_config.range_percent}%R, "
                        f"Score: {best_score:.2f}")
        
        return best_config, best_result
    
    def _generate_combinations(self, param_ranges: Dict[str, List]) -> List[Dict]:
        """Generate all parameter combinations"""
        if not param_ranges:
            return [{}]
        
        keys = list(param_ranges.keys())
        values = list(param_ranges.values())
        
        combinations = []
        
        def recurse(index, current):
            if index == len(keys):
                combinations.append(current.copy())
                return
            
            key = keys[index]
            for value in values[index]:
                current[key] = value
                recurse(index + 1, current)
        
        recurse(0, {})
        return combinations
    
    def _calculate_score(self, result: BacktestResult) -> float:
        """
        Calculate score for optimization
        Balances return, risk, and consistency
        """
        if result.total_trades == 0:
            return float('-inf')
        
        # Components
        return_score = result.total_pnl_percent
        risk_score = -result.max_drawdown_percent * 2  # Penalize drawdown
        consistency_score = result.win_rate / 10  # Bonus for high win rate
        sharpe_score = result.sharpe_ratio * 5 if result.sharpe_ratio > 0 else result.sharpe_ratio * 10
        
        # Activity penalty (too few trades = unreliable)
        if result.total_trades < 10:
            return float('-inf')
        
        return return_score + risk_score + consistency_score + sharpe_score


def format_backtest_report(result: BacktestResult) -> str:
    """Format backtest result as readable report"""
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║                    BACKTEST REPORT                           ║
╠══════════════════════════════════════════════════════════════╣
║ Pair: {result.config.pair:<15} Duration: {result.duration_days:.1f} days           ║
║ Capital: ${result.config.initial_capital:<10.2f} Leverage: {result.config.leverage}x                  ║
╠══════════════════════════════════════════════════════════════╣
║ PERFORMANCE                                                  ║
║ ─────────────────────────────────────────────────────────── ║
║ Total P&L:      ${result.total_pnl:>10.2f} ({result.total_pnl_percent:>+6.2f}%)                  ║
║ Total Trades:   {result.total_trades:>10}                                    ║
║ Win Rate:       {result.win_rate:>10.1f}%                                   ║
║ Avg Trade:      ${result.avg_trade_pnl:>10.2f}                                    ║
║ Best Trade:     ${result.best_trade:>10.2f}                                    ║
║ Worst Trade:    ${result.worst_trade:>10.2f}                                    ║
╠══════════════════════════════════════════════════════════════╣
║ RISK METRICS                                                 ║
║ ─────────────────────────────────────────────────────────── ║
║ Max Drawdown:   {result.max_drawdown_percent:>10.2f}% (${result.max_drawdown_usd:.2f})                ║
║ Sharpe Ratio:   {result.sharpe_ratio:>10.2f}                                    ║
║ Sortino Ratio:  {result.sortino_ratio:>10.2f}                                    ║
║ Calmar Ratio:   {result.calmar_ratio:>10.2f}                                    ║
║ Profit Factor:  {result.profit_factor:>10.2f}                                    ║
╠══════════════════════════════════════════════════════════════╣
║ COSTS                                                        ║
║ ─────────────────────────────────────────────────────────── ║
║ Total Fees:     ${result.total_fees:>10.2f}                                    ║
║ Total Funding:  ${result.total_funding:>10.2f}                                    ║
║ Grid Rebuilds:  {result.grid_rebuilds:>10}                                    ║
╠══════════════════════════════════════════════════════════════╣
║ GRID CONFIG                                                  ║
║ ─────────────────────────────────────────────────────────── ║
║ Levels: {result.config.grid_levels:<5} Range: {result.config.range_percent}%  Spacing: {result.config.grid_spacing:<12} ║
║ Min Profit: {result.config.min_profit_percent}%  Mode: {result.config.grid_mode:<10}                      ║
╚══════════════════════════════════════════════════════════════╝
"""
    return report
