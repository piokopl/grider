import os
import yaml
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class AutoAdjustConfig:
    enabled: bool
    rebuild_on_exit: bool
    exit_buffer_percent: float
    scheduled_check_hours: int
    rebuild_on_inefficiency: bool
    inefficiency_threshold: int
    min_grid_age_minutes: int
    volatility_lookback_hours: int
    volatility_multiplier: float
    min_range_percent: float
    max_range_percent: float
    close_position_on_rebuild: bool
    min_rebuild_interval_minutes: int


@dataclass
class AutoOptimizeConfig:
    enabled: bool
    optimize_interval_hours: int
    min_confidence: int
    auto_apply: bool
    optimize_mode: bool
    optimize_range: bool
    optimize_levels: bool
    optimize_min_profit: bool
    optimize_spacing: bool
    max_range_change_percent: float
    max_level_change: int
    reallocate_capital: bool
    reallocation_interval_hours: int


@dataclass
class AIConfigData:
    enabled: bool
    claude_api_key: str
    model: str
    coingecko_api_key: str
    cryptocompare_api_key: str
    news_enabled: bool
    news_lookback_hours: int
    news_update_interval_minutes: int
    regime_enabled: bool
    regime_update_interval_minutes: int
    auto_adjust_mode: bool
    auto_adjust_range: bool
    min_confidence: float


@dataclass
class NotificationConfigData:
    discord_enabled: bool
    discord_webhook_url: str
    telegram_enabled: bool
    telegram_bot_token: str
    telegram_chat_id: str
    notify_trades: bool
    notify_rebuilds: bool
    notify_optimizations: bool
    notify_errors: bool
    notify_daily_summary: bool
    notify_risk_alerts: bool
    min_interval_seconds: int
    batch_trades: bool
    batch_interval_seconds: int


@dataclass
class GlobalConfig:
    exchange: str
    paper_trading: bool
    api_key: str
    api_secret: str
    passphrase: str
    update_interval: int
    log_level: str
    web_port: int
    auto_adjust: AutoAdjustConfig
    auto_optimize: AutoOptimizeConfig
    ai: AIConfigData
    notifications: NotificationConfigData
    # Global trading defaults
    max_orders_per_side: int = 3  # Can be overridden per-pair
    notifications: NotificationConfigData


def load_global_config(config_path: str = "config.yaml") -> GlobalConfig:
    """Load global configuration from YAML file"""
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Parse auto_adjust section
    auto_adjust_data = data.get("auto_adjust", {})
    auto_adjust = AutoAdjustConfig(
        enabled=auto_adjust_data.get("enabled", True),
        rebuild_on_exit=auto_adjust_data.get("rebuild_on_exit", True),
        exit_buffer_percent=auto_adjust_data.get("exit_buffer_percent", 1.0),
        scheduled_check_hours=auto_adjust_data.get("scheduled_check_hours", 6),
        rebuild_on_inefficiency=auto_adjust_data.get("rebuild_on_inefficiency", True),
        inefficiency_threshold=auto_adjust_data.get("inefficiency_threshold", 70),
        min_grid_age_minutes=auto_adjust_data.get("min_grid_age_minutes", 60),
        volatility_lookback_hours=auto_adjust_data.get("volatility_lookback_hours", 24),
        volatility_multiplier=auto_adjust_data.get("volatility_multiplier", 2.0),
        min_range_percent=auto_adjust_data.get("min_range_percent", 5),
        max_range_percent=auto_adjust_data.get("max_range_percent", 30),
        close_position_on_rebuild=auto_adjust_data.get("close_position_on_rebuild", False),
        min_rebuild_interval_minutes=auto_adjust_data.get("min_rebuild_interval_minutes", 30)
    )
    
    # Parse auto_optimize section
    auto_optimize_data = data.get("auto_optimize", {})
    auto_optimize = AutoOptimizeConfig(
        enabled=auto_optimize_data.get("enabled", True),
        optimize_interval_hours=auto_optimize_data.get("optimize_interval_hours", 4),
        min_confidence=auto_optimize_data.get("min_confidence", 60),
        auto_apply=auto_optimize_data.get("auto_apply", True),
        optimize_mode=auto_optimize_data.get("optimize_mode", True),
        optimize_range=auto_optimize_data.get("optimize_range", True),
        optimize_levels=auto_optimize_data.get("optimize_levels", True),
        optimize_min_profit=auto_optimize_data.get("optimize_min_profit", True),
        optimize_spacing=auto_optimize_data.get("optimize_spacing", True),
        max_range_change_percent=auto_optimize_data.get("max_range_change_percent", 50),
        max_level_change=auto_optimize_data.get("max_level_change", 10),
        reallocate_capital=auto_optimize_data.get("reallocate_capital", False),
        reallocation_interval_hours=auto_optimize_data.get("reallocation_interval_hours", 24)
    )
    
    # Parse AI section
    ai_data = data.get("ai", {})
    ai_config = AIConfigData(
        enabled=ai_data.get("enabled", False),
        claude_api_key=ai_data.get("claude_api_key", ""),
        model=ai_data.get("model", "claude-sonnet-4-20250514"),
        coingecko_api_key=ai_data.get("coingecko_api_key", ""),
        cryptocompare_api_key=ai_data.get("cryptocompare_api_key", ""),
        news_enabled=ai_data.get("news_enabled", True),
        news_lookback_hours=ai_data.get("news_lookback_hours", 24),
        news_update_interval_minutes=ai_data.get("news_update_interval_minutes", 30),
        regime_enabled=ai_data.get("regime_enabled", True),
        regime_update_interval_minutes=ai_data.get("regime_update_interval_minutes", 15),
        auto_adjust_mode=ai_data.get("auto_adjust_mode", True),
        auto_adjust_range=ai_data.get("auto_adjust_range", True),
        min_confidence=ai_data.get("min_confidence", 0.7)
    )
    
    # Parse notifications section
    notif_data = data.get("notifications", {})
    notifications = NotificationConfigData(
        discord_enabled=notif_data.get("discord_enabled", False),
        discord_webhook_url=notif_data.get("discord_webhook_url", ""),
        telegram_enabled=notif_data.get("telegram_enabled", False),
        telegram_bot_token=notif_data.get("telegram_bot_token", ""),
        telegram_chat_id=notif_data.get("telegram_chat_id", ""),
        notify_trades=notif_data.get("notify_trades", True),
        notify_rebuilds=notif_data.get("notify_rebuilds", True),
        notify_optimizations=notif_data.get("notify_optimizations", True),
        notify_errors=notif_data.get("notify_errors", True),
        notify_daily_summary=notif_data.get("notify_daily_summary", True),
        notify_risk_alerts=notif_data.get("notify_risk_alerts", True),
        min_interval_seconds=notif_data.get("min_interval_seconds", 5),
        batch_trades=notif_data.get("batch_trades", True),
        batch_interval_seconds=notif_data.get("batch_interval_seconds", 60)
    )
    
    return GlobalConfig(
        exchange=data.get("exchange", "bitget"),
        paper_trading=data.get("paper_trading", True),
        api_key=data.get("api_key", ""),
        api_secret=data.get("api_secret", ""),
        passphrase=data.get("passphrase", ""),
        update_interval=data.get("update_interval", 5),
        log_level=data.get("log_level", "INFO"),
        web_port=data.get("web_port", 80),
        auto_adjust=auto_adjust,
        auto_optimize=auto_optimize,
        ai=ai_config,
        notifications=notifications,
        max_orders_per_side=data.get("max_orders_per_side", 3)
    )


def load_pair_config(config_path: str) -> 'GridConfig':
    """Load pair configuration from YAML file"""
    from grid_manager_v2 import GridConfig
    
    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)
    
    return GridConfig(
        pair=data.get("pair", ""),
        enabled=data.get("enabled", True),
        leverage=data.get("leverage", 5),
        margin_mode=data.get("margin_mode", "crossed"),
        
        grid_mode=data.get("grid_mode", "neutral"),
        total_capital=data.get("total_capital", 500),
        capital_per_grid=data.get("capital_per_grid", 0),
        
        upper_price=data.get("upper_price", 0),
        lower_price=data.get("lower_price", 0),
        range_percent=data.get("range_percent", 10),
        grid_levels=data.get("grid_levels", 20),
        grid_spacing=data.get("grid_spacing", "arithmetic"),
        
        order_type=data.get("order_type", "limit"),
        min_profit_percent=data.get("min_profit_percent", 0.3),
        
        sl_percent=data.get("sl_percent", 0),
        tp_percent=data.get("tp_percent", 0),
        
        trailing_stop=data.get("trailing_stop", False),
        trailing_stop_activation=data.get("trailing_stop_activation", 5.0),
        trailing_stop_callback=data.get("trailing_stop_callback", 2.0),
        
        trend_filter=data.get("trend_filter", False),
        trend_indicator=data.get("trend_indicator", "ema_cross"),
        trend_tf=data.get("trend_tf", "4H"),
        trend_params=data.get("trend_params", {"fast": 20, "slow": 50}),
        
        rebalance_enabled=data.get("rebalance_enabled", False),
        rebalance_threshold=data.get("rebalance_threshold", 20),
        
        check_interval=data.get("check_interval", 10),
        cooldown_after_fill=data.get("cooldown_after_fill", 0),
        max_open_orders=data.get("max_open_orders", 0),
        
        # V2 parameters
        tp_steps=data.get("tp_steps", 1),
        reanchor_threshold=data.get("reanchor_threshold", 0.03),
        max_position_levels=data.get("max_position_levels", 10),
        max_orders_per_side=data.get("max_orders_per_side", 3)
    )


def load_all_pair_configs(configs_dir: str = "configs") -> List['GridConfig']:
    """Load all pair configurations from directory"""
    configs = []
    configs_path = Path(configs_dir)
    
    if not configs_path.exists():
        return configs
    
    for config_file in configs_path.glob("*.yaml"):
        try:
            config = load_pair_config(str(config_file))
            if config.pair:
                configs.append(config)
        except Exception as e:
            print(f"Error loading {config_file}: {e}")
    
    return configs


def setup_logger(name: str, log_file: str = None, level: str = "INFO") -> logging.Logger:
    """Setup logger with file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_format = logging.Formatter(
        '%(asctime)s | %(name)-12s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def setup_pair_logger(pair: str, logs_dir: str, level: str = "INFO") -> logging.Logger:
    """Setup logger for a specific pair"""
    log_file = os.path.join(logs_dir, f"{pair}.log")
    return setup_logger(pair, log_file, level)


def format_price(price: float, decimals: int = 4) -> str:
    """Format price with appropriate decimals"""
    if price >= 1000:
        return f"{price:,.2f}"
    elif price >= 1:
        return f"{price:.{decimals}f}"
    else:
        return f"{price:.6f}"


def format_percent(value: float) -> str:
    """Format percentage value"""
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"


def format_pnl(value: float) -> str:
    """Format P&L value"""
    sign = "+" if value > 0 else ""
    return f"{sign}${value:.2f}"


def calculate_grid_levels(lower: float, upper: float, n: int, 
                          spacing: str = "arithmetic") -> List[float]:
    """Calculate grid level prices"""
    levels = []
    
    if spacing == "arithmetic":
        step = (upper - lower) / (n - 1) if n > 1 else 0
        for i in range(n):
            levels.append(lower + step * i)
    else:  # geometric
        ratio = (upper / lower) ** (1 / (n - 1)) if n > 1 else 1
        price = lower
        for i in range(n):
            levels.append(price)
            price *= ratio
    
    return levels


def calculate_atr(klines: List[Dict], period: int = 14) -> float:
    """Calculate Average True Range"""
    if len(klines) < period + 1:
        return 0
    
    tr_values = []
    for i in range(1, len(klines)):
        high = float(klines[i]["high"])
        low = float(klines[i]["low"])
        prev_close = float(klines[i-1]["close"])
        
        tr = max(
            high - low,
            abs(high - prev_close),
            abs(low - prev_close)
        )
        tr_values.append(tr)
    
    if len(tr_values) < period:
        return sum(tr_values) / len(tr_values) if tr_values else 0
    
    return sum(tr_values[-period:]) / period


def calculate_volatility(klines: List[Dict], period: int = 24) -> float:
    """Calculate price volatility as percentage"""
    if len(klines) < period:
        return 0
    
    recent_klines = klines[-period:]
    closes = [float(k["close"]) for k in recent_klines]
    
    if not closes:
        return 0
    
    avg = sum(closes) / len(closes)
    variance = sum((c - avg) ** 2 for c in closes) / len(closes)
    std = variance ** 0.5
    
    return (std / avg) * 100 if avg > 0 else 0


def suggest_grid_range(current_price: float, volatility: float, 
                       multiplier: float = 2.0) -> tuple:
    """Suggest grid range based on volatility"""
    range_percent = volatility * multiplier
    range_percent = max(5, min(30, range_percent))  # Clamp between 5% and 30%
    
    upper = current_price * (1 + range_percent / 100)
    lower = current_price * (1 - range_percent / 100)
    
    return lower, upper, range_percent


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to YAML file"""
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def merge_configs(base: Dict, updates: Dict) -> Dict:
    """Merge two configurations, with updates taking precedence"""
    result = base.copy()
    for key, value in updates.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result
