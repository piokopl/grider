# 🤖 Grider Trading Bot V2

Bitget AI Grid Crypto Bot

You must pay for using this bot:
- Anthropic api access key (about 50$ - 100$ monthly) or change code to your prefered AI agent
- Coingecko api key (Basic plan 29$/m for up to 60 currency pair)

<img width="976" height="978" alt="obraz" src="https://github.com/user-attachments/assets/21564061-75d1-477a-a961-9dad753c85bf" />

## ✨ Features

- **AI-Powered Analysis** - Automatic market regime detection (trend/ranging) and sentiment analysis
- **Dynamic Grid Modes** - LONG/SHORT/NEUTRAL modes selected by AI
- **Smart Order Placement** - Only places orders near current price (saves margin)
- **Auto-Rebuild** - Automatically rebuilds grid when price exits range
- **Web Dashboard** - Real-time monitoring on port 80
- **Telegram / Discord Notifications** - Trade alerts and daily summaries
- **Multi-Pair Support** - Trade 20+ pairs simultaneously

## 📊 How It Works

```
┌─────────────────────────────────────────────────────────┐
│  1. AI analyzes market (sentiment + regime)             │
│                      ↓                                  │
│  2. Creates grid with 15 price levels                   │
│                      ↓                                  │
│  3. Places 1-3 orders per side (dynamic)                │
│                      ↓                                  │
│  4. Every 10s checks:                                   │
│     • Order filled? → Place TP order                    │
│     • Price exited grid? → Rebuild                      │
│     • Efficiency < 70%? → Rebuild                       │
│                      ↓                                  │
│  5. Repeat                                              │
└─────────────────────────────────────────────────────────┘
```

### Grid Modes

| Mode | When | Strategy |
|------|------|----------|
| **LONG** | Uptrend | BUY below price → SELL TP above |
| **SHORT** | Downtrend | SELL above price → BUY TP below |
| **NEUTRAL** | Ranging | Both BUY and SELL around price |

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/yourusername/grid-bot.git
cd grid-bot
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure

```bash
# Copy example configs
cp config.yaml.example config.yaml
cp configs/BTCUSDT.yaml.example configs/BTCUSDT.yaml

# Edit with your API keys
nano config.yaml
```

### 4. Run

```bash
python3 main.py
```

## ⚙️ Configuration

### Main Config (`config.yaml`)

```yaml
# API credentials (from Bitget)
api_key: "YOUR_API_KEY"
api_secret: "YOUR_API_SECRET"
passphrase: "YOUR_PASSPHRASE"

# Trading settings
max_orders_per_side: 1  # Orders per side per pair

# AI settings
ai:
  enabled: true
  coingecko_api_key: "YOUR_KEY"  # Free from coingecko.com
```

### Pair Config (`configs/BTCUSDT.yaml`)

```yaml
pair: BTCUSDT
enabled: true
total_capital: 50      # USD per pair
leverage: 5            # 5x leverage
grid_levels: 15        # Price levels
min_profit_percent: 0.3  # Min profit per trade
```

## 💰 Capital Requirements

| Pairs | max_orders | Recommended Margin |
|-------|------------|-------------------|
| 10 | 1 | $150-200 |
| 10 | 3 | $400-500 |
| 25 | 1 | $300-400 |
| 25 | 3 | $800-1000 |

## 📁 Project Structure

```
grid-bot/
├── main.py              # Main entry point
├── grid_manager_v2.py   # Grid logic (V2 with AI)
├── exchange.py          # Bitget API wrapper
├── database.py          # SQLite storage
├── ai_analysis.py       # AI market analysis
├── webserver.py         # Dashboard server
├── notifications.py     # Discord/Telegram alerts
├── config.yaml          # Main configuration
└── configs/             # Per-pair configurations
    ├── BTCUSDT.yaml
    ├── ETHUSDT.yaml
    └── ...
```

## 🛠️ Utilities

### Cancel All Orders
```bash
python3 cancel_all_orders.py
```

### Fresh Start
```bash
pkill -f "python3 main.py"
python3 cancel_all_orders.py
rm grid_bot.db
python3 main.py
```

## 📈 Dashboard

Access at `http://your-server:80`

Features:
- Real-time P&L tracking
- Grid visualization
- Trade history
- AI recommendations

## ⚠️ Risk Warning

**This bot trades with real money. Use at your own risk!**

- Start with paper trading (`paper_trading: true`)
- Use small capital first
- Monitor regularly
- Never invest more than you can afford to lose

## 🤝 Contributing

Pull requests welcome! Please test thoroughly before submitting.

---

Made with ❤️ for crypto traders
