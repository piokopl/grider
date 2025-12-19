import sqlite3
import json
from datetime import datetime
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import threading

@dataclass
class Grid:
    """Represents a grid trading session"""
    id: Optional[int]
    pair: str
    status: str  # active / stopped / completed
    grid_mode: str  # neutral / long / short
    upper_price: float
    lower_price: float
    grid_levels: int
    grid_spacing: str  # arithmetic / geometric
    total_capital: float
    capital_per_grid: float
    
    # Position tracking
    total_bought: float       # Total amount bought
    total_sold: float         # Total amount sold
    net_position: float       # Current net position (bought - sold)
    average_buy_price: float  # Weighted average buy price
    average_sell_price: float # Weighted average sell price
    total_buy_cost: float     # Total USD spent on buys
    total_sell_revenue: float # Total USD received from sells
    
    # Performance
    realized_pnl: float       # Realized profit from completed grid trades
    unrealized_pnl: float     # Unrealized P&L from open position
    total_trades: int         # Total number of filled orders
    grid_profits: int         # Number of profitable grid cycles
    
    # Timestamps
    created_at: str
    updated_at: str
    stopped_at: Optional[str]
    
    # Stop Loss / Take Profit
    sl_price: Optional[float]
    tp_price: Optional[float]
    stop_reason: Optional[str]  # sl / tp / manual / trend


@dataclass 
class GridLevel:
    """Represents a single grid level"""
    id: Optional[int]
    grid_id: int
    pair: str
    level_index: int          # 0 = lowest, N-1 = highest
    price: float              # Target price for this level
    
    # Order tracking
    buy_order_id: Optional[str]   # Current buy order ID
    sell_order_id: Optional[str]  # Current sell order ID
    buy_status: str               # none / pending / filled
    sell_status: str              # none / pending / filled
    
    # Fill history
    last_buy_price: Optional[float]
    last_buy_amount: Optional[float]
    last_buy_time: Optional[str]
    last_sell_price: Optional[float]
    last_sell_amount: Optional[float]
    last_sell_time: Optional[str]
    
    # Stats for this level
    total_buys: int
    total_sells: int
    level_pnl: float  # Profit from this level


@dataclass
class GridOrder:
    """Order record for grid trades"""
    id: Optional[int]
    grid_id: int
    level_id: int
    pair: str
    order_id: str           # Exchange order ID
    order_type: str         # buy / sell
    side: str               # buy / sell (for futures direction)
    price: float
    amount: float
    status: str             # pending / filled / cancelled / failed
    created_at: str
    filled_at: Optional[str]
    fill_price: Optional[float]
    fill_amount: Optional[float]
    fee: Optional[float]


class Database:
    _local = threading.local()
    
    def __init__(self, db_path: str = "grid_bot.db"):
        self.db_path = db_path
        self._init_db()
    
    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def _init_db(self):
        conn = self._get_conn()
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS grids (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pair TEXT NOT NULL,
                status TEXT NOT NULL,
                grid_mode TEXT NOT NULL,
                upper_price REAL NOT NULL,
                lower_price REAL NOT NULL,
                grid_levels INTEGER NOT NULL,
                grid_spacing TEXT NOT NULL,
                total_capital REAL NOT NULL,
                capital_per_grid REAL NOT NULL,
                
                total_bought REAL DEFAULT 0,
                total_sold REAL DEFAULT 0,
                net_position REAL DEFAULT 0,
                average_buy_price REAL DEFAULT 0,
                average_sell_price REAL DEFAULT 0,
                total_buy_cost REAL DEFAULT 0,
                total_sell_revenue REAL DEFAULT 0,
                
                realized_pnl REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                total_trades INTEGER DEFAULT 0,
                grid_profits INTEGER DEFAULT 0,
                
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                stopped_at TEXT,
                
                sl_price REAL,
                tp_price REAL,
                stop_reason TEXT
            );
            
            CREATE TABLE IF NOT EXISTS grid_levels (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grid_id INTEGER NOT NULL,
                pair TEXT NOT NULL,
                level_index INTEGER NOT NULL,
                price REAL NOT NULL,
                
                buy_order_id TEXT,
                sell_order_id TEXT,
                buy_status TEXT DEFAULT 'none',
                sell_status TEXT DEFAULT 'none',
                
                last_buy_price REAL,
                last_buy_amount REAL,
                last_buy_time TEXT,
                last_sell_price REAL,
                last_sell_amount REAL,
                last_sell_time TEXT,
                
                total_buys INTEGER DEFAULT 0,
                total_sells INTEGER DEFAULT 0,
                level_pnl REAL DEFAULT 0,
                
                FOREIGN KEY (grid_id) REFERENCES grids(id)
            );
            
            CREATE TABLE IF NOT EXISTS grid_orders (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                grid_id INTEGER NOT NULL,
                level_id INTEGER NOT NULL,
                pair TEXT NOT NULL,
                order_id TEXT NOT NULL,
                order_type TEXT NOT NULL,
                side TEXT NOT NULL,
                price REAL NOT NULL,
                amount REAL NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                filled_at TEXT,
                fill_price REAL,
                fill_amount REAL,
                fee REAL,
                
                FOREIGN KEY (grid_id) REFERENCES grids(id),
                FOREIGN KEY (level_id) REFERENCES grid_levels(id)
            );
            
            CREATE TABLE IF NOT EXISTS bot_state (
                pair TEXT PRIMARY KEY,
                active_grid_id INTEGER,
                trend_direction TEXT,
                is_paused INTEGER DEFAULT 0,
                last_price REAL,
                last_update TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_grids_pair_status ON grids(pair, status);
            CREATE INDEX IF NOT EXISTS idx_levels_grid_id ON grid_levels(grid_id);
            CREATE INDEX IF NOT EXISTS idx_orders_grid_id ON grid_orders(grid_id);
            CREATE INDEX IF NOT EXISTS idx_orders_status ON grid_orders(status);
            CREATE INDEX IF NOT EXISTS idx_orders_order_id ON grid_orders(order_id);
            
            -- Exchange state cache (synced from Bitget periodically)
            CREATE TABLE IF NOT EXISTS exchange_balance (
                id INTEGER PRIMARY KEY DEFAULT 1,
                total REAL DEFAULT 0,
                available REAL DEFAULT 0,
                wallet_balance REAL DEFAULT 0,
                used_margin REAL DEFAULT 0,
                unrealized_pnl REAL DEFAULT 0,
                roi_percent REAL DEFAULT 0,
                bonus REAL DEFAULT 0,
                updated_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS exchange_positions (
                pair TEXT PRIMARY KEY,
                side TEXT,
                size REAL,
                entry_price REAL,
                unrealized_pnl REAL,
                leverage INTEGER,
                margin_mode TEXT,
                updated_at TEXT
            );
            
            CREATE TABLE IF NOT EXISTS exchange_fills (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trade_id TEXT UNIQUE,
                pair TEXT,
                side TEXT,
                price REAL,
                amount REAL,
                profit REAL,
                fee REAL,
                filled_at TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS exchange_orders_count (
                id INTEGER PRIMARY KEY DEFAULT 1,
                total_orders INTEGER DEFAULT 0,
                buy_orders INTEGER DEFAULT 0,
                sell_orders INTEGER DEFAULT 0,
                updated_at TEXT
            );
        ''')
        conn.commit()
    
    # === GRIDS ===
    
    def create_grid(self, grid: Grid) -> int:
        conn = self._get_conn()
        cursor = conn.execute('''
            INSERT INTO grids (pair, status, grid_mode, upper_price, lower_price,
                             grid_levels, grid_spacing, total_capital, capital_per_grid,
                             total_bought, total_sold, net_position,
                             average_buy_price, average_sell_price,
                             total_buy_cost, total_sell_revenue,
                             realized_pnl, unrealized_pnl, total_trades, grid_profits,
                             created_at, updated_at, stopped_at, sl_price, tp_price, stop_reason)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (grid.pair, grid.status, grid.grid_mode, grid.upper_price, grid.lower_price,
              grid.grid_levels, grid.grid_spacing, grid.total_capital, grid.capital_per_grid,
              grid.total_bought, grid.total_sold, grid.net_position,
              grid.average_buy_price, grid.average_sell_price,
              grid.total_buy_cost, grid.total_sell_revenue,
              grid.realized_pnl, grid.unrealized_pnl, grid.total_trades, grid.grid_profits,
              grid.created_at, grid.updated_at, grid.stopped_at, 
              grid.sl_price, grid.tp_price, grid.stop_reason))
        conn.commit()
        return cursor.lastrowid
    
    def update_grid(self, grid: Grid):
        conn = self._get_conn()
        conn.execute('''
            UPDATE grids SET status=?, grid_mode=?,
                           total_bought=?, total_sold=?, net_position=?,
                           average_buy_price=?, average_sell_price=?,
                           total_buy_cost=?, total_sell_revenue=?,
                           realized_pnl=?, unrealized_pnl=?, total_trades=?, grid_profits=?,
                           updated_at=?, stopped_at=?, sl_price=?, tp_price=?, stop_reason=?
            WHERE id=?
        ''', (grid.status, grid.grid_mode,
              grid.total_bought, grid.total_sold, grid.net_position,
              grid.average_buy_price, grid.average_sell_price,
              grid.total_buy_cost, grid.total_sell_revenue,
              grid.realized_pnl, grid.unrealized_pnl, grid.total_trades, grid.grid_profits,
              grid.updated_at, grid.stopped_at, grid.sl_price, grid.tp_price, grid.stop_reason,
              grid.id))
        conn.commit()
    
    def get_active_grid(self, pair: str) -> Optional[Grid]:
        conn = self._get_conn()
        row = conn.execute(
            'SELECT * FROM grids WHERE pair=? AND status=? ORDER BY id DESC LIMIT 1',
            (pair, 'active')
        ).fetchone()
        return self._row_to_grid(row) if row else None
    
    def get_grid_by_id(self, grid_id: int) -> Optional[Grid]:
        conn = self._get_conn()
        row = conn.execute('SELECT * FROM grids WHERE id=?', (grid_id,)).fetchone()
        return self._row_to_grid(row) if row else None
    
    def get_grids_history(self, pair: str = None, limit: int = 100) -> List[Grid]:
        conn = self._get_conn()
        if pair:
            rows = conn.execute(
                'SELECT * FROM grids WHERE pair=? ORDER BY created_at DESC LIMIT ?',
                (pair, limit)
            ).fetchall()
        else:
            rows = conn.execute(
                'SELECT * FROM grids ORDER BY created_at DESC LIMIT ?', (limit,)
            ).fetchall()
        return [self._row_to_grid(row) for row in rows]
    
    def _row_to_grid(self, row: sqlite3.Row) -> Grid:
        return Grid(
            id=row['id'], pair=row['pair'], status=row['status'],
            grid_mode=row['grid_mode'], upper_price=row['upper_price'],
            lower_price=row['lower_price'], grid_levels=row['grid_levels'],
            grid_spacing=row['grid_spacing'], total_capital=row['total_capital'],
            capital_per_grid=row['capital_per_grid'],
            total_bought=row['total_bought'], total_sold=row['total_sold'],
            net_position=row['net_position'], average_buy_price=row['average_buy_price'],
            average_sell_price=row['average_sell_price'],
            total_buy_cost=row['total_buy_cost'], total_sell_revenue=row['total_sell_revenue'],
            realized_pnl=row['realized_pnl'], unrealized_pnl=row['unrealized_pnl'],
            total_trades=row['total_trades'], grid_profits=row['grid_profits'],
            created_at=row['created_at'], updated_at=row['updated_at'],
            stopped_at=row['stopped_at'], sl_price=row['sl_price'],
            tp_price=row['tp_price'], stop_reason=row['stop_reason']
        )
    
    # === GRID LEVELS ===
    
    def create_grid_level(self, level: GridLevel) -> int:
        conn = self._get_conn()
        cursor = conn.execute('''
            INSERT INTO grid_levels (grid_id, pair, level_index, price,
                                   buy_order_id, sell_order_id, buy_status, sell_status,
                                   last_buy_price, last_buy_amount, last_buy_time,
                                   last_sell_price, last_sell_amount, last_sell_time,
                                   total_buys, total_sells, level_pnl)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (level.grid_id, level.pair, level.level_index, level.price,
              level.buy_order_id, level.sell_order_id, level.buy_status, level.sell_status,
              level.last_buy_price, level.last_buy_amount, level.last_buy_time,
              level.last_sell_price, level.last_sell_amount, level.last_sell_time,
              level.total_buys, level.total_sells, level.level_pnl))
        conn.commit()
        return cursor.lastrowid
    
    def update_grid_level(self, level: GridLevel):
        conn = self._get_conn()
        conn.execute('''
            UPDATE grid_levels SET buy_order_id=?, sell_order_id=?,
                                 buy_status=?, sell_status=?,
                                 last_buy_price=?, last_buy_amount=?, last_buy_time=?,
                                 last_sell_price=?, last_sell_amount=?, last_sell_time=?,
                                 total_buys=?, total_sells=?, level_pnl=?
            WHERE id=?
        ''', (level.buy_order_id, level.sell_order_id,
              level.buy_status, level.sell_status,
              level.last_buy_price, level.last_buy_amount, level.last_buy_time,
              level.last_sell_price, level.last_sell_amount, level.last_sell_time,
              level.total_buys, level.total_sells, level.level_pnl, level.id))
        conn.commit()
    
    def get_grid_levels(self, grid_id: int) -> List[GridLevel]:
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT * FROM grid_levels WHERE grid_id=? ORDER BY level_index',
            (grid_id,)
        ).fetchall()
        return [self._row_to_level(row) for row in rows]
    
    def get_level_by_id(self, level_id: int) -> Optional[GridLevel]:
        conn = self._get_conn()
        row = conn.execute('SELECT * FROM grid_levels WHERE id=?', (level_id,)).fetchone()
        return self._row_to_level(row) if row else None
    
    def _row_to_level(self, row: sqlite3.Row) -> GridLevel:
        return GridLevel(
            id=row['id'], grid_id=row['grid_id'], pair=row['pair'],
            level_index=row['level_index'], price=row['price'],
            buy_order_id=row['buy_order_id'], sell_order_id=row['sell_order_id'],
            buy_status=row['buy_status'], sell_status=row['sell_status'],
            last_buy_price=row['last_buy_price'], last_buy_amount=row['last_buy_amount'],
            last_buy_time=row['last_buy_time'], last_sell_price=row['last_sell_price'],
            last_sell_amount=row['last_sell_amount'], last_sell_time=row['last_sell_time'],
            total_buys=row['total_buys'], total_sells=row['total_sells'],
            level_pnl=row['level_pnl']
        )
    
    # === GRID ORDERS ===
    
    def create_grid_order(self, order: GridOrder) -> int:
        conn = self._get_conn()
        cursor = conn.execute('''
            INSERT INTO grid_orders (grid_id, level_id, pair, order_id, order_type, side,
                                   price, amount, status, created_at, filled_at,
                                   fill_price, fill_amount, fee)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (order.grid_id, order.level_id, order.pair, order.order_id,
              order.order_type, order.side, order.price, order.amount,
              order.status, order.created_at, order.filled_at,
              order.fill_price, order.fill_amount, order.fee))
        conn.commit()
        return cursor.lastrowid
    
    def update_grid_order(self, order: GridOrder):
        conn = self._get_conn()
        conn.execute('''
            UPDATE grid_orders SET status=?, filled_at=?, fill_price=?, fill_amount=?, fee=?
            WHERE id=?
        ''', (order.status, order.filled_at, order.fill_price, order.fill_amount, 
              order.fee, order.id))
        conn.commit()
    
    def update_order_status_by_order_id(self, order_id: str, status: str, 
                                        filled_at: str = None, fill_price: float = None,
                                        fill_amount: float = None, fee: float = None):
        conn = self._get_conn()
        conn.execute('''
            UPDATE grid_orders SET status=?, filled_at=?, fill_price=?, fill_amount=?, fee=?
            WHERE order_id=?
        ''', (status, filled_at, fill_price, fill_amount, fee, order_id))
        conn.commit()
    
    def get_pending_orders(self, grid_id: int) -> List[GridOrder]:
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT * FROM grid_orders WHERE grid_id=? AND status=?',
            (grid_id, 'pending')
        ).fetchall()
        return [self._row_to_order(row) for row in rows]
    
    def get_orders_by_grid(self, grid_id: int, limit: int = 500) -> List[GridOrder]:
        conn = self._get_conn()
        rows = conn.execute(
            'SELECT * FROM grid_orders WHERE grid_id=? ORDER BY created_at DESC LIMIT ?',
            (grid_id, limit)
        ).fetchall()
        return [self._row_to_order(row) for row in rows]
    
    def get_order_by_order_id(self, order_id: str) -> Optional[GridOrder]:
        conn = self._get_conn()
        row = conn.execute(
            'SELECT * FROM grid_orders WHERE order_id=?', (order_id,)
        ).fetchone()
        return self._row_to_order(row) if row else None
    
    def _row_to_order(self, row: sqlite3.Row) -> GridOrder:
        return GridOrder(
            id=row['id'], grid_id=row['grid_id'], level_id=row['level_id'],
            pair=row['pair'], order_id=row['order_id'], order_type=row['order_type'],
            side=row['side'], price=row['price'], amount=row['amount'],
            status=row['status'], created_at=row['created_at'],
            filled_at=row['filled_at'], fill_price=row['fill_price'],
            fill_amount=row['fill_amount'], fee=row['fee']
        )
    
    # === BOT STATE ===
    
    def get_bot_state(self, pair: str) -> Dict[str, Any]:
        conn = self._get_conn()
        row = conn.execute('SELECT * FROM bot_state WHERE pair=?', (pair,)).fetchone()
        if row:
            return dict(row)
        return {'pair': pair, 'active_grid_id': None, 'trend_direction': None, 
                'is_paused': 0, 'last_price': None, 'last_update': None}
    
    def update_bot_state(self, pair: str, **kwargs):
        conn = self._get_conn()
        state = self.get_bot_state(pair)
        state.update(kwargs)
        conn.execute('''
            INSERT OR REPLACE INTO bot_state (pair, active_grid_id, trend_direction, 
                                             is_paused, last_price, last_update)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (pair, state.get('active_grid_id'), state.get('trend_direction'),
              state.get('is_paused', 0), state.get('last_price'), state.get('last_update')))
        conn.commit()
    
    # === STATISTICS ===
    
    def get_grid_stats(self, grid_id: int) -> Dict[str, Any]:
        """Get detailed statistics for a grid"""
        conn = self._get_conn()
        
        # Order stats
        orders = conn.execute('''
            SELECT 
                COUNT(*) as total_orders,
                SUM(CASE WHEN status='filled' THEN 1 ELSE 0 END) as filled_orders,
                SUM(CASE WHEN order_type='buy' AND status='filled' THEN 1 ELSE 0 END) as buy_fills,
                SUM(CASE WHEN order_type='sell' AND status='filled' THEN 1 ELSE 0 END) as sell_fills,
                SUM(CASE WHEN status='filled' THEN fee ELSE 0 END) as total_fees
            FROM grid_orders WHERE grid_id=?
        ''', (grid_id,)).fetchone()
        
        # Level stats
        levels = conn.execute('''
            SELECT 
                COUNT(*) as total_levels,
                SUM(total_buys) as total_level_buys,
                SUM(total_sells) as total_level_sells,
                SUM(level_pnl) as total_level_pnl,
                AVG(level_pnl) as avg_level_pnl
            FROM grid_levels WHERE grid_id=?
        ''', (grid_id,)).fetchone()
        
        return {
            'total_orders': orders['total_orders'] or 0,
            'filled_orders': orders['filled_orders'] or 0,
            'buy_fills': orders['buy_fills'] or 0,
            'sell_fills': orders['sell_fills'] or 0,
            'total_fees': orders['total_fees'] or 0,
            'total_levels': levels['total_levels'] or 0,
            'total_level_buys': levels['total_level_buys'] or 0,
            'total_level_sells': levels['total_level_sells'] or 0,
            'total_level_pnl': levels['total_level_pnl'] or 0,
            'avg_level_pnl': levels['avg_level_pnl'] or 0
        }
    
    def get_all_time_stats(self, pair: str = None) -> Dict[str, Any]:
        """Get all-time statistics"""
        conn = self._get_conn()
        
        where_clause = "WHERE pair=?" if pair else ""
        params = (pair,) if pair else ()
        
        stats = conn.execute(f'''
            SELECT 
                COUNT(*) as total_grids,
                SUM(CASE WHEN status='active' THEN 1 ELSE 0 END) as active_grids,
                SUM(realized_pnl) as total_realized_pnl,
                SUM(total_trades) as total_trades,
                SUM(grid_profits) as total_grid_profits,
                AVG(realized_pnl) as avg_grid_pnl
            FROM grids {where_clause}
        ''', params).fetchone()
        
        return {
            'total_grids': stats['total_grids'] or 0,
            'active_grids': stats['active_grids'] or 0,
            'total_realized_pnl': stats['total_realized_pnl'] or 0,
            'total_trades': stats['total_trades'] or 0,
            'total_grid_profits': stats['total_grid_profits'] or 0,
            'avg_grid_pnl': stats['avg_grid_pnl'] or 0
        }
    
    def get_recent_filled_orders(self, limit: int = 20) -> List[Dict]:
        """Get recent filled orders across all grids"""
        conn = self._get_conn()
        
        rows = conn.execute('''
            SELECT pair, side, fill_price, fill_amount, filled_at, fee
            FROM grid_orders 
            WHERE status = 'filled' AND filled_at IS NOT NULL
            ORDER BY filled_at DESC 
            LIMIT ?
        ''', (limit,)).fetchall()
        
        return [dict(row) for row in rows]
    
    # === EXCHANGE STATE CACHE ===
    
    def save_exchange_balance(self, balance: dict):
        """Save exchange balance to cache"""
        conn = self._get_conn()
        conn.execute('''
            INSERT OR REPLACE INTO exchange_balance 
            (id, total, available, wallet_balance, used_margin, unrealized_pnl, roi_percent, bonus, updated_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        ''', (
            balance.get('total', 0),
            balance.get('available', 0),
            balance.get('wallet_balance', 0),
            balance.get('used_margin', 0),
            balance.get('unrealized_pnl', 0),
            balance.get('roi_percent', 0),
            balance.get('bonus', 0)
        ))
        conn.commit()
    
    def get_exchange_balance(self) -> dict:
        """Get cached exchange balance"""
        conn = self._get_conn()
        row = conn.execute('SELECT * FROM exchange_balance WHERE id = 1').fetchone()
        if row:
            return dict(row)
        return {'total': 0, 'available': 0, 'wallet_balance': 0, 'used_margin': 0, 
                'unrealized_pnl': 0, 'roi_percent': 0, 'bonus': 0}
    
    def save_exchange_positions(self, positions: list):
        """Save exchange positions to cache"""
        conn = self._get_conn()
        
        # Clear old positions
        conn.execute('DELETE FROM exchange_positions')
        
        # Insert new positions
        for pos in positions:
            conn.execute('''
                INSERT INTO exchange_positions 
                (pair, side, size, entry_price, unrealized_pnl, leverage, margin_mode, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))
            ''', (
                pos.pair if hasattr(pos, 'pair') else pos.get('pair'),
                pos.side if hasattr(pos, 'side') else pos.get('side'),
                pos.size if hasattr(pos, 'size') else pos.get('size'),
                pos.entry_price if hasattr(pos, 'entry_price') else pos.get('entry_price'),
                pos.unrealized_pnl if hasattr(pos, 'unrealized_pnl') else pos.get('unrealized_pnl'),
                pos.leverage if hasattr(pos, 'leverage') else pos.get('leverage'),
                pos.margin_mode if hasattr(pos, 'margin_mode') else pos.get('margin_mode')
            ))
        conn.commit()
    
    def get_exchange_positions(self) -> list:
        """Get cached exchange positions"""
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT pair, side, size, entry_price, unrealized_pnl, leverage, margin_mode, updated_at
            FROM exchange_positions
            ORDER BY pair
        ''').fetchall()
        return [dict(row) for row in rows]
    
    def save_exchange_fills(self, fills: list):
        """Save exchange fills to cache (with deduplication by trade_id)"""
        conn = self._get_conn()
        
        for fill in fills:
            trade_id = fill.get('trade_id') or fill.get('tradeId') or f"{fill.get('pair')}_{fill.get('filled_at')}"
            try:
                conn.execute('''
                    INSERT OR IGNORE INTO exchange_fills 
                    (trade_id, pair, side, price, amount, profit, fee, filled_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    trade_id,
                    fill.get('pair', ''),
                    fill.get('side', ''),
                    fill.get('fill_price') or fill.get('price', 0),
                    fill.get('fill_amount') or fill.get('amount', 0),
                    fill.get('profit', 0),
                    fill.get('fee', 0),
                    fill.get('filled_at', '')  # ISO format from exchange.py
                ))
            except:
                pass
        conn.commit()
    
    def get_exchange_fills(self, limit: int = 20) -> list:
        """Get cached exchange fills with display_time for dashboard"""
        conn = self._get_conn()
        rows = conn.execute('''
            SELECT pair, side, price as fill_price, amount as fill_amount, profit, fee, filled_at
            FROM exchange_fills
            ORDER BY id DESC
            LIMIT ?
        ''', (limit,)).fetchall()
        
        result = []
        for row in rows:
            fill = dict(row)
            # Extract HH:MM for dashboard display
            filled_at = fill.get('filled_at', '')
            if filled_at and 'T' in filled_at:
                # ISO format: "2025-01-15T11:28:00+00:00" -> "11:28"
                try:
                    time_part = filled_at.split('T')[1][:5]  # Get HH:MM
                    fill['display_time'] = time_part
                except:
                    fill['display_time'] = filled_at[:5] if len(filled_at) >= 5 else filled_at
            else:
                fill['display_time'] = filled_at
            result.append(fill)
        
        return result
    
    def get_today_fills(self) -> list:
        """Get only today's fills for Today P&L calculation"""
        conn = self._get_conn()
        # Filter by date part of ISO timestamp
        today = datetime.now().strftime('%Y-%m-%d')
        rows = conn.execute('''
            SELECT pair, side, price as fill_price, amount as fill_amount, profit, fee, filled_at
            FROM exchange_fills
            WHERE filled_at LIKE ?
            ORDER BY id DESC
        ''', (f'{today}%',)).fetchall()
        return [dict(row) for row in rows]
    
    def save_exchange_orders_count(self, total: int, buys: int, sells: int):
        """Save exchange orders count to cache"""
        conn = self._get_conn()
        conn.execute('''
            INSERT OR REPLACE INTO exchange_orders_count 
            (id, total_orders, buy_orders, sell_orders, updated_at)
            VALUES (1, ?, ?, ?, datetime('now'))
        ''', (total, buys, sells))
        conn.commit()
    
    def get_exchange_orders_count(self) -> dict:
        """Get cached exchange orders count"""
        conn = self._get_conn()
        row = conn.execute('SELECT * FROM exchange_orders_count WHERE id = 1').fetchone()
        if row:
            return dict(row)
        return {'total_orders': 0, 'buy_orders': 0, 'sell_orders': 0}
    
    def get_exchange_cache_age(self) -> int:
        """Get age of exchange cache in seconds"""
        conn = self._get_conn()
        row = conn.execute('''
            SELECT (julianday('now') - julianday(updated_at)) * 86400 as age_seconds
            FROM exchange_balance WHERE id = 1
        ''').fetchone()
        if row and row['age_seconds']:
            return int(row['age_seconds'])
        return 9999  # Very old if no data
