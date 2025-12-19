import hmac
import hashlib
import base64
import time
import json
import asyncio
import aiohttp
import os
from datetime import datetime
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
import logging

@dataclass
class Position:
    pair: str
    side: str  # long / short
    size: float
    entry_price: float
    unrealized_pnl: float
    leverage: int
    margin_mode: str

@dataclass
class OrderResult:
    order_id: str
    pair: str
    side: str
    price: float
    amount: float
    status: str
    filled_amount: float
    avg_fill_price: float

class BitgetExchange:
    BASE_URL = "https://api.bitget.com"
    PAPER_STATE_FILE = "paper_state.json"
    
    def __init__(self, api_key: str, api_secret: str, passphrase: str, 
                 paper_trading: bool = False, logger: logging.Logger = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.paper_trading = paper_trading
        self.logger = logger or logging.getLogger(__name__)
        
        # Paper trading state
        self._paper_positions: Dict[str, Position] = {}
        self._paper_orders: Dict[str, OrderResult] = {}
        self._paper_balance: float = 10000.0  # Starting balance for paper
        self._order_counter = 0
        
        # Instrument info cache (tick size, min size, etc.)
        self._instrument_cache: Dict[str, Dict] = {}
        
        # Rate limiting - max 3 concurrent requests, 250ms between each
        self._semaphore = asyncio.Semaphore(3)
        self._last_request_time = 0
        self._min_request_interval = 0.25  # 250ms between requests
        
        # Load paper state from file if exists
        if self.paper_trading:
            self._load_paper_state()
    
    def _load_paper_state(self):
        """Load paper trading state from file"""
        try:
            if os.path.exists(self.PAPER_STATE_FILE):
                with open(self.PAPER_STATE_FILE, 'r') as f:
                    data = json.load(f)
                
                self._paper_balance = data.get("balance", 10000.0)
                self._order_counter = data.get("order_counter", 0)
                
                # Restore positions
                for pair, pos_data in data.get("positions", {}).items():
                    self._paper_positions[pair] = Position(
                        pair=pos_data["pair"],
                        side=pos_data["side"],
                        size=pos_data["size"],
                        entry_price=pos_data["entry_price"],
                        unrealized_pnl=pos_data.get("unrealized_pnl", 0),
                        leverage=pos_data.get("leverage", 5),
                        margin_mode=pos_data.get("margin_mode", "crossed")
                    )
                
                # Restore pending orders
                for oid, order_data in data.get("orders", {}).items():
                    if order_data.get("status") == "pending":
                        self._paper_orders[oid] = OrderResult(
                            order_id=order_data["order_id"],
                            pair=order_data["pair"],
                            side=order_data["side"],
                            price=order_data["price"],
                            amount=order_data["amount"],
                            status=order_data["status"],
                            filled_amount=order_data.get("filled_amount", 0),
                            avg_fill_price=order_data.get("avg_fill_price", 0)
                        )
                
                self.logger.info(f"[PAPER] Loaded state: {len(self._paper_positions)} positions, "
                               f"{len(self._paper_orders)} pending orders, balance: ${self._paper_balance:.2f}")
        except Exception as e:
            self.logger.warning(f"[PAPER] Could not load state: {e}")
    
    def _save_paper_state(self):
        """Save paper trading state to file"""
        try:
            data = {
                "balance": self._paper_balance,
                "order_counter": self._order_counter,
                "positions": {},
                "orders": {}
            }
            
            for pair, pos in self._paper_positions.items():
                data["positions"][pair] = {
                    "pair": pos.pair,
                    "side": pos.side,
                    "size": pos.size,
                    "entry_price": pos.entry_price,
                    "unrealized_pnl": pos.unrealized_pnl,
                    "leverage": pos.leverage,
                    "margin_mode": pos.margin_mode
                }
            
            for oid, order in self._paper_orders.items():
                data["orders"][oid] = {
                    "order_id": order.order_id,
                    "pair": order.pair,
                    "side": order.side,
                    "price": order.price,
                    "amount": order.amount,
                    "status": order.status,
                    "filled_amount": order.filled_amount,
                    "avg_fill_price": order.avg_fill_price
                }
            
            with open(self.PAPER_STATE_FILE, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"[PAPER] Could not save state: {e}")
    
    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            self.api_secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256
        )
        return base64.b64encode(mac.digest()).decode('utf-8')
    
    def _headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        return {
            "ACCESS-KEY": self.api_key,
            "ACCESS-SIGN": self._sign(timestamp, method, path, body),
            "ACCESS-TIMESTAMP": timestamp,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US"
        }
    
    async def _request(self, method: str, path: str, params: Dict = None, 
                       body: Dict = None, retries: int = 3) -> Dict:
        # Rate limiting - wait for semaphore and minimum interval
        async with self._semaphore:
            # Ensure minimum interval between requests
            now = time.time()
            elapsed = now - self._last_request_time
            if elapsed < self._min_request_interval:
                await asyncio.sleep(self._min_request_interval - elapsed)
            self._last_request_time = time.time()
            
            url = self.BASE_URL + path
            body_str = json.dumps(body) if body else ""
            
            if params:
                query = "&".join(f"{k}={v}" for k, v in params.items())
                path = f"{path}?{query}"
                url = f"{url}?{query}"
            
            headers = self._headers(method, path, body_str)
            
            last_error = None
            for attempt in range(retries):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.request(method, url, headers=headers, 
                                                  data=body_str if body else None,
                                                  timeout=aiohttp.ClientTimeout(total=10)) as resp:
                            # Check for server errors
                            if resp.status >= 500:
                                last_error = f"Server error {resp.status}"
                                self.logger.warning(f"API server error {resp.status}, attempt {attempt + 1}/{retries}")
                                await asyncio.sleep(2 ** attempt)  # Exponential backoff
                                continue
                            
                            # Check content type
                            content_type = resp.headers.get('Content-Type', '')
                            if 'application/json' not in content_type:
                                last_error = f"Unexpected content type: {content_type}"
                                self.logger.warning(f"API returned non-JSON ({resp.status}), attempt {attempt + 1}/{retries}")
                                await asyncio.sleep(2 ** attempt)
                                continue
                            
                            data = await resp.json()
                            
                            if data.get("code") != "00000":
                                error_msg = data.get('msg', 'Unknown error')
                                error_code = data.get("code", "")
                                
                                # Rate limit - wait and retry
                                if 'rate' in error_msg.lower() or error_code in ["40014", "429"]:
                                    wait_time = 3 * (attempt + 1)  # Progressive wait
                                    self.logger.warning(f"Rate limited, waiting {wait_time}s...")
                                    await asyncio.sleep(wait_time)
                                    continue
                                
                                # Order not found (40109) - this is normal, don't log as ERROR
                                if error_code == "40109":
                                    # Don't spam logs - this happens when checking old orders
                                    raise Exception(f"Bitget API error: {error_msg}")
                                
                                # Minimum amount error (45110) - don't log as ERROR
                                if error_code == "45110":
                                    # This happens when order value is below 5 USDT - handled by caller
                                    raise Exception(f"Bitget API error: {error_msg}")
                                
                                # Order not found (40034) - this is normal when checking cancelled orders
                                if error_code == "40034":
                                    self.logger.debug(f"Order not found: {data}")
                                    raise Exception(f"Bitget API error: {error_msg}")
                                
                                self.logger.error(f"API Error: {data}")
                                raise Exception(f"Bitget API error: {error_msg}")
                            
                            return data.get("data", {})
                            
                except aiohttp.ClientError as e:
                    last_error = str(e)
                    self.logger.warning(f"Network error: {e}, attempt {attempt + 1}/{retries}")
                    await asyncio.sleep(2 ** attempt)
                except asyncio.TimeoutError:
                    last_error = "Request timeout"
                    self.logger.warning(f"Timeout, attempt {attempt + 1}/{retries}")
                    await asyncio.sleep(2 ** attempt)
            
            raise Exception(f"API request failed after {retries} attempts: {last_error}")
    
    # === MARKET DATA ===
    
    async def get_ticker(self, pair: str) -> Dict[str, float]:
        """Get current price for a pair"""
        path = "/api/v2/mix/market/ticker"
        params = {"symbol": pair, "productType": "USDT-FUTURES"}
        data = await self._request("GET", path, params=params)
        if isinstance(data, list) and len(data) > 0:
            data = data[0]
        return {
            "last": float(data.get("lastPr", 0)),
            "bid": float(data.get("bidPr", 0)),
            "ask": float(data.get("askPr", 0)),
            "volume": float(data.get("baseVolume", 0))
        }
    
    async def get_klines(self, pair: str, interval: str, limit: int = 100) -> List[Dict]:
        """Get candlestick data
        interval: 1m, 5m, 15m, 30m, 1H, 4H, 1D, 1W
        """
        path = "/api/v2/mix/market/candles"
        params = {
            "symbol": pair,
            "productType": "USDT-FUTURES",
            "granularity": interval,
            "limit": str(limit)
        }
        data = await self._request("GET", path, params=params)
        
        klines = []
        
        # Handle None or non-list response
        if not data or not isinstance(data, list):
            return klines
        
        for k in data:
            klines.append({
                "timestamp": int(k[0]),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5])
            })
        return klines
    
    async def get_instrument_info(self, pair: str) -> Dict:
        """Get instrument info including tick size and min size"""
        # Check cache first
        if pair in self._instrument_cache:
            return self._instrument_cache[pair]
        
        try:
            path = "/api/v2/mix/market/contracts"
            params = {"productType": "USDT-FUTURES", "symbol": pair}
            data = await self._request("GET", path, params=params)
            
            if isinstance(data, list) and len(data) > 0:
                info = data[0]
            else:
                info = data
            
            # pricePlace = number of decimal places
            # priceEndStep = tick step (e.g., 1, 5, 10 means last digit must be 0, 5, 0)
            price_decimals = int(info.get("pricePlace", 2))
            price_end_step = int(info.get("priceEndStep", 1))
            
            # Calculate actual tick size
            # e.g., pricePlace=1, priceEndStep=1 -> tick = 0.1
            # e.g., pricePlace=2, priceEndStep=5 -> tick = 0.05
            tick_size = (10 ** -price_decimals) * price_end_step
            
            result = {
                "tick_size": tick_size,
                "price_decimals": price_decimals,
                "size_decimals": int(info.get("volumePlace", 4)),
                "min_size": float(info.get("minTradeNum", 0.001)),
            }
            
            self.logger.debug(f"{pair} instrument: tick={tick_size}, price_dec={price_decimals}, size_dec={result['size_decimals']}")
            
            # Cache it
            self._instrument_cache[pair] = result
            return result
            
        except Exception as e:
            self.logger.warning(f"Failed to get instrument info for {pair}: {e}")
            # Return safe defaults
            return {
                "tick_size": 0.01,
                "price_decimals": 2,
                "size_decimals": 4,
                "min_size": 0.001,
            }
    
    def round_price(self, price: float, tick_size: float) -> float:
        """Round price to tick size"""
        # Round to nearest tick
        return round(round(price / tick_size) * tick_size, 10)
    
    def round_size(self, size: float, decimals: int) -> float:
        """Round size to proper decimals"""
        return round(size, int(decimals))
    
    # === ACCOUNT ===
    
    async def get_balance(self) -> float:
        """Get USDT balance"""
        if self.paper_trading:
            return self._paper_balance
        
        path = "/api/v2/mix/account/accounts"
        params = {"productType": "USDT-FUTURES"}
        data = await self._request("GET", path, params=params)
        for acc in data:
            if acc.get("marginCoin") == "USDT":
                return float(acc.get("available", 0))
        return 0.0
    
    async def get_positions(self, pair: str = None) -> List[Position]:
        """Get open positions"""
        if self.paper_trading:
            if pair:
                pos = self._paper_positions.get(pair)
                return [pos] if pos else []
            return list(self._paper_positions.values())
        
        # Helper to safely parse float (handles empty strings)
        def safe_float(val, default=0.0):
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        path = "/api/v2/mix/position/all-position"
        params = {"productType": "USDT-FUTURES"}
        if pair:
            params["symbol"] = pair
        
        data = await self._request("GET", path, params=params)
        positions = []
        
        # Handle None or non-list response
        if not data or not isinstance(data, list):
            return positions
        
        for p in data:
            symbol = p.get("symbol")
            # Filter by pair if specified (Bitget may ignore symbol param)
            if pair and symbol != pair:
                continue
                
            size = safe_float(p.get("total"))
            if size > 0:
                # Try different field names for entry price
                entry_price = safe_float(p.get("averageOpenPrice")) or safe_float(p.get("openPriceAvg")) or safe_float(p.get("entryPrice"))
                
                positions.append(Position(
                    pair=symbol,
                    side="long" if p.get("holdSide") == "long" else "short",
                    size=size,
                    entry_price=entry_price,
                    unrealized_pnl=safe_float(p.get("unrealizedPL")),
                    leverage=int(safe_float(p.get("leverage"), 1)),
                    margin_mode=p.get("marginMode", "crossed")
                ))
        return positions
    
    async def get_real_positions(self, pair: str = None) -> List[Position]:
        """Get REAL positions from exchange (ignores paper trading mode)"""
        if not self.api_key:
            return []
        
        # Helper to safely parse float (handles empty strings)
        def safe_float(val, default=0.0):
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        try:
            path = "/api/v2/mix/position/all-position"
            params = {"productType": "USDT-FUTURES"}
            if pair:
                params["symbol"] = pair
            
            data = await self._request("GET", path, params=params)
            positions = []
            
            # Handle None or non-list response
            if not data or not isinstance(data, list):
                return positions
            
            for p in data:
                symbol = p.get("symbol")
                # Filter by pair if specified (Bitget may ignore symbol param)
                if pair and symbol != pair:
                    continue
                    
                size = safe_float(p.get("total"))
                if size > 0:
                    # Try different field names for entry price
                    entry_price = safe_float(p.get("averageOpenPrice")) or safe_float(p.get("openPriceAvg")) or safe_float(p.get("entryPrice"))
                    
                    positions.append(Position(
                        pair=symbol,
                        side="long" if p.get("holdSide") == "long" else "short",
                        size=size,
                        entry_price=entry_price,
                        unrealized_pnl=safe_float(p.get("unrealizedPL")),
                        leverage=int(safe_float(p.get("leverage"), 1)),
                        margin_mode=p.get("marginMode", "crossed")
                    ))
            return positions
        except Exception as e:
            self.logger.warning(f"Failed to get real positions: {e}")
            return []
    
    async def get_real_open_orders(self, pair: str = None) -> List[OrderResult]:
        """Get ALL open orders from exchange with pagination"""
        if not self.api_key:
            return []
        
        # Helper to safely parse float (handles empty strings)
        def safe_float(val, default=0.0):
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        all_orders = []
        
        try:
            # If pair specified, get orders for that pair only
            if pair:
                path = "/api/v2/mix/order/orders-pending"
                params = {
                    "productType": "USDT-FUTURES",
                    "symbol": pair
                }
                
                data = await self._request("GET", path, params=params)
                
                if data is None:
                    return all_orders
                
                order_list = []
                if isinstance(data, list):
                    order_list = data
                elif isinstance(data, dict):
                    order_list = data.get("entrustedList", []) or []
                
                for o in order_list:
                    all_orders.append(OrderResult(
                        order_id=o.get("orderId", ""),
                        pair=o.get("symbol", ""),
                        side=o.get("side", "").lower(),
                        price=safe_float(o.get("price")),
                        amount=safe_float(o.get("size")),
                        status="pending",
                        filled_amount=safe_float(o.get("filledQty")),
                        avg_fill_price=safe_float(o.get("priceAvg"))
                    ))
            else:
                # No pair specified - need to iterate through all pairs or use pagination
                # Bitget API has a limit of 100 orders per request
                # We need to paginate using idLessThan
                
                path = "/api/v2/mix/order/orders-pending"
                last_order_id = None
                max_iterations = 20  # Safety limit (20 * 100 = 2000 orders max)
                
                for iteration in range(max_iterations):
                    params = {
                        "productType": "USDT-FUTURES",
                        "limit": "100"
                    }
                    
                    if last_order_id:
                        params["idLessThan"] = last_order_id
                    
                    data = await self._request("GET", path, params=params)
                    
                    if data is None:
                        break
                    
                    order_list = []
                    if isinstance(data, list):
                        order_list = data
                    elif isinstance(data, dict):
                        order_list = data.get("entrustedList", []) or []
                    
                    if not order_list:
                        break  # No more orders
                    
                    for o in order_list:
                        all_orders.append(OrderResult(
                            order_id=o.get("orderId", ""),
                            pair=o.get("symbol", ""),
                            side=o.get("side", "").lower(),
                            price=safe_float(o.get("price")),
                            amount=safe_float(o.get("size")),
                            status="pending",
                            filled_amount=safe_float(o.get("filledQty")),
                            avg_fill_price=safe_float(o.get("priceAvg"))
                        ))
                    
                    # Get last order ID for next page
                    if order_list:
                        last_order_id = order_list[-1].get("orderId")
                    
                    # If we got less than 100, we're done
                    if len(order_list) < 100:
                        break
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
                
                if self.logger:
                    self.logger.debug(f"Fetched {len(all_orders)} open orders from Bitget ({iteration + 1} API calls)")
            
            return all_orders
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to get real orders: {e}")
            return all_orders
    
    async def get_real_account_balance(self) -> dict:
        """Get REAL account balance from exchange with full details"""
        if not self.api_key:
            return {"available": 0, "total": 0, "unrealized_pnl": 0}
        
        try:
            path = "/api/v2/mix/account/accounts"
            params = {"productType": "USDT-FUTURES"}
            
            data = await self._request("GET", path, params=params)
            if data and len(data) > 0:
                acc = data[0]
                
                # Parse all available fields - CORRECT FIELD NAMES
                total_equity = float(acc.get("accountEquity", 0) or 0)
                
                # Available = crossedMaxAvailable (not "available" which is wallet balance!)
                available = float(acc.get("crossedMaxAvailable", 0) or acc.get("unionAvailable", 0) or 0)
                
                # Wallet balance = "available" field (confusingly named)
                wallet_balance = float(acc.get("available", 0) or 0)
                
                unrealized_pnl = float(acc.get("unrealizedPL", 0) or 0)
                
                # Used margin = crossedMargin (not "locked" which includes pending orders)
                used_margin = float(acc.get("crossedMargin", 0) or 0)
                
                usdt_balance = float(acc.get("usdtEquity", 0) or 0)
                bonus = float(acc.get("coupon", 0) or acc.get("grant", 0) or 0)
                
                # Calculate ROI based on unrealized PnL vs used margin
                roi_percent = 0
                if used_margin > 0:
                    roi_percent = (unrealized_pnl / used_margin) * 100
                
                return {
                    # Core fields
                    "available": available,  # Free margin
                    "total": total_equity,   # Account equity
                    "unrealized_pnl": unrealized_pnl,
                    "used_margin": used_margin,  # Crossed margin
                    
                    # Extended fields
                    "usdt_balance": usdt_balance,
                    "wallet_balance": wallet_balance,  # USDT balance before PnL
                    "bonus": bonus,
                    "roi_percent": round(roi_percent, 2),
                    
                    # Raw data for debugging
                    "raw": acc
                }
            return {"available": 0, "total": 0, "unrealized_pnl": 0}
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to get real balance: {e}")
            return {"available": 0, "total": 0, "unrealized_pnl": 0}
    
    # === TRADING ===
    
    async def set_position_mode(self, one_way: bool = True):
        """Set position mode to one-way or hedge"""
        if self.paper_trading:
            self.logger.info(f"[PAPER] Set position mode: {'one_way' if one_way else 'hedge'}")
            return
        
        path = "/api/v2/mix/account/set-position-mode"
        body = {
            "productType": "USDT-FUTURES",
            "posMode": "one_way_mode" if one_way else "hedge_mode"
        }
        try:
            await self._request("POST", path, body=body)
            self.logger.info(f"Set position mode to {'one_way' if one_way else 'hedge'}")
        except Exception as e:
            # Ignore if already set
            if "same" not in str(e).lower() and "already" not in str(e).lower():
                self.logger.warning(f"Could not set position mode: {e}")
    
    async def set_leverage(self, pair: str, leverage: int, margin_mode: str = "crossed"):
        """Set leverage for a pair"""
        if self.paper_trading:
            self.logger.info(f"[PAPER] Set leverage {pair}: {leverage}x {margin_mode}")
            return
        
        # Set margin mode
        path = "/api/v2/mix/account/set-margin-mode"
        body = {
            "symbol": pair,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "marginMode": margin_mode
        }
        try:
            await self._request("POST", path, body=body)
        except Exception as e:
            if "same" not in str(e).lower():
                raise
        
        # Set leverage
        path = "/api/v2/mix/account/set-leverage"
        body = {
            "symbol": pair,
            "productType": "USDT-FUTURES",
            "marginCoin": "USDT",
            "leverage": str(leverage)
        }
        await self._request("POST", path, body=body)
    
    async def place_market_order(self, pair: str, side: str, amount: float, 
                                  reduce_only: bool = False) -> OrderResult:
        """Place market order
        side: buy / sell
        """
        if self.paper_trading:
            return await self._paper_market_order(pair, side, amount, reduce_only)
        
        # Get instrument info and round size
        info = await self.get_instrument_info(pair)
        amount = self.round_size(amount, info["size_decimals"])
        
        # Ensure minimum size
        if amount < info["min_size"]:
            amount = info["min_size"]
        
        path = "/api/v2/mix/order/place-order"
        body = {
            "symbol": pair,
            "productType": "USDT-FUTURES",
            "marginMode": "crossed",
            "marginCoin": "USDT",
            "side": side,
            "orderType": "market",
            "size": str(amount),
            "reduceOnly": "YES" if reduce_only else "NO"
        }
        
        data = await self._request("POST", path, body=body)
        order_id = data.get("orderId")
        
        # Get order details
        await asyncio.sleep(0.5)
        order = await self.get_order(pair, order_id)
        return order
    
    async def place_limit_order(self, pair: str, side: str, price: float, 
                                 amount: float, reduce_only: bool = False) -> OrderResult:
        """Place limit order"""
        if self.paper_trading:
            return await self._paper_limit_order(pair, side, price, amount, reduce_only)
        
        # Get instrument info and round price/size
        info = await self.get_instrument_info(pair)
        price = self.round_price(price, info["tick_size"])
        amount = self.round_size(amount, info["size_decimals"])
        
        # Ensure minimum size
        if amount < info["min_size"]:
            amount = info["min_size"]
        
        path = "/api/v2/mix/order/place-order"
        body = {
            "symbol": pair,
            "productType": "USDT-FUTURES",
            "marginMode": "crossed",
            "marginCoin": "USDT",
            "side": side,
            "orderType": "limit",
            "price": str(price),
            "size": str(amount),
            "reduceOnly": "YES" if reduce_only else "NO"
        }
        
        data = await self._request("POST", path, body=body)
        return OrderResult(
            order_id=data.get("orderId"),
            pair=pair,
            side=side,
            price=price,
            amount=amount,
            status="pending",
            filled_amount=0,
            avg_fill_price=0
        )
    
    async def cancel_order(self, pair: str, order_id: str):
        """Cancel an order"""
        if self.paper_trading:
            if order_id in self._paper_orders:
                self._paper_orders[order_id].status = "cancelled"
                self._save_paper_state()  # Persist state
            return
        
        path = "/api/v2/mix/order/cancel-order"
        body = {
            "symbol": pair,
            "productType": "USDT-FUTURES",
            "orderId": order_id
        }
        await self._request("POST", path, body=body)
    
    async def cancel_all_orders(self, pair: str):
        """Cancel all orders for a pair"""
        if self.paper_trading:
            for oid, order in self._paper_orders.items():
                if order.pair == pair and order.status == "pending":
                    order.status = "cancelled"
            self._save_paper_state()  # Persist state
            return
        
        path = "/api/v2/mix/order/cancel-all-orders"
        body = {
            "symbol": pair,
            "productType": "USDT-FUTURES"
        }
        await self._request("POST", path, body=body)
    
    async def get_order(self, pair: str, order_id: str) -> OrderResult:
        """Get order details"""
        if self.paper_trading:
            return self._paper_orders.get(order_id)
        
        path = "/api/v2/mix/order/detail"
        params = {
            "symbol": pair,
            "productType": "USDT-FUTURES",
            "orderId": order_id
        }
        data = await self._request("GET", path, params=params)
        
        # Helper to safely parse float (handles empty strings)
        def safe_float(val, default=0.0):
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        return OrderResult(
            order_id=data.get("orderId"),
            pair=pair,
            side=data.get("side"),
            price=safe_float(data.get("price")),
            amount=safe_float(data.get("size")) or safe_float(data.get("sz")),
            status="filled" if data.get("state") == "filled" else "pending",
            filled_amount=safe_float(data.get("filledQty")) or safe_float(data.get("fillSz")) or safe_float(data.get("accFillSz")),
            avg_fill_price=safe_float(data.get("priceAvg")) or safe_float(data.get("avgPx")) or safe_float(data.get("fillPx"))
        )
    
    async def get_open_orders(self, pair: str) -> List[OrderResult]:
        """Get open orders for a pair"""
        if self.paper_trading:
            return [o for o in self._paper_orders.values() 
                    if o.pair == pair and o.status == "pending"]
        
        # Helper to safely parse float (handles empty strings)
        def safe_float(val, default=0.0):
            if val is None or val == '':
                return default
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        path = "/api/v2/mix/order/orders-pending"
        params = {
            "symbol": pair,
            "productType": "USDT-FUTURES"
        }
        data = await self._request("GET", path, params=params)
        orders = []
        
        # Handle None or empty response
        if data is None:
            return orders
        
        # API can return list directly or dict with entrustedList
        order_list = []
        if isinstance(data, list):
            order_list = data
        elif isinstance(data, dict):
            order_list = data.get("entrustedList", []) or []
        
        for o in order_list:
            orders.append(OrderResult(
                order_id=o.get("orderId"),
                pair=pair,
                side=o.get("side"),
                price=safe_float(o.get("price")),
                amount=safe_float(o.get("size")),
                status="pending",
                filled_amount=safe_float(o.get("filledQty")),
                avg_fill_price=safe_float(o.get("priceAvg"))
            ))
        return orders
    
    async def close_position(self, pair: str, side: str) -> Optional[OrderResult]:
        """Close entire position for a pair"""
        positions = await self.get_positions(pair)
        for pos in positions:
            if pos.side == side and pos.size > 0:
                close_side = "sell" if side == "long" else "buy"
                return await self.place_market_order(pair, close_side, pos.size, reduce_only=True)
        return None
    
    # === PAPER TRADING ===
    
    async def _paper_market_order(self, pair: str, side: str, amount: float, 
                                   reduce_only: bool) -> OrderResult:
        self._order_counter += 1
        order_id = f"paper_{self._order_counter}"
        
        ticker = await self.get_ticker(pair)
        price = ticker["ask"] if side == "buy" else ticker["bid"]
        
        # Update paper position
        if reduce_only:
            if pair in self._paper_positions:
                pos = self._paper_positions[pair]
                pos.size -= amount
                if pos.size <= 0:
                    del self._paper_positions[pair]
        else:
            pos_side = "long" if side == "buy" else "short"
            if pair in self._paper_positions:
                pos = self._paper_positions[pair]
                if pos.side == pos_side:
                    # Add to position
                    total_cost = pos.entry_price * pos.size + price * amount
                    pos.size += amount
                    pos.entry_price = total_cost / pos.size
                else:
                    # Reduce opposite position
                    pos.size -= amount
                    if pos.size <= 0:
                        del self._paper_positions[pair]
            else:
                self._paper_positions[pair] = Position(
                    pair=pair, side=pos_side, size=amount,
                    entry_price=price, unrealized_pnl=0, leverage=5,
                    margin_mode="crossed"
                )
        
        order = OrderResult(
            order_id=order_id, pair=pair, side=side, price=price,
            amount=amount, status="filled", filled_amount=amount, avg_fill_price=price
        )
        self._paper_orders[order_id] = order
        self._save_paper_state()  # Persist state
        self.logger.info(f"[PAPER] Market {side} {amount} {pair} @ {price}")
        return order
    
    async def _paper_limit_order(self, pair: str, side: str, price: float,
                                  amount: float, reduce_only: bool) -> OrderResult:
        self._order_counter += 1
        order_id = f"paper_{self._order_counter}"
        
        order = OrderResult(
            order_id=order_id, pair=pair, side=side, price=price,
            amount=amount, status="pending", filled_amount=0, avg_fill_price=0
        )
        self._paper_orders[order_id] = order
        self._save_paper_state()  # Persist state
        self.logger.info(f"[PAPER] Limit {side} {amount} {pair} @ {price}")
        return order
    
    async def check_paper_orders(self, pair: str):
        """Check if any paper limit orders should be filled"""
        if not self.paper_trading:
            return
        
        ticker = await self.get_ticker(pair)
        current_price = ticker["last"]
        
        for order in self._paper_orders.values():
            if order.pair == pair and order.status == "pending":
                should_fill = False
                if order.side == "buy" and current_price <= order.price:
                    should_fill = True
                elif order.side == "sell" and current_price >= order.price:
                    should_fill = True
                
                if should_fill:
                    order.status = "filled"
                    order.filled_amount = order.amount
                    order.avg_fill_price = order.price
                    self._save_paper_state()  # Persist state
                    self.logger.info(f"[PAPER] Filled {order.side} {order.amount} {pair} @ {order.price}")
    
    async def get_today_pnl(self) -> dict:
        """Get today's realized PnL from Bitget fills history"""
        if not self.api_key:
            return {"today_pnl": 0, "today_trades": 0, "fills": []}
        
        try:
            from datetime import datetime, timezone
            
            # Get start of today UTC
            now = datetime.now(timezone.utc)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = int(today_start.timestamp() * 1000)
            end_time = int(now.timestamp() * 1000)
            
            path = "/api/v2/mix/order/fill-history"
            params = {
                "productType": "USDT-FUTURES",
                "startTime": str(start_time),
                "endTime": str(end_time),
                "limit": "100"
            }
            
            data = await self._request("GET", path, params=params)
            
            if not data:
                return {"today_pnl": 0, "today_trades": 0, "fills": []}
            
            # Parse fills
            fill_list = data.get("fillList", []) if isinstance(data, dict) else data
            
            total_pnl = 0
            total_fees = 0
            trade_count = 0
            fills = []
            
            for fill in fill_list:
                profit = float(fill.get("profit", 0) or 0)
                fee = float(fill.get("fee", 0) or 0)
                total_pnl += profit
                total_fees += abs(fee)
                trade_count += 1
                
                fills.append({
                    "pair": fill.get("symbol", ""),
                    "side": fill.get("side", "").lower(),
                    "price": float(fill.get("price", 0) or 0),
                    "size": float(fill.get("size", 0) or 0),
                    "profit": profit,
                    "fee": fee,
                    "time": fill.get("cTime", "")
                })
            
            return {
                "today_pnl": round(total_pnl, 4),
                "today_fees": round(total_fees, 4),
                "today_net_pnl": round(total_pnl - total_fees, 4),
                "today_trades": trade_count,
                "fills": fills[:20]  # Return last 20 fills
            }
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to get today's PnL: {e}")
            return {"today_pnl": 0, "today_trades": 0, "fills": [], "error": str(e)}
    
    async def get_recent_fills(self, limit: int = 20) -> list:
        """Get recent fills from Bitget with profit info"""
        if not self.api_key:
            return []
        
        try:
            from datetime import datetime, timezone, timedelta
            
            # Get last 7 days to ensure we have enough fills
            now = datetime.now(timezone.utc)
            week_ago = now - timedelta(days=7)
            start_time = int(week_ago.timestamp() * 1000)
            end_time = int(now.timestamp() * 1000)
            
            path = "/api/v2/mix/order/fill-history"
            params = {
                "productType": "USDT-FUTURES",
                "startTime": str(start_time),
                "endTime": str(end_time),
                "limit": str(min(limit * 2, 100))
            }
            
            data = await self._request("GET", path, params=params)
            
            if not data:
                return []
            
            # Parse fills - handle both list and dict responses
            fill_list = []
            if isinstance(data, dict):
                fill_list = data.get("fillList", []) or []
            elif isinstance(data, list):
                fill_list = data
            
            fills = []
            for fill in fill_list[:limit]:
                # Parse timestamp - try multiple field names
                timestamp_ms = 0
                for time_field in ["cTime", "ctime", "createTime", "updatedTime", "uTime"]:
                    val = fill.get(time_field)
                    if val:
                        try:
                            timestamp_ms = int(val)
                            break
                        except:
                            continue
                
                if timestamp_ms:
                    try:
                        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                        time_str = dt.isoformat()  # Full ISO format with date
                    except:
                        time_str = ""
                else:
                    time_str = ""
                
                # Also save display time (HH:MM) for dashboard
                display_time = ""
                if timestamp_ms:
                    try:
                        dt = datetime.fromtimestamp(timestamp_ms / 1000, tz=timezone.utc)
                        display_time = dt.strftime("%H:%M")
                    except:
                        pass
                
                # Parse size - use baseVolume (not size!)
                size = 0
                for size_field in ["baseVolume", "size", "sizeQty", "qty", "filledQty", "amount"]:
                    val = fill.get(size_field)
                    if val:
                        try:
                            size = float(val)
                            if size > 0:
                                break
                        except:
                            continue
                
                # Parse profit
                profit = 0
                for profit_field in ["profit", "pnl", "realizedPnl"]:
                    val = fill.get(profit_field)
                    if val:
                        try:
                            profit = float(val)
                            break
                        except:
                            continue
                
                # Parse price
                price = 0
                for price_field in ["price", "priceAvg", "fillPrice", "avgPrice"]:
                    val = fill.get(price_field)
                    if val:
                        try:
                            price = float(val)
                            if price > 0:
                                break
                        except:
                            continue
                
                # Parse fee - it's in feeDetail array!
                fee = 0
                fee_detail = fill.get("feeDetail")
                if fee_detail and isinstance(fee_detail, list) and len(fee_detail) > 0:
                    try:
                        fee = abs(float(fee_detail[0].get("totalFee", 0) or 0))
                    except:
                        pass
                if fee == 0:
                    fee_val = fill.get("fee")
                    if fee_val:
                        try:
                            fee = abs(float(fee_val))
                        except:
                            pass
                
                fills.append({
                    "trade_id": fill.get("tradeId", ""),
                    "pair": fill.get("symbol", ""),
                    "side": (fill.get("side", "") or "").lower(),
                    "fill_price": price,
                    "fill_amount": size,
                    "profit": profit,
                    "fee": fee,
                    "filled_at": time_str,  # Full ISO format for filtering
                    "display_time": display_time,  # HH:MM for dashboard display
                    "timestamp_ms": timestamp_ms  # Raw timestamp for calculations
                })
            
            return fills
            
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to get recent fills: {e}")
            return []
