"""
AI Analysis Module for Grid Bot
Uses Claude API for sentiment analysis and market regime detection
"""

import asyncio
import aiohttp
import json
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import re


class Sentiment(Enum):
    VERY_BULLISH = "very_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    VERY_BEARISH = "very_bearish"


class MarketRegime(Enum):
    STRONG_UPTREND = "strong_uptrend"
    UPTREND = "uptrend"
    RANGING = "ranging"
    DOWNTREND = "downtrend"
    STRONG_DOWNTREND = "strong_downtrend"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"


@dataclass
class SentimentResult:
    sentiment: Sentiment
    score: float  # -1 to 1
    confidence: float  # 0 to 1
    summary: str
    key_factors: List[str]
    recommended_action: str
    timestamp: str


@dataclass
class RegimeResult:
    regime: MarketRegime
    confidence: float
    characteristics: List[str]
    recommended_grid_mode: str  # neutral / long / short
    recommended_range_adjustment: float  # multiplier
    reasoning: str
    timestamp: str


@dataclass
class AIConfig:
    enabled: bool = True
    claude_api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    
    # API Keys for news sources
    coingecko_api_key: str = ""      # CoinGecko Pro API key
    cryptocompare_api_key: str = ""  # CryptoCompare API key
    
    # News sources
    news_enabled: bool = True
    news_sources: List[str] = field(default_factory=lambda: ["coingecko", "cryptocompare", "rss"])
    news_lookback_hours: int = 24
    news_update_interval_minutes: int = 30
    
    # Regime detection
    regime_enabled: bool = True
    regime_update_interval_minutes: int = 15
    
    # Trading integration
    auto_adjust_mode: bool = True  # Auto change grid_mode based on AI
    auto_adjust_range: bool = True  # Auto adjust range based on regime
    min_confidence: float = 0.7  # Minimum confidence to act


class NewsAggregator:
    """Aggregates news from multiple FREE sources + CoinGecko Basic"""
    
    # CoinGecko Basic Plan: 100k calls/month, 250/min
    # No asset restrictions needed!
    
    # Asset name mappings for different APIs
    ASSET_NAMES = {
        "BTC": ["bitcoin", "btc"],
        "ETH": ["ethereum", "eth"],
        "SOL": ["solana", "sol"],
        "XRP": ["ripple", "xrp"],
        "ADA": ["cardano", "ada"],
        "AVAX": ["avalanche", "avax"],
        "DOT": ["polkadot", "dot"],
        "LINK": ["chainlink", "link"],
        "ATOM": ["cosmos", "atom"],
        "NEAR": ["near-protocol", "near"],
        "UNI": ["uniswap", "uni"],
        "OP": ["optimism-ethereum", "op"],
        "ARB": ["arbitrum", "arb"],
        "SUI": ["sui", "sui"],
        "DOGE": ["dogecoin", "doge"],
        "SHIB": ["shiba-inu", "shib"],
        "PEPE": ["pepe", "pepe"],
        "WIF": ["dogwifcoin", "wif"],
        "BNB": ["binancecoin", "bnb"],
        "MATIC": ["matic-network", "matic"],
        "TRX": ["tron", "trx"],
        "LTC": ["litecoin", "ltc"],
        "BCH": ["bitcoin-cash", "bch"],
        "ETC": ["ethereum-classic", "etc"],
        "FIL": ["filecoin", "fil"],
        "HBAR": ["hedera-hashgraph", "hbar"],
        "ALGO": ["algorand", "algo"],
        "RUNE": ["thorchain", "rune"],
        "CRV": ["curve-dao-token", "crv"],
        "AAVE": ["aave", "aave"],
        "TAO": ["bittensor", "tao"],
        "IOTA": ["iota", "iota"],
        "ZEC": ["zcash", "zec"],
        "ENA": ["ethena", "ena"],
        "HYPE": ["hyperliquid", "hype"],
        "ASTER": ["aster", "aster"],
    }
    
    # CoinGecko coin IDs (first element is the official ID)
    COINGECKO_IDS = {
        "BTC": "bitcoin",
        "ETH": "ethereum", 
        "SOL": "solana",
        "XRP": "ripple",
        "ADA": "cardano",
        "AVAX": "avalanche-2",
        "DOT": "polkadot",
        "LINK": "chainlink",
        "ATOM": "cosmos",
        "NEAR": "near",
        "UNI": "uniswap",
        "OP": "optimism",
        "ARB": "arbitrum",
        "SUI": "sui",
        "DOGE": "dogecoin",
        "SHIB": "shiba-inu",
        "PEPE": "pepe",
        "WIF": "dogwifcoin",
        "BNB": "binancecoin",
        "MATIC": "matic-network",
        "TRX": "tron",
        "LTC": "litecoin",
        "BCH": "bitcoin-cash",
        "ETC": "ethereum-classic",
        "FIL": "filecoin",
        "HBAR": "hedera-hashgraph",
        "ALGO": "algorand",
        "RUNE": "thorchain",
        "CRV": "curve-dao-token",
        "AAVE": "aave",
        "TAO": "bittensor",
        "IOTA": "iota",
        "ZEC": "zcash",
        "ENA": "ethena",
        "HYPE": "hyperliquid",
        "ASTER": "aster-defi",
    }
    
    def __init__(self, logger: logging.Logger, coingecko_api_key: str = ""):
        self.logger = logger
        self.coingecko_api_key = coingecko_api_key
        self._cache: Dict[str, List[Dict]] = {}
        self._cache_time: Dict[str, datetime] = {}
        self._cache_duration_minutes = 30  # Cache news for 30 min (Basic plan has plenty of quota)
        
        # CoinGecko API base URL
        if coingecko_api_key:
            self.coingecko_base = "https://pro-api.coingecko.com/api/v3"
            self.logger.info(f"CoinGecko Pro API initialized (key: {coingecko_api_key[:8]}...)")
        else:
            self.coingecko_base = "https://api.coingecko.com/api/v3"
            self.logger.warning("CoinGecko API key not set - using public API (limited)")
    
    def _get_coingecko_headers(self) -> Dict:
        """Get headers for CoinGecko API"""
        headers = {"accept": "application/json"}
        if self.coingecko_api_key:
            headers["x-cg-pro-api-key"] = self.coingecko_api_key
        return headers
    
    def _get_search_terms(self, asset: str) -> List[str]:
        """Get search terms for an asset"""
        asset_upper = asset.upper()
        if asset_upper in self.ASSET_NAMES:
            return self.ASSET_NAMES[asset_upper]
        return [asset.lower()]
    
    def _get_coingecko_id(self, asset: str) -> str:
        """Get CoinGecko coin ID for an asset"""
        return self.COINGECKO_IDS.get(asset.upper(), asset.lower())
    
    def _is_cache_valid(self, asset: str) -> bool:
        """Check if cached data is still valid"""
        if asset not in self._cache_time:
            return False
        age = (datetime.utcnow() - self._cache_time[asset]).total_seconds() / 60
        return age < self._cache_duration_minutes
    
    async def fetch_news(self, pair: str, hours: int = 24) -> List[Dict]:
        """Fetch news for a trading pair from multiple sources"""
        # Extract base asset (BTCUSDT -> BTC)
        asset = pair.replace("USDT", "").replace("USD", "")
        
        # Check cache first
        if self._is_cache_valid(asset) and asset in self._cache:
            self.logger.debug(f"Using cached news for {asset}")
            return self._cache[asset]
        
        all_news = []
        
        # 1. CoinGecko coin data + status updates (Basic plan - all assets!)
        self.logger.debug(f"Fetching CoinGecko data for {asset}...")
        coingecko_news = await self._fetch_coingecko_news(asset, hours)
        all_news.extend(coingecko_news)
        self.logger.debug(f"CoinGecko returned {len(coingecko_news)} items for {asset}")
        
        # 2. CoinGecko trending (global market sentiment)
        if asset in ["BTC", "ETH"]:
            trending = await self._fetch_coingecko_trending()
            all_news.extend(trending)
        
        # 3. CryptoCompare News (FREE - 100k calls/month backup)
        cryptocompare_news = await self._fetch_cryptocompare(asset, hours)
        all_news.extend(cryptocompare_news)
        
        # 4. RSS Feeds (FREE - unlimited)
        rss_news = await self._fetch_rss_feeds(asset, hours)
        all_news.extend(rss_news)
        
        # 5. Reddit for major assets
        if asset in ["BTC", "ETH", "SOL", "DOGE", "XRP"]:
            reddit_news = await self._fetch_reddit(asset, hours)
            all_news.extend(reddit_news)
        
        # Remove duplicates based on title similarity
        unique_news = self._deduplicate_news(all_news)
        
        # Sort by date (newest first)
        unique_news.sort(key=lambda x: x.get('published', ''), reverse=True)
        
        # Cache results
        result = unique_news[:20]
        self._cache[asset] = result
        self._cache_time[asset] = datetime.utcnow()
        
        # Log summary
        sources = {}
        for item in result:
            src = item.get('source', 'Unknown')
            sources[src] = sources.get(src, 0) + 1
        self.logger.info(f"News for {asset}: {len(result)} items from {len(sources)} sources")
        
        return result
    
    async def _fetch_coingecko_news(self, asset: str, hours: int) -> List[Dict]:
        """Fetch from CoinGecko coin data (Basic plan)"""
        news = []
        coin_id = self._get_coingecko_id(asset)
        
        try:
            # Get coin data with market data for context
            url = f"{self.coingecko_base}/coins/{coin_id}?localization=false&tickers=false&market_data=true&community_data=true&developer_data=false&sparkline=false"
            
            self.logger.debug(f"CoinGecko request: {url[:80]}...")
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, 
                    headers=self._get_coingecko_headers(),
                    timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    self.logger.debug(f"CoinGecko response for {asset}: status={resp.status}")
                    
                    if resp.status == 200:
                        data = await resp.json()
                        
                        # Get market data for sentiment
                        market_data = data.get("market_data", {})
                        price_change_24h = market_data.get("price_change_percentage_24h", 0)
                        price_change_7d = market_data.get("price_change_percentage_7d", 0)
                        
                        # Create market update news item
                        if price_change_24h:
                            direction = "📈" if price_change_24h > 0 else "📉"
                            news.append({
                                "title": f"{direction} {asset} {price_change_24h:+.2f}% (24h), {price_change_7d:+.2f}% (7d)",
                                "source": "CoinGecko Market",
                                "url": f"https://www.coingecko.com/en/coins/{coin_id}",
                                "published": datetime.utcnow().isoformat(),
                                "asset": asset,
                                "sentiment_hint": "bullish" if price_change_24h > 5 else "bearish" if price_change_24h < -5 else "neutral"
                            })
                        
                        # Community data
                        community = data.get("community_data", {})
                        twitter_followers = community.get("twitter_followers", 0)
                        reddit_subscribers = community.get("reddit_subscribers", 0)
                        
                        if twitter_followers > 100000 or reddit_subscribers > 50000:
                            news.append({
                                "title": f"{asset} community: {twitter_followers:,} Twitter, {reddit_subscribers:,} Reddit",
                                "source": "CoinGecko Social",
                                "url": f"https://www.coingecko.com/en/coins/{coin_id}",
                                "published": datetime.utcnow().isoformat(),
                                "asset": asset
                            })
                        
                        # Get status updates if available
                        status_updates = data.get("status_updates", [])
                        cutoff = datetime.utcnow() - timedelta(hours=hours)
                        
                        for update in status_updates[:5]:
                            try:
                                created = datetime.fromisoformat(update.get("created_at", "").replace("Z", "+00:00"))
                                if created.replace(tzinfo=None) > cutoff:
                                    news.append({
                                        "title": update.get("description", "")[:200],
                                        "source": f"CoinGecko ({update.get('category', 'update')})",
                                        "url": update.get("url", ""),
                                        "published": update.get("created_at", ""),
                                        "asset": asset
                                    })
                            except:
                                continue
                    else:
                        self.logger.warning(f"CoinGecko returned {resp.status} for {coin_id}")
                        
        except Exception as e:
            self.logger.warning(f"CoinGecko fetch error for {asset}: {e}")
        
        return news
    
    async def _fetch_coingecko_trending(self) -> List[Dict]:
        """Fetch trending coins from CoinGecko (market sentiment indicator)"""
        news = []
        
        try:
            url = f"{self.coingecko_base}/search/trending"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_coingecko_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        coins = data.get("coins", [])[:5]
                        
                        if coins:
                            trending_names = [c.get("item", {}).get("name", "") for c in coins]
                            news.append({
                                "title": f"🔥 Trending on CoinGecko: {', '.join(trending_names)}",
                                "source": "CoinGecko Trending",
                                "url": "https://www.coingecko.com/en/trending-cryptocurrencies",
                                "published": datetime.utcnow().isoformat(),
                                "asset": "MARKET"
                            })
                            
        except Exception as e:
            self.logger.debug(f"CoinGecko trending error: {e}")
        
        return news
    
    async def fetch_market_overview(self) -> Dict:
        """Fetch global market data from CoinGecko (Basic plan feature)"""
        try:
            url = f"{self.coingecko_base}/global"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url,
                    headers=self._get_coingecko_headers(),
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        global_data = data.get("data", {})
                        
                        return {
                            "total_market_cap": global_data.get("total_market_cap", {}).get("usd", 0),
                            "total_volume": global_data.get("total_volume", {}).get("usd", 0),
                            "btc_dominance": global_data.get("market_cap_percentage", {}).get("btc", 0),
                            "eth_dominance": global_data.get("market_cap_percentage", {}).get("eth", 0),
                            "market_cap_change_24h": global_data.get("market_cap_change_percentage_24h_usd", 0),
                            "active_cryptocurrencies": global_data.get("active_cryptocurrencies", 0),
                            "markets": global_data.get("markets", 0),
                        }
                        
        except Exception as e:
            self.logger.debug(f"CoinGecko global error: {e}")
        
        return {}
    
    async def _fetch_cryptocompare(self, asset: str, hours: int) -> List[Dict]:
        """Fetch from CryptoCompare News API (FREE tier - 100k calls/month)"""
        news = []
        
        try:
            # CryptoCompare news API - free tier
            url = f"https://min-api.cryptocompare.com/data/v2/news/?categories={asset}&extraParams=gridbot"
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        articles = data.get("Data", [])
                        
                        cutoff = datetime.utcnow() - timedelta(hours=hours)
                        
                        for article in articles[:15]:
                            try:
                                pub_time = datetime.fromtimestamp(article.get("published_on", 0))
                                if pub_time > cutoff:
                                    news.append({
                                        "title": article.get("title", ""),
                                        "source": article.get("source", "CryptoCompare"),
                                        "url": article.get("url", ""),
                                        "published": pub_time.isoformat(),
                                        "asset": asset,
                                        "body": article.get("body", "")[:200]
                                    })
                            except:
                                continue
                    else:
                        self.logger.debug(f"CryptoCompare returned {resp.status}")
                        
        except Exception as e:
            self.logger.debug(f"CryptoCompare fetch error: {e}")
        
        return news
    
    async def _fetch_rss_feeds(self, asset: str, hours: int) -> List[Dict]:
        """Fetch from free RSS feeds (CoinDesk, CoinTelegraph, etc.)"""
        news = []
        search_terms = self._get_search_terms(asset)
        
        # Free RSS feeds for crypto news
        rss_feeds = [
            "https://cointelegraph.com/rss",
            "https://www.coindesk.com/arc/outboundfeeds/rss/",
            "https://decrypt.co/feed",
            "https://bitcoinmagazine.com/.rss/full/",
        ]
        
        try:
            async with aiohttp.ClientSession() as session:
                for feed_url in rss_feeds:
                    try:
                        async with session.get(
                            feed_url, 
                            timeout=aiohttp.ClientTimeout(total=8),
                            headers={"User-Agent": "Mozilla/5.0 GridBot/1.0"}
                        ) as resp:
                            if resp.status == 200:
                                text = await resp.text()
                                root = ET.fromstring(text)
                                
                                # Find all items (RSS 2.0 format)
                                items = root.findall('.//item')
                                
                                cutoff = datetime.utcnow() - timedelta(hours=hours)
                                
                                for item in items[:10]:
                                    title = item.find('title')
                                    title_text = title.text if title is not None else ""
                                    
                                    # Check if news is relevant to our asset
                                    title_lower = title_text.lower()
                                    is_relevant = any(term in title_lower for term in search_terms)
                                    
                                    # Also check for general crypto news for major assets
                                    if asset in ["BTC", "ETH"]:
                                        is_relevant = is_relevant or any(
                                            term in title_lower 
                                            for term in ["crypto", "bitcoin", "ethereum", "defi", "web3"]
                                        )
                                    
                                    if is_relevant:
                                        link = item.find('link')
                                        pub_date = item.find('pubDate')
                                        
                                        news.append({
                                            "title": title_text,
                                            "source": feed_url.split('/')[2],
                                            "url": link.text if link is not None else "",
                                            "published": pub_date.text if pub_date is not None else datetime.utcnow().isoformat(),
                                            "asset": asset
                                        })
                    except Exception as e:
                        self.logger.debug(f"RSS feed {feed_url} error: {e}")
                        continue
                        
        except Exception as e:
            self.logger.debug(f"RSS fetch error: {e}")
        
        return news
    
    async def _fetch_reddit(self, asset: str, hours: int) -> List[Dict]:
        """Fetch from Reddit (FREE, no auth needed for public data)"""
        news = []
        search_terms = self._get_search_terms(asset)
        
        # Subreddits to check
        subreddits = ["cryptocurrency", "bitcoin", "ethtrader", "CryptoMarkets"]
        
        # Add asset-specific subreddit if exists
        asset_subreddits = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "SOL": "solana",
            "ADA": "cardano",
            "DOT": "polkadot",
            "AVAX": "avax",
            "ATOM": "cosmosnetwork",
            "LINK": "chainlink",
            "XRP": "ripple",
            "DOGE": "dogecoin",
        }
        
        if asset in asset_subreddits:
            subreddits.insert(0, asset_subreddits[asset])
        
        try:
            async with aiohttp.ClientSession() as session:
                for subreddit in subreddits[:3]:  # Limit to 3 subreddits
                    try:
                        # Reddit JSON API (no auth needed)
                        url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=15"
                        
                        headers = {"User-Agent": "GridBot/1.0"}
                        
                        async with session.get(
                            url, 
                            timeout=aiohttp.ClientTimeout(total=8),
                            headers=headers
                        ) as resp:
                            if resp.status == 200:
                                data = await resp.json()
                                posts = data.get("data", {}).get("children", [])
                                
                                cutoff = datetime.utcnow() - timedelta(hours=hours)
                                
                                for post in posts:
                                    post_data = post.get("data", {})
                                    title = post_data.get("title", "")
                                    title_lower = title.lower()
                                    
                                    # Check relevance
                                    is_relevant = any(term in title_lower for term in search_terms)
                                    
                                    # For general crypto subreddits, be more selective
                                    if subreddit in ["cryptocurrency", "CryptoMarkets"]:
                                        is_relevant = is_relevant and post_data.get("score", 0) > 50
                                    
                                    if is_relevant:
                                        created = datetime.fromtimestamp(post_data.get("created_utc", 0))
                                        if created > cutoff:
                                            news.append({
                                                "title": f"[Reddit] {title}",
                                                "source": f"r/{subreddit}",
                                                "url": f"https://reddit.com{post_data.get('permalink', '')}",
                                                "published": created.isoformat(),
                                                "asset": asset,
                                                "score": post_data.get("score", 0),
                                                "comments": post_data.get("num_comments", 0)
                                            })
                            
                            # Rate limit for Reddit
                            await asyncio.sleep(0.5)
                            
                    except Exception as e:
                        self.logger.debug(f"Reddit r/{subreddit} error: {e}")
                        continue
                        
        except Exception as e:
            self.logger.debug(f"Reddit fetch error: {e}")
        
        return news
    
    def _deduplicate_news(self, news: List[Dict]) -> List[Dict]:
        """Remove duplicate news items based on title similarity"""
        unique = []
        seen_titles = set()
        
        for item in news:
            # Create a simplified title for comparison
            title = item.get("title", "").lower()
            # Remove common words and get first 50 chars
            simple_title = ''.join(c for c in title if c.isalnum())[:50]
            
            if simple_title and simple_title not in seen_titles:
                seen_titles.add(simple_title)
                unique.append(item)
        
        return unique
    
    async def fetch_social_sentiment(self, asset: str) -> Dict:
        """Fetch social media sentiment indicators"""
        # Placeholder - could integrate with LunarCrush, Santiment, etc.
        return {
            "twitter_mentions": 0,
            "reddit_posts": 0,
            "sentiment_score": 0
        }


class ClaudeAnalyzer:
    """Uses Claude API for intelligent analysis"""
    
    def __init__(self, api_key: str, model: str, logger: logging.Logger):
        self.api_key = api_key
        self.model = model
        self.logger = logger
        self.api_url = "https://api.anthropic.com/v1/messages"
    
    async def _call_claude(self, system_prompt: str, user_prompt: str, 
                          max_tokens: int = 1000) -> Optional[str]:
        """Make a call to Claude API"""
        if not self.api_key:
            self.logger.warning("Claude API key not configured")
            return None
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.api_url,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data["content"][0]["text"]
                    else:
                        error = await resp.text()
                        self.logger.error(f"Claude API error {resp.status}: {error}")
                        return None
        except Exception as e:
            self.logger.error(f"Claude API call failed: {e}")
            return None
    
    async def analyze_sentiment(self, asset: str, news: List[Dict]) -> Optional[SentimentResult]:
        """Analyze news sentiment using Claude"""
        if not news:
            return SentimentResult(
                sentiment=Sentiment.NEUTRAL,
                score=0,
                confidence=0.5,
                summary="No recent news available",
                key_factors=[],
                recommended_action="hold",
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Prepare news summary for Claude
        news_text = "\n".join([
            f"- [{n['source']}] {n['title']}"
            for n in news[:15]  # Limit to 15 most recent
        ])
        
        system_prompt = """You are a crypto market analyst AI. Analyze news sentiment for trading decisions.

You must respond in valid JSON format only, with no additional text. Use this exact structure:
{
    "sentiment": "very_bullish|bullish|neutral|bearish|very_bearish",
    "score": <float from -1 to 1>,
    "confidence": <float from 0 to 1>,
    "summary": "<brief summary of overall sentiment>",
    "key_factors": ["<factor1>", "<factor2>", "<factor3>"],
    "recommended_action": "strong_buy|buy|hold|sell|strong_sell"
}

Scoring guide:
- very_bullish (0.7 to 1.0): Major positive catalysts, institutional adoption, regulatory wins
- bullish (0.3 to 0.7): Positive developments, upgrades, partnerships
- neutral (-0.3 to 0.3): Mixed news, no clear direction
- bearish (-0.7 to -0.3): Negative developments, concerns, minor issues
- very_bearish (-1.0 to -0.7): Major negative events, hacks, regulatory crackdowns"""

        user_prompt = f"""Analyze the sentiment of these recent news headlines for {asset}:

{news_text}

Consider:
1. Overall market impact
2. Short-term vs long-term implications
3. Credibility and significance of news sources
4. How this might affect price action

Respond with JSON only."""

        response = await self._call_claude(system_prompt, user_prompt)
        
        if not response:
            return None
        
        try:
            # Parse JSON response
            # Clean up response if needed
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return SentimentResult(
                sentiment=Sentiment(data["sentiment"]),
                score=float(data["score"]),
                confidence=float(data["confidence"]),
                summary=data["summary"],
                key_factors=data["key_factors"],
                recommended_action=data["recommended_action"],
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            self.logger.error(f"Failed to parse sentiment response: {e}")
            self.logger.debug(f"Response was: {response}")
            return None
    
    async def detect_market_regime(self, asset: str, 
                                   price_data: List[Dict],
                                   current_price: float,
                                   volatility: float,
                                   trend_info: Dict) -> Optional[RegimeResult]:
        """Detect market regime using Claude"""
        if not price_data:
            return None
        
        # Prepare price summary
        prices = [float(p["close"]) for p in price_data[-48:]]  # Last 48 hours
        
        if len(prices) < 10:
            return None
        
        price_change_24h = (prices[-1] - prices[-24]) / prices[-24] * 100 if len(prices) >= 24 else 0
        price_change_48h = (prices[-1] - prices[0]) / prices[0] * 100
        
        high_24h = max(prices[-24:]) if len(prices) >= 24 else max(prices)
        low_24h = min(prices[-24:]) if len(prices) >= 24 else min(prices)
        range_24h = (high_24h - low_24h) / low_24h * 100
        
        # Calculate simple momentum
        momentum = sum(1 if prices[i] > prices[i-1] else -1 for i in range(1, len(prices)))
        
        system_prompt = """You are a market regime detection AI for crypto trading. Classify the current market state.

You must respond in valid JSON format only, with no additional text. Use this exact structure:
{
    "regime": "strong_uptrend|uptrend|ranging|downtrend|strong_downtrend|high_volatility|low_volatility",
    "confidence": <float from 0 to 1>,
    "characteristics": ["<char1>", "<char2>", "<char3>"],
    "recommended_grid_mode": "long|neutral|short",
    "recommended_range_adjustment": <float multiplier, e.g., 1.0 for normal, 1.5 for wider>,
    "reasoning": "<brief explanation>"
}

Regime definitions:
- strong_uptrend: Sustained bullish momentum, >5% gain, high confidence continuation
- uptrend: Bullish bias, 2-5% gain, moderate momentum
- ranging: Sideways movement, <2% change, mean-reverting
- downtrend: Bearish bias, 2-5% loss, moderate selling pressure
- strong_downtrend: Sustained bearish momentum, >5% loss, high confidence continuation
- high_volatility: Large swings regardless of direction, >8% range
- low_volatility: Tight range, <3% movement, compression

IMPORTANT - Grid mode MUST match regime:
- strong_uptrend: ALWAYS "long"
- uptrend: ALWAYS "long"
- ranging: ALWAYS "neutral"
- downtrend: ALWAYS "short"
- strong_downtrend: ALWAYS "short"
- high_volatility: "neutral" with range_adjustment > 1.3
- low_volatility: "neutral" with range_adjustment < 0.8

Never recommend "neutral" for uptrend/downtrend regimes."""

        user_prompt = f"""Analyze the current market regime for {asset}:

Current price: ${current_price:.2f}
24h change: {price_change_24h:.2f}%
48h change: {price_change_48h:.2f}%
24h range: {range_24h:.2f}%
Volatility (daily): {volatility:.2f}%
Momentum score: {momentum} (positive = bullish candles dominating)
Trend direction: {trend_info.get('direction', 'unknown')}
Trend strength: {trend_info.get('strength', 0):.0f}%

Recent price action (last 12 data points):
{', '.join([f'${p:.0f}' for p in prices[-12:]])}

Based on this data, classify the market regime and provide trading recommendations.
Respond with JSON only."""

        response = await self._call_claude(system_prompt, user_prompt)
        
        if not response:
            return None
        
        try:
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(response)
            
            return RegimeResult(
                regime=MarketRegime(data["regime"]),
                confidence=float(data["confidence"]),
                characteristics=data["characteristics"],
                recommended_grid_mode=data["recommended_grid_mode"],
                recommended_range_adjustment=float(data["recommended_range_adjustment"]),
                reasoning=data["reasoning"],
                timestamp=datetime.utcnow().isoformat()
            )
        except Exception as e:
            self.logger.error(f"Failed to parse regime response: {e}")
            self.logger.debug(f"Response was: {response}")
            return None


class AITradingAdvisor:
    """
    Main AI advisor that combines sentiment and regime analysis
    """
    
    def __init__(self, config: AIConfig, logger: logging.Logger,
                 notification_manager=None):
        self.config = config
        self.logger = logger
        self.notifications = notification_manager
        
        self.news_aggregator = NewsAggregator(logger, config.coingecko_api_key)
        self.claude = ClaudeAnalyzer(config.claude_api_key, config.model, logger)
        
        # Cache results
        self._sentiment_cache: Dict[str, SentimentResult] = {}
        self._regime_cache: Dict[str, RegimeResult] = {}
        self._recommendation_cache: Dict[str, Dict] = {}  # Full recommendations
        self._last_update: Dict[str, datetime] = {}
    
    async def get_trading_recommendation(self, pair: str,
                                         price_data: List[Dict],
                                         current_price: float,
                                         volatility: float,
                                         trend_info: Dict) -> Dict:
        """
        Get comprehensive AI trading recommendation
        """
        if not self.config.enabled:
            return self._default_recommendation()
        
        recommendations = {
            "pair": pair,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": None,
            "regime": None,
            "final_recommendation": {
                "grid_mode": "neutral",
                "range_multiplier": 1.0,
                "confidence": 0.5,
                "reasoning": []
            }
        }
        
        # Get sentiment analysis
        if self.config.news_enabled:
            sentiment = await self._get_sentiment(pair)
            if sentiment:
                recommendations["sentiment"] = {
                    "sentiment": sentiment.sentiment.value,
                    "score": sentiment.score,
                    "confidence": sentiment.confidence,
                    "summary": sentiment.summary,
                    "action": sentiment.recommended_action
                }
        
        # Get regime detection
        if self.config.regime_enabled:
            regime = await self._get_regime(pair, price_data, current_price, 
                                           volatility, trend_info)
            if regime:
                recommendations["regime"] = {
                    "regime": regime.regime.value,
                    "confidence": regime.confidence,
                    "grid_mode": regime.recommended_grid_mode,
                    "range_adjustment": regime.recommended_range_adjustment,
                    "reasoning": regime.reasoning
                }
        
        # Combine recommendations
        final = self._combine_recommendations(
            recommendations.get("sentiment"),
            recommendations.get("regime")
        )
        recommendations["final_recommendation"] = final
        
        # Cache full recommendation for dashboard
        self._recommendation_cache[pair] = recommendations
        
        return recommendations
    
    async def _get_sentiment(self, pair: str) -> Optional[SentimentResult]:
        """Get sentiment with caching"""
        cache_key = f"sentiment_{pair}"
        
        # Check cache
        if cache_key in self._sentiment_cache:
            last_update = self._last_update.get(cache_key)
            if last_update:
                age_minutes = (datetime.utcnow() - last_update).total_seconds() / 60
                if age_minutes < self.config.news_update_interval_minutes:
                    return self._sentiment_cache[cache_key]
        
        # Fetch fresh data
        try:
            news = await self.news_aggregator.fetch_news(
                pair, self.config.news_lookback_hours
            )
            
            asset = pair.replace("USDT", "").replace("USD", "")
            sentiment = await self.claude.analyze_sentiment(asset, news)
            
            if sentiment:
                self._sentiment_cache[cache_key] = sentiment
                self._last_update[cache_key] = datetime.utcnow()
                
                self.logger.info(f"AI Sentiment {pair}: {sentiment.sentiment.value} "
                               f"(score: {sentiment.score:.2f}, conf: {sentiment.confidence:.2f})")
            
            return sentiment
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed for {pair}: {e}")
            return self._sentiment_cache.get(cache_key)
    
    async def _get_regime(self, pair: str, price_data: List[Dict],
                         current_price: float, volatility: float,
                         trend_info: Dict) -> Optional[RegimeResult]:
        """Get regime with caching"""
        cache_key = f"regime_{pair}"
        
        # Check cache
        if cache_key in self._regime_cache:
            last_update = self._last_update.get(cache_key)
            if last_update:
                age_minutes = (datetime.utcnow() - last_update).total_seconds() / 60
                if age_minutes < self.config.regime_update_interval_minutes:
                    return self._regime_cache[cache_key]
        
        # Fetch fresh analysis
        try:
            asset = pair.replace("USDT", "").replace("USD", "")
            regime = await self.claude.detect_market_regime(
                asset, price_data, current_price, volatility, trend_info
            )
            
            if regime:
                self._regime_cache[cache_key] = regime
                self._last_update[cache_key] = datetime.utcnow()
                
                self.logger.info(f"AI Regime {pair}: {regime.regime.value} "
                               f"(conf: {regime.confidence:.2f}, mode: {regime.recommended_grid_mode})")
            
            return regime
        except Exception as e:
            self.logger.error(f"Regime detection failed for {pair}: {e}")
            return self._regime_cache.get(cache_key)
    
    def _combine_recommendations(self, sentiment: Optional[Dict],
                                 regime: Optional[Dict]) -> Dict:
        """Combine sentiment and regime into final recommendation"""
        result = {
            "grid_mode": "neutral",
            "range_multiplier": 1.0,
            "confidence": 0.5,
            "reasoning": []
        }
        
        # Map regime to grid_mode
        regime_to_mode = {
            "strong_uptrend": "long",
            "uptrend": "long", 
            "ranging": "neutral",
            "downtrend": "short",
            "strong_downtrend": "short",
            "high_volatility": "neutral",
            "low_volatility": "neutral"
        }
        
        # Start with regime recommendation (60% weight)
        regime_mode = "neutral"
        regime_conf = 0.5
        if regime:
            regime_value = regime.get("regime", "ranging")
            regime_mode = regime.get("grid_mode") or regime_to_mode.get(regime_value, "neutral")
            regime_conf = regime.get("confidence", 0.5)
            result["range_multiplier"] = regime.get("range_adjustment", 1.0)
            result["reasoning"].append(f"Regime: {regime_value} → {regime_mode}")
        
        # Get sentiment mode (40% weight)
        sentiment_mode = "neutral"
        sentiment_conf = 0.5
        if sentiment:
            sentiment_value = sentiment.get("sentiment", "neutral")
            score = sentiment.get("score", 0)
            sentiment_mode = self._sentiment_to_mode(sentiment_value, score)
            sentiment_conf = sentiment.get("confidence", 0.5)
            result["reasoning"].append(f"Sentiment: {sentiment_value} (score: {score:.2f}) → {sentiment_mode}")
        
        # Combine modes with weighted voting
        # Regime = 60%, Sentiment = 40%
        if regime_mode == sentiment_mode:
            # Agreement - high confidence boost!
            result["grid_mode"] = regime_mode
            result["confidence"] = min(1.0, (regime_conf * 0.6 + sentiment_conf * 0.4) * 1.5)  # 1.5x boost
            result["reasoning"].append("✓ Regime & sentiment agree")
        elif sentiment_mode == "neutral":
            # Sentiment neutral, use regime with good confidence
            result["grid_mode"] = regime_mode
            result["confidence"] = regime_conf * 0.9  # High trust in regime alone
            result["reasoning"].append(f"Using regime ({regime_mode}), sentiment neutral")
        elif regime_mode == "neutral":
            # Regime neutral, use sentiment if has direction
            score = sentiment.get("score", 0) if sentiment else 0
            if abs(score) > 0.3:  # Sentiment has direction
                result["grid_mode"] = sentiment_mode
                result["confidence"] = sentiment_conf * 0.85
                result["reasoning"].append(f"Using sentiment ({sentiment_mode}), regime neutral")
            else:
                result["grid_mode"] = "neutral"
                result["confidence"] = 0.6
                result["reasoning"].append("Both neutral")
        else:
            # Disagreement - prefer stronger signal
            score = abs(sentiment.get("score", 0)) if sentiment else 0
            
            if score > 0.5 and sentiment_conf > 0.6:
                # Strong sentiment wins
                result["grid_mode"] = sentiment_mode
                result["confidence"] = sentiment_conf * 0.7
                result["reasoning"].append(f"Strong sentiment ({sentiment_mode}) overrides regime")
            elif regime_conf > 0.6:
                # Strong regime wins
                result["grid_mode"] = regime_mode
                result["confidence"] = regime_conf * 0.7
                result["reasoning"].append(f"Regime ({regime_mode}) preferred over sentiment")
            else:
                # Neither is strong - use the one with higher confidence
                if regime_conf >= sentiment_conf:
                    result["grid_mode"] = regime_mode
                    result["confidence"] = regime_conf * 0.6
                    result["reasoning"].append(f"Weak conflict, using regime ({regime_mode})")
                else:
                    result["grid_mode"] = sentiment_mode
                    result["confidence"] = sentiment_conf * 0.6
                    result["reasoning"].append(f"Weak conflict, using sentiment ({sentiment_mode})")
        
        # Only force neutral if confidence is very low
        min_conf = self.config.min_confidence if hasattr(self.config, 'min_confidence') else 0.35
        # But cap at 0.4 to prevent always-neutral
        min_conf = min(min_conf, 0.4)
        
        if result["confidence"] < min_conf:
            result["grid_mode"] = "neutral"
            result["reasoning"].append(f"Low confidence ({result['confidence']:.0%}) → neutral")
        
        return result
    
    def _sentiment_to_mode(self, sentiment: str, score: float) -> str:
        """Convert sentiment to grid mode"""
        if sentiment in ["very_bullish", "bullish"] or score > 0.3:
            return "long"
        elif sentiment in ["very_bearish", "bearish"] or score < -0.3:
            return "short"
        return "neutral"
    
    def _default_recommendation(self) -> Dict:
        """Return default recommendation when AI is disabled"""
        return {
            "pair": "",
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": None,
            "regime": None,
            "final_recommendation": {
                "grid_mode": "neutral",
                "range_multiplier": 1.0,
                "confidence": 0.5,
                "reasoning": ["AI analysis disabled"]
            }
        }
    
    def get_cached_analysis(self, pair: str) -> Dict:
        """Get cached analysis for display"""
        # Try to get full recommendation first
        if pair in self._recommendation_cache:
            return self._recommendation_cache[pair]
        
        # Fallback to building from individual caches
        sentiment_result = self._sentiment_cache.get(f"sentiment_{pair}")
        regime_result = self._regime_cache.get(f"regime_{pair}")
        
        sentiment_dict = None
        if sentiment_result:
            sentiment_dict = {
                "sentiment": sentiment_result.sentiment.value if hasattr(sentiment_result, 'sentiment') else None,
                "score": sentiment_result.score if hasattr(sentiment_result, 'score') else 0,
                "confidence": sentiment_result.confidence if hasattr(sentiment_result, 'confidence') else 0,
                "summary": sentiment_result.summary if hasattr(sentiment_result, 'summary') else ""
            }
        
        regime_dict = None
        if regime_result:
            regime_dict = {
                "regime": regime_result.regime.value if hasattr(regime_result, 'regime') else None,
                "confidence": regime_result.confidence if hasattr(regime_result, 'confidence') else 0,
                "grid_mode": regime_result.recommended_grid_mode if hasattr(regime_result, 'recommended_grid_mode') else "neutral"
            }
        
        return {
            "pair": pair,
            "timestamp": datetime.utcnow().isoformat(),
            "sentiment": sentiment_dict,
            "regime": regime_dict,
            "final_recommendation": self._combine_recommendations(sentiment_dict, regime_dict) if (sentiment_dict or regime_dict) else {
                "grid_mode": "neutral",
                "range_multiplier": 1.0,
                "confidence": 0.5,
                "reasoning": []
            }
        }
    
    async def notify_significant_change(self, pair: str, 
                                       old_recommendation: Dict,
                                       new_recommendation: Dict):
        """Send notification when AI recommendation changes significantly"""
        if not self.notifications:
            return
        
        old_mode = old_recommendation.get("final_recommendation", {}).get("grid_mode", "neutral")
        new_mode = new_recommendation.get("final_recommendation", {}).get("grid_mode", "neutral")
        
        if old_mode != new_mode:
            await self.notifications.notify(
                notification_type="optimization",
                title=f"🤖 AI Recommendation Change: {pair}",
                message=f"Grid mode: {old_mode} → {new_mode}\n"
                       f"Confidence: {new_recommendation['final_recommendation']['confidence']:.0%}\n"
                       f"Reasoning: {', '.join(new_recommendation['final_recommendation']['reasoning'])}",
                color=0x9b59b6
            )
