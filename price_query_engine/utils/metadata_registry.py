"""
Metadata registry for the Price Query Engine.

This module provides a cache of metadata from the Coin Metrics API,
including assets, markets, metrics, and exchanges. It helps validate
and normalize parameters for API calls.
"""
import os
import json
import time
import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime

try:
    from fuzzywuzzy import process as fuzzy_process
    FUZZY_MATCHING_AVAILABLE = True
except ImportError:
    FUZZY_MATCHING_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("metadata_registry")

class MetadataRegistry:
    """
    A registry that caches metadata from the Coin Metrics API.
    
    This registry stores information about assets, markets, metrics, and exchanges,
    and provides methods to validate and normalize parameters for API calls.
    """
    
    def __init__(self, client, cache_ttl=3600, cache_file=None):
        """
        Initialize the metadata registry.
        
        Args:
            client: CoinMetricsClient instance
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
            cache_file: Path to file for persistent cache (default: ~/.price_query_engine_metadata_cache.json)
        """
        self.client = client
        self.cache_ttl = cache_ttl
        
        if cache_file is None:
            self.cache_file = os.path.join(
                os.path.expanduser("~"),
                ".price_query_engine_metadata_cache.json"
            )
        else:
            self.cache_file = cache_file
            
        self.last_updated = {}
        self.cache = {
            'assets': {},
            'markets': {},
            'metrics': {},
            'exchanges': {},
            'asset_metrics_map': {},  # Maps assets to their available metrics
            'exchange_markets_map': {},  # Maps exchanges to their available markets
            'market_available_freqs': {},  # Maps markets to available frequencies
            'assets_by_name': {}  # Maps asset names to their symbols
        }
        
        # Try to load cache from file, refresh if needed or if load fails
        if not self.load_cache():
            self.refresh_all_metadata()

    def refresh_all_metadata(self):
        """Refresh all metadata caches"""
        logger.info("Refreshing all metadata")
        self._cache_assets()
        self._cache_markets() 
        self._cache_metrics()
        self._cache_exchanges()
        self._build_relationship_maps()
        self.save_cache()
        
    def _cache_assets(self):
        """Fetch and cache asset data"""
        try:
            assets_df = self.client.reference_data_assets().to_dataframe()
            
            # Primary lookup by asset symbol
            self.cache['assets'] = {
                row['asset']: row for row in assets_df.to_dict('records')
            }
            
            # Secondary lookup by name (lowercase for case-insensitive matching)
            self.cache['assets_by_name'] = {}
            for asset_data in self.cache['assets'].values():
                if 'name' in asset_data and asset_data['name']:
                    name_key = asset_data['name'].lower()
                    self.cache['assets_by_name'][name_key] = asset_data['asset']
            
            self.last_updated['assets'] = time.time()
            logger.info(f"Cached metadata for {len(self.cache['assets'])} assets")
        except Exception as e:
            logger.error(f"Error caching assets: {str(e)}")

    def _cache_markets(self):
        """Fetch and cache market data"""
        try:
            markets_df = self.client.reference_data_markets().to_dataframe()
            
            # Primary lookup by market identifier
            self.cache['markets'] = {
                row['market']: row for row in markets_df.to_dict('records')
            }
            
            self.last_updated['markets'] = time.time()
            logger.info(f"Cached metadata for {len(self.cache['markets'])} markets")
        except Exception as e:
            logger.error(f"Error caching markets: {str(e)}")

    def _cache_metrics(self):
        """Fetch and cache metric data"""
        try:
            metrics_df = self.client.reference_data_asset_metrics().to_dataframe()
            
            # Primary lookup by metric name
            self.cache['metrics'] = {
                row['metric']: row for row in metrics_df.to_dict('records')
            }
            
            self.last_updated['metrics'] = time.time()
            logger.info(f"Cached metadata for {len(self.cache['metrics'])} metrics")
        except Exception as e:
            logger.error(f"Error caching metrics: {str(e)}")

    def _cache_exchanges(self):
        """Fetch and cache exchange data"""
        try:
            exchanges_df = self.client.reference_data_exchanges().to_dataframe()
            
            # Primary lookup by exchange name
            self.cache['exchanges'] = {
                row['exchange']: row for row in exchanges_df.to_dict('records')
            }
            
            self.last_updated['exchanges'] = time.time()
            logger.info(f"Cached metadata for {len(self.cache['exchanges'])} exchanges")
        except Exception as e:
            logger.error(f"Error caching exchanges: {str(e)}")

    def _build_relationship_maps(self):
        """Build maps of relationships between different entities"""
        try:
            # 1. Map assets to their available metrics
            self._build_asset_metrics_map()
            
            # 2. Map exchanges to their available markets
            self._build_exchange_markets_map()
            
            # 3. Map markets to their available frequencies
            self._build_market_frequencies_map()
            
            logger.info("Built relationship maps")
        except Exception as e:
            logger.error(f"Error building relationship maps: {str(e)}")

    def _build_asset_metrics_map(self):
        """Build a map of assets to their available metrics"""
        asset_metrics_map = {}
        
        try:
            # Get all assets first
            assets = list(self.cache['assets'].keys())
            
            # Get all metrics
            metrics = list(self.cache['metrics'].keys())
            
            # For simplicity, let's assume all metrics are available for all assets
            # This is not ideal but avoids the API compatibility issue
            for asset in assets:
                asset_metrics_map[asset] = metrics
            
            # Convert sets to lists for serialization
            self.cache['asset_metrics_map'] = asset_metrics_map
            
            logger.info(f"Built asset-to-metrics mapping for {len(asset_metrics_map)} assets")
        except Exception as e:
            logger.error(f"Error building asset metrics map: {str(e)}")

    def _build_exchange_markets_map(self):
        """Build a map of exchanges to their available markets"""
        exchange_markets_map = {}
        
        try:
            for market_id, market_data in self.cache['markets'].items():
                # Extract exchange from market ID (typically the first part before dash)
                parts = market_id.split('-')
                if len(parts) > 1:
                    exchange = parts[0]
                    if exchange not in exchange_markets_map:
                        exchange_markets_map[exchange] = []
                    exchange_markets_map[exchange].append(market_id)
            
            self.cache['exchange_markets_map'] = exchange_markets_map
            logger.info(f"Built exchange-to-markets mapping for {len(exchange_markets_map)} exchanges")
        except Exception as e:
            logger.error(f"Error building exchange markets map: {str(e)}")

    def _build_market_frequencies_map(self):
        """Build a map of markets to their available frequencies"""
        market_freqs_map = {}
        
        try:
            # This would typically come from catalog_market_candles or similar
            # For now, we'll use a placeholder with common frequencies
            common_freqs = ["1s", "1m", "5m", "15m", "1h", "4h", "1d"]
            
            for market_id in self.cache['markets'].keys():
                market_freqs_map[market_id] = common_freqs
                
            self.cache['market_available_freqs'] = market_freqs_map
            logger.info(f"Built market-to-frequencies mapping for {len(market_freqs_map)} markets")
        except Exception as e:
            logger.error(f"Error building market frequencies map: {str(e)}")

    def get_asset_symbol(self, asset_input):
        """
        Convert various asset inputs to canonical asset symbols.
        Handles exact matches, fuzzy matches, and common name variations.
        
        Examples:
        - "BTC" → "btc"
        - "Bitcoin" → "btc"
        - "Ethereum" → "eth"
        - "bticoin" → "btc" (fuzzy match)
        
        Args:
            asset_input: String input representing an asset
            
        Returns:
            Normalized asset symbol or the original input if no match found
        """
        if not asset_input:
            return None
            
        # Check if cache needs refresh
        self._check_refresh('assets')
        
        # Direct lookup (case insensitive)
        asset_lower = asset_input.lower()
        for key in self.cache['assets'].keys():
            if key.lower() == asset_lower:
                return key
        
        # Name lookup
        if asset_lower in self.cache['assets_by_name']:
            return self.cache['assets_by_name'][asset_lower]
        
        # Fuzzy matching for minor typos (using Levenshtein distance if available)
        if FUZZY_MATCHING_AVAILABLE:
            # Try to match against asset symbols
            asset_symbols = list(self.cache['assets'].keys())
            symbol_match = fuzzy_process.extractOne(asset_lower, asset_symbols, score_cutoff=85)
            if symbol_match:
                return symbol_match[0]
                
            # Try to match against asset names
            asset_names = list(self.cache['assets_by_name'].keys())
            name_match = fuzzy_process.extractOne(asset_lower, asset_names, score_cutoff=85)
            if name_match:
                return self.cache['assets_by_name'][name_match[0]]
        
        # No match found
        return asset_input

    def get_normalized_metric(self, metric_input):
        """
        Convert metric input to a canonical metric name.
        
        Args:
            metric_input: String input representing a metric
            
        Returns:
            Normalized metric name or the original input if no match found
        """
        if not metric_input:
            return None
            
        # Check if cache needs refresh
        self._check_refresh('metrics')
        
        # Direct lookup (case insensitive)
        metric_lower = metric_input.lower()
        for key in self.cache['metrics'].keys():
            if key.lower() == metric_lower:
                return key
        
        # Fuzzy matching if available
        if FUZZY_MATCHING_AVAILABLE:
            metric_names = list(self.cache['metrics'].keys())
            match = fuzzy_process.extractOne(metric_lower, metric_names, score_cutoff=85)
            if match:
                return match[0]
        
        # No match found
        return metric_input

    def get_normalized_market(self, market_input):
        """
        Convert market input to a canonical market identifier.
        
        Args:
            market_input: String input representing a market
            
        Returns:
            Normalized market identifier or the original input if no match found
        """
        if not market_input:
            return None
            
        # Check if cache needs refresh
        self._check_refresh('markets')
        
        # Direct lookup (case sensitive, as market IDs are exact)
        if market_input in self.cache['markets']:
            return market_input
        
        # Case-insensitive lookup
        market_lower = market_input.lower()
        for key in self.cache['markets'].keys():
            if key.lower() == market_lower:
                return key
        
        # Try to construct a valid market ID
        # Example: input might be "binance btc-usdt" and we need "binance-BTCUSDT-spot"
        
        # No match found
        return market_input

    def get_valid_metrics_for_asset(self, asset):
        """
        Get the list of metrics available for a specific asset.
        
        Args:
            asset: Asset symbol
            
        Returns:
            List of metric names available for the asset, or empty list if none found
        """
        asset_symbol = self.get_asset_symbol(asset)
        self._check_refresh('asset_metrics_map')
        
        if asset_symbol in self.cache['asset_metrics_map']:
            return self.cache['asset_metrics_map'][asset_symbol]
        return []

    def get_valid_markets_for_exchange(self, exchange):
        """
        Get the list of markets available for a specific exchange.
        
        Args:
            exchange: Exchange name
            
        Returns:
            List of market identifiers for the exchange, or empty list if none found
        """
        self._check_refresh('exchange_markets_map')
        
        if exchange in self.cache['exchange_markets_map']:
            return self.cache['exchange_markets_map'][exchange]
        return []

    def is_valid_asset(self, asset):
        """Check if an asset is valid"""
        asset_symbol = self.get_asset_symbol(asset)
        return asset_symbol in self.cache['assets']

    def is_valid_metric(self, metric):
        """Check if a metric is valid"""
        metric_name = self.get_normalized_metric(metric)
        return metric_name in self.cache['metrics']

    def is_valid_market(self, market):
        """Check if a market is valid"""
        market_id = self.get_normalized_market(market)
        return market_id in self.cache['markets']

    def is_metric_valid_for_asset(self, metric, asset):
        """Check if a metric is valid for a specific asset"""
        asset_symbol = self.get_asset_symbol(asset)
        metric_name = self.get_normalized_metric(metric)
        
        valid_metrics = self.get_valid_metrics_for_asset(asset_symbol)
        return metric_name in valid_metrics

    def _check_refresh(self, cache_type):
        """Check if a specific cache needs refreshing based on TTL"""
        current_time = time.time()
        if (cache_type not in self.last_updated or 
                current_time - self.last_updated.get(cache_type, 0) > self.cache_ttl):
            logger.info(f"Refreshing {cache_type} cache (TTL expired)")
            
            if cache_type == 'assets':
                self._cache_assets()
            elif cache_type == 'markets':
                self._cache_markets()
            elif cache_type == 'metrics':
                self._cache_metrics()
            elif cache_type == 'exchanges':
                self._cache_exchanges()
            elif cache_type == 'asset_metrics_map':
                self._build_asset_metrics_map()
            elif cache_type == 'exchange_markets_map':
                self._build_exchange_markets_map()
            elif cache_type == 'market_available_freqs':
                self._build_market_frequencies_map()

    def load_cache(self):
        """Load cache from disk
        
        Returns:
            bool: True if cache was successfully loaded and is still valid, False otherwise
        """
        try:
            if not os.path.exists(self.cache_file):
                logger.info(f"Cache file {self.cache_file} does not exist")
                return False
                
            # Check file modification time
            file_age = time.time() - os.path.getmtime(self.cache_file)
            if file_age > self.cache_ttl:
                logger.info(f"Cache file is too old ({file_age:.1f} seconds)")
                return False
                
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                
            if 'cache' not in data or 'last_updated' not in data:
                logger.warning("Invalid cache file format")
                return False
                
            self.cache = data['cache']
            self.last_updated = data['last_updated']
            
            # Check if any individual cache components are expired
            current_time = time.time()
            for cache_type, timestamp in self.last_updated.items():
                if current_time - timestamp > self.cache_ttl:
                    logger.info(f"Cache component {cache_type} is expired")
                    return False
                    
            logger.info(f"Loaded metadata cache from {self.cache_file}")
            return True
        except Exception as e:
            logger.error(f"Error loading metadata cache: {str(e)}")
            return False
            
    def save_cache(self):
        """Save cache to disk"""
        try:
            cache_dir = os.path.dirname(self.cache_file)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            
            # Create a sanitized copy of the cache for JSON serialization
            sanitized_cache = self._sanitize_for_json(self.cache)
            
            with open(self.cache_file, 'w') as f:
                json.dump({
                    'cache': sanitized_cache,
                    'last_updated': self.last_updated
                }, f)
            logger.info(f"Saved metadata cache to {self.cache_file}")
        except Exception as e:
            logger.error(f"Error saving metadata cache: {str(e)}")
    def _sanitize_for_json(self, obj):
        """Recursively sanitize an object for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._sanitize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize_for_json(item) for item in obj]
        elif isinstance(obj, (set, tuple)):
            return [self._sanitize_for_json(item) for item in obj]
        elif pd.isna(obj):  # Handle NaT, NaN, None
            return None
        elif hasattr(obj, 'isoformat'):  # Handle datetime-like objects
            return obj.isoformat()
        else:
            return obj
            
    def get_metadata_context(self):
        """
        Generate a context string with information about available assets, metrics, exchanges, etc.
        This is used to provide the AI assistant with information about what data is available.
        
        Returns:
            str: A formatted string with metadata context information
        """
        # Ensure cache is refreshed
        self._check_refresh('assets')
        self._check_refresh('metrics')
        self._check_refresh('exchanges')
        
        # Count how many assets have each metric
        asset_count = len(self.cache['assets'])
        metric_count = len(self.cache['metrics'])
        exchange_count = len(self.cache['exchanges'])
        market_count = len(self.cache['markets'])
        
        # Get top assets by market cap (if available)
        top_assets = []
        try:
            # Simple approach - just get some well-known assets
            common_assets = ["btc", "eth", "xrp", "usdt", "usdc", "sol", "ada", "avax", "dot", "doge"]
            top_assets = [asset for asset in common_assets if asset in self.cache['assets']]
            if not top_assets:
                # Fallback to first 10 assets in cache
                top_assets = list(self.cache['assets'].keys())[:10]
        except Exception as e:
            logger.error(f"Error getting top assets: {str(e)}")
            top_assets = list(self.cache['assets'].keys())[:10]
            
        # Get popular metrics
        popular_metrics = [
            "PriceUSD", "ReferenceRate", "TxCnt", "BlkCnt", "AdrActCnt",
            "FeeTotUSD", "DiffMean", "HashRate", "MarketCapUSD", "VolumeUSD"
        ]
        available_popular_metrics = [m for m in popular_metrics if m in self.cache['metrics']]
        
        # Get exchanges
        top_exchanges = list(self.cache['exchanges'].keys())[:10]
        
        # Build context string
        context = f"""
Available Data Context:
- {asset_count} cryptocurrency assets
- {metric_count} different metrics
- {exchange_count} exchanges
- {market_count} markets

Popular cryptocurrencies you can query include: {', '.join(top_assets)}

Common metrics you can query include: {', '.join(available_popular_metrics)}

Available exchanges include: {', '.join(top_exchanges)}

When users ask about cryptocurrency data, help them get the specific data they need.
- If they ask about a specific asset (like Bitcoin), match it to the appropriate asset symbol (btc).
- If they ask about prices, determine if they want current prices or historical data.
- If they ask about exchanges, markets, or specific metrics, use the appropriate tools.
"""
        
        return context
