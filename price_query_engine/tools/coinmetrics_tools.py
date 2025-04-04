"""
Coin Metrics API tools for the LangChain framework.

This module provides tools for accessing Coin Metrics API data through LangChain,
with parameter validation and normalization using the metadata registry.
"""
import os
import json
import logging
import functools
from typing import Dict, List, Any, Optional, Union, Callable

from langchain.tools import Tool, StructuredTool
from pydantic.v1 import BaseModel, Field
import pandas as pd

from coinmetrics.api_client import CoinMetricsClient
from coinmetrics.constants import PagingFrom
from price_query_engine.utils.metadata_registry import MetadataRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple cache implementation
_API_CACHE = {}
# Shared instances
_client = None
_metadata_registry = None


def get_client() -> CoinMetricsClient:
    """Get or create a shared Coin Metrics API client."""
    global _client
    if _client is None:
        api_key = os.environ.get("COINMETRICS_API_KEY")
        _client = CoinMetricsClient(api_key)
        logger.info("Initialized Coin Metrics API client")
    return _client


def get_metadata_registry() -> MetadataRegistry:
    """Get or create a shared metadata registry."""
    global _metadata_registry, _client
    if _metadata_registry is None:
        client = get_client()
        _metadata_registry = MetadataRegistry(client)
        logger.info("Initialized metadata registry")
    return _metadata_registry


def cache_result(func: Callable) -> Callable:
    """Decorator to cache API results based on function arguments."""
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Create a cache key from the function name and arguments
        key_parts = [func.__name__]
        key_parts.extend([str(arg) for arg in args])
        key_parts.extend([f"{k}={v}" for k, v in sorted(kwargs.items())])
        cache_key = "|".join(key_parts)
        
        # Check if result is in cache
        if cache_key in _API_CACHE:
            logger.info(f"Cache hit for {func.__name__}")
            return _API_CACHE[cache_key]
        
        # Call the function and cache the result
        result = func(*args, **kwargs)
        _API_CACHE[cache_key] = result
        return result
    
    return wrapper


def _parse_date(date_str: str) -> str:
    """Parse various date formats to ISO format."""
    try:
        # Try parsing as ISO format (YYYY-MM-DD)
        return date_str
    except ValueError:
        try:
            # Try parsing as a natural language date
            from dateutil import parser
            dt = parser.parse(date_str)
            return dt.date().isoformat()
        except Exception:
            raise ValueError(f"Could not parse date: {date_str}")


@cache_result
def get_asset_metrics(tool_input: str) -> str:
    """
    Get metrics for cryptocurrency assets such as price, volume, supply, etc.
    
    Parameters:
    - assets: List or single asset ticker (e.g., "btc", "eth")
    - metrics: List or single metric name (e.g., "PriceUSD", "ReferenceRate")
    - start_time: Start date in ISO 8601 format (e.g., "2020-02-29T00:00:00")
    - end_time: End date in ISO 8601 format (e.g., "2020-02-29T00:00:00")
    - frequency: Data frequency - "1d" (daily), "1h" (hourly), etc.
    """
    try:
        # Parse the input JSON
        params = json.loads(tool_input)
        
        # Validate and prepare parameters
        assets = params.get("assets")
        metrics = params.get("metrics")
        start_time = params.get("start_time")
        end_time = params.get("end_time")
        frequency = params.get("frequency", "1d")
        
        # Basic validation
        if not assets:
            return "Error: 'assets' parameter is required"
        if not metrics:
            return "Error: 'metrics' parameter is required"
        
        # Convert lists to comma-separated strings if needed
        if isinstance(assets, list):
            assets = ",".join(assets)
        if isinstance(metrics, list):
            metrics = ",".join(metrics)
        
        # Get asset metrics
        client = get_client()
        logger.info(f"Fetching asset metrics for {assets}, metrics={metrics}")
        result = client.get_asset_metrics(
            assets=assets,
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            frequency=frequency,
            paging_from=PagingFrom.START,
            page_size=10000
        )
        
        # Convert to dataframe and to JSON string
        df = result.to_dataframe()
        
        # Return a summary and the first few rows
        summary = f"Retrieved {len(df)} rows of data for {assets} ({metrics})"
        sample_data = df.head().to_json(orient="records", date_format="iso")
        
        return f"{summary}\n\nSample data:\n{sample_data}"
        
    except Exception as e:
        logger.error(f"Error in get_asset_metrics: {str(e)}")
        return f"Error: {str(e)}"


@cache_result
def get_market_candles(tool_input: str) -> str:
    """
    Get OHLCV (Open, High, Low, Close, Volume) candle data for specific markets.
    
    Parameters:
    - markets: List or single market identifier (e.g., "binance-BTCUSDT-spot")
    - start_time: Start date in ISO 8601 format (e.g., "2020-02-29T00:00:00")
    - end_time: End date in ISO 8601 format (e.g., "2020-02-29T00:00:00")
    - frequency: Data frequency - "1d" (daily), "1h" (hourly), etc.
    """
    try:
        # Parse the input JSON
        params = json.loads(tool_input)
        
        # Validate and prepare parameters
        markets = params.get("markets")
        start_time = params.get("start_time")
        end_time = params.get("end_time")
        frequency = params.get("frequency", "1d")
        
        # Basic validation
        if not markets:
            return "Error: 'markets' parameter is required"
        
        # Convert lists to comma-separated strings if needed
        if isinstance(markets, list):
            markets = ",".join(markets)
        
        # Get market candles
        client = get_client()
        logger.info(f"Fetching market candles for {markets}")
        result = client.get_market_candles(
            markets=markets,
            start_time=start_time,
            end_time=end_time,
            frequency=frequency,
            paging_from=PagingFrom.START,
            page_size=10000
        )
        
        # Convert to dataframe and to JSON string
        df = result.to_dataframe()
        
        # Return a summary and the first few rows
        summary = f"Retrieved {len(df)} candles for {markets}"
        sample_data = df.head().to_json(orient="records", date_format="iso")
        
        return f"{summary}\n\nSample data:\n{sample_data}"
        
    except Exception as e:
        logger.error(f"Error in get_market_candles: {str(e)}")
        return f"Error: {str(e)}"


@cache_result
def get_reference_data(tool_input: str) -> str:
    """
    Get reference data about available assets, markets, or metrics.
    
    Parameters:
    - data_type: Type of reference data to retrieve:
      * "assets" - available assets
      * "markets" - available markets
      * "metrics" - available metrics
      * "exchanges" - available exchanges
    - filters: Optional filters for the data
    """
    try:
        # Parse the input JSON
        params = json.loads(tool_input)
        
        # Get the data type
        data_type = params.get("data_type")
        filters = params.get("filters", {})
        
        if not data_type:
            return "Error: 'data_type' parameter is required"
        
        # Get the appropriate reference data based on type
        client = get_client()
        if data_type == "assets":
            result = client.reference_data_assets(**filters)
        elif data_type == "markets":
            result = client.reference_data_markets(**filters)
        elif data_type == "metrics":
            result = client.reference_data_asset_metrics(**filters)
        elif data_type == "exchanges":
            result = client.reference_data_exchanges(**filters)
        else:
            return f"Error: Unsupported data_type '{data_type}'"
        
        # Convert to dataframe and return first rows
        df = result.to_dataframe()
        
        # Return a summary and the first few rows
        summary = f"Retrieved {len(df)} {data_type} entries"
        
        # Limit to 10 rows to avoid overwhelming the LLM
        sample_data = df.head(10).to_json(orient="records")
        
        return f"{summary}\n\nSample data:\n{sample_data}"
        
    except Exception as e:
        logger.error(f"Error in get_reference_data: {str(e)}")
        return f"Error: {str(e)}"


@cache_result
def get_exchange_metrics(tool_input: str) -> str:
    """
    Get metrics for cryptocurrency exchanges such as volume, trade count, etc.
    
    Parameters:
    - exchanges: List or single exchange (e.g., "binance", "coinbase")
    - metrics: List or single metric name
    - start_time: Start date in ISO 8601 format (e.g., "2020-02-29T00:00:00")
    - end_time: End date in ISO 8601 format (e.g., "2020-02-29T00:00:00")
    - frequency: Data frequency - "1d" (daily), "1h" (hourly), etc.
    """
    try:
        # Parse the input JSON
        params = json.loads(tool_input)
        
        # Validate and prepare parameters
        exchanges = params.get("exchanges")
        metrics = params.get("metrics")
        start_time = params.get("start_time")
        end_time = params.get("end_time")
        frequency = params.get("frequency", "1d")
        
        # Basic validation
        if not exchanges:
            return "Error: 'exchanges' parameter is required"
        if not metrics:
            return "Error: 'metrics' parameter is required"
        
        # Convert lists to comma-separated strings if needed
        if isinstance(exchanges, list):
            exchanges = ",".join(exchanges)
        if isinstance(metrics, list):
            metrics = ",".join(metrics)
        
        # Get exchange metrics
        client = get_client()
        logger.info(f"Fetching exchange metrics for {exchanges}, metrics={metrics}")
        result = client.get_exchange_metrics(
            exchanges=exchanges,
            metrics=metrics,
            start_time=start_time,
            end_time=end_time,
            frequency=frequency,
            paging_from=PagingFrom.START,
            page_size=10000
        )
        
        # Convert to dataframe and to JSON string
        df = result.to_dataframe()
        
        # Return a summary and the first few rows
        summary = f"Retrieved {len(df)} rows of exchange data for {exchanges} ({metrics})"
        sample_data = df.head().to_json(orient="records", date_format="iso")
        
        return f"{summary}\n\nSample data:\n{sample_data}"
        
    except Exception as e:
        logger.error(f"Error in get_exchange_metrics: {str(e)}")
        return f"Error: {str(e)}"


# Define structured input schemas
class AssetMetricsInput(BaseModel):
    assets: str = Field(..., description="Asset ticker (e.g., 'btc', 'eth') or comma-separated list")
    metrics: str = Field(..., description="Metric name (e.g., 'PriceUSD', 'ReferenceRate') or comma-separated list")
    start_time: str = Field(..., description="Start date in ISO 8601 format (e.g., '2020-02-29T00:00:00')")
    end_time: str = Field(..., description="End date in ISO 8601 format (e.g., '2020-02-29T00:00:00')")
    frequency: str = Field(default="1d", description="Data frequency - '1d' (daily), '1h' (hourly), etc.")

class MarketCandlesInput(BaseModel):
    markets: str = Field(..., description="Market identifier (e.g., 'binance-BTCUSDT-spot') or comma-separated list")
    start_time: str = Field(..., description="Start date in ISO 8601 format (e.g., '2020-02-29T00:00:00')")
    end_time: str = Field(..., description="End date in ISO 8601 format (e.g., '2020-02-29T00:00:00')")
    frequency: str = Field(default="1d", description="Data frequency - '1d' (daily), '1h' (hourly), etc.")

class ReferenceDataInput(BaseModel):
    data_type: str = Field(..., description="Type of reference data: 'assets', 'markets', 'metrics', or 'exchanges'")
    filters: dict = Field(default_factory=dict, description="Optional filters for the data")

class ExchangeMetricsInput(BaseModel):
    exchanges: str = Field(..., description="Exchange name (e.g., 'binance', 'coinbase') or comma-separated list")
    metrics: str = Field(..., description="Metric name or comma-separated list")
    start_time: str = Field(..., description="Start date in ISO 8601 format (e.g., '2020-02-29T00:00:00')")
    end_time: str = Field(..., description="End date in ISO 8601 format (e.g., '2020-02-29T00:00:00')")
    frequency: str = Field(default="1d", description="Data frequency - '1d' (daily), '1h' (hourly), etc.")

# Create structured tools
def run_asset_metrics(
    assets: str,
    metrics: str,
    start_time: str,
    end_time: str,
    frequency: str = "1d"
) -> str:
    """
    Get metrics for cryptocurrency assets such as price, volume, supply, etc.
    """
    client = get_client()
    metadata = get_metadata_registry()
    
    # Split and validate assets
    asset_list = [s.strip() for s in assets.split(',')]
    validated_assets = []
    ignored_assets = []
    
    for asset in asset_list:
        normalized_asset = metadata.get_asset_symbol(asset)
        if metadata.is_valid_asset(normalized_asset):
            validated_assets.append(normalized_asset)
        else:
            ignored_assets.append(asset)
    
    if not validated_assets:
        return f"Error: No valid assets found. Please check the asset names: {assets}"
        
    if ignored_assets:
        logger.warning(f"Ignoring invalid assets: {', '.join(ignored_assets)}")
    
    # Split and validate metrics
    metric_list = [s.strip() for s in metrics.split(',')]
    validated_metrics = []
    ignored_metrics = []
    
    for metric in metric_list:
        normalized_metric = metadata.get_normalized_metric(metric)
        # Check if metric is valid for at least one of the assets
        valid_for_any = any(
            metadata.is_metric_valid_for_asset(normalized_metric, asset)
            for asset in validated_assets
        )
        
        if valid_for_any:
            validated_metrics.append(normalized_metric)
        else:
            ignored_metrics.append(metric)
    
    if not validated_metrics:
        return f"Error: No valid metrics found for the selected assets. Please check the metric names: {metrics}"
        
    if ignored_metrics:
        logger.warning(f"Ignoring invalid/unavailable metrics: {', '.join(ignored_metrics)}")
    
    # Join the validated parameters
    validated_assets_str = ','.join(validated_assets)
    validated_metrics_str = ','.join(validated_metrics)
    
    logger.info(f"Fetching asset metrics for validated assets: {validated_assets_str}, metrics: {validated_metrics_str}")
    
    # Call API with validated parameters
    result = client.get_asset_metrics(
        assets=validated_assets_str,
        metrics=validated_metrics_str,
        start_time=start_time,
        end_time=end_time,
        frequency=frequency,
        paging_from=PagingFrom.START,
        page_size=10000
    )
    
    # Convert to dataframe and to JSON string
    df = result.to_dataframe()
    
    # Print the dataframe head for visual inspection
    print("\nData preview:")
    print(df.head())
    
    # Return a summary and the first few rows
    summary = f"Retrieved {len(df)} rows of data for {validated_assets_str} ({validated_metrics_str})"
    sample_data = df.head().to_json(orient="records", date_format="iso")
    
    return f"{summary}\n\nSample data:\n{sample_data}"

def run_market_candles(
    markets: str,
    start_time: str,
    end_time: str,
    frequency: str = "1d"
) -> str:
    """
    Get OHLCV (Open, High, Low, Close, Volume) candle data for specific markets.
    """
    client = get_client()
    metadata = get_metadata_registry()
    
    # Split and validate markets
    market_list = [s.strip() for s in markets.split(',')]
    validated_markets = []
    ignored_markets = []
    
    for market in market_list:
        normalized_market = metadata.get_normalized_market(market)
        if metadata.is_valid_market(normalized_market):
            validated_markets.append(normalized_market)
        else:
            ignored_markets.append(market)
    
    if not validated_markets:
        return f"Error: No valid markets found. Please check the market identifiers: {markets}"
        
    if ignored_markets:
        logger.warning(f"Ignoring invalid markets: {', '.join(ignored_markets)}")
    
    # Join the validated parameters
    validated_markets_str = ','.join(validated_markets)
    
    logger.info(f"Fetching market candles for validated markets: {validated_markets_str}")
    
    # Call API with validated parameters
    result = client.get_market_candles(
        markets=validated_markets_str,
        start_time=start_time,
        end_time=end_time,
        frequency=frequency,
        paging_from=PagingFrom.START,
        page_size=10000
    )
    
    # Convert to dataframe and to JSON string
    df = result.to_dataframe()
    
    # Print the dataframe head for visual inspection
    print("\nData preview:")
    print(df.head())
    
    # Return a summary and the first few rows
    summary = f"Retrieved {len(df)} candles for {validated_markets_str}"
    sample_data = df.head().to_json(orient="records", date_format="iso")
    
    return f"{summary}\n\nSample data:\n{sample_data}"

def run_reference_data(
    data_type: str,
    filters: dict = {}
) -> str:
    """
    Get reference data about available assets, markets, or metrics.
    """
    client = get_client()
    if data_type == "assets":
        result = client.reference_data_assets(**filters)
    elif data_type == "markets":
        result = client.reference_data_markets(**filters)
    elif data_type == "metrics":
        result = client.reference_data_asset_metrics(**filters)
    elif data_type == "exchanges":
        result = client.reference_data_exchanges(**filters)
    else:
        return f"Error: Unsupported data_type '{data_type}'"
    
    # Convert to dataframe and return first rows
    df = result.to_dataframe()
    
    # Print the dataframe head for visual inspection
    print("\nData preview:")
    print(df.head(10))
    
    # Return a summary and the first few rows
    summary = f"Retrieved {len(df)} {data_type} entries"
    
    # Limit to 10 rows to avoid overwhelming the LLM
    sample_data = df.head(10).to_json(orient="records")
    
    return f"{summary}\n\nSample data:\n{sample_data}"

def run_exchange_metrics(
    exchanges: str,
    metrics: str,
    start_time: str,
    end_time: str,
    frequency: str = "1d"
) -> str:
    """
    Get metrics for cryptocurrency exchanges such as volume, trade count, etc.
    """
    client = get_client()
    metadata = get_metadata_registry()
    
    # Split and validate exchanges
    exchange_list = [s.strip() for s in exchanges.split(',')]
    validated_exchanges = []
    ignored_exchanges = []
    
    # Ideally we would validate against a list of valid exchanges from the metadata
    # For simplicity, we'll pass through the exchanges for now as this would require additional metadata methods
    validated_exchanges = exchange_list
    
    # Join the validated parameters
    validated_exchanges_str = ','.join(validated_exchanges)
    
    logger.info(f"Fetching exchange metrics for validated exchanges: {validated_exchanges_str}, metrics={metrics}")
    
    # Call API with validated parameters
    result = client.get_exchange_metrics(
        exchanges=validated_exchanges_str,
        metrics=metrics,
        start_time=start_time,
        end_time=end_time,
        frequency=frequency,
        paging_from=PagingFrom.START,
        page_size=10000
    )
    
    # Convert to dataframe and to JSON string
    df = result.to_dataframe()
    
    # Print the dataframe head for visual inspection
    print("\nData preview:")
    print(df.head())
    
    # Return a summary and the first few rows
    summary = f"Retrieved {len(df)} rows of exchange data for {validated_exchanges_str} ({metrics})"
    sample_data = df.head().to_json(orient="records", date_format="iso")
    
    return f"{summary}\n\nSample data:\n{sample_data}"

# Create tools
asset_metrics_tool = StructuredTool.from_function(
    func=run_asset_metrics,
    name="get_asset_metrics",
    description="Get metrics for cryptocurrency assets such as price, volume, supply, etc."
)

market_candles_tool = StructuredTool.from_function(
    func=run_market_candles,
    name="get_market_candles",
    description="Get OHLCV (Open, High, Low, Close, Volume) candle data for specific markets."
)

reference_data_tool = StructuredTool.from_function(
    func=run_reference_data,
    name="get_reference_data",
    description="Get reference data about available assets, markets, or metrics."
)

exchange_metrics_tool = StructuredTool.from_function(
    func=run_exchange_metrics,
    name="get_exchange_metrics",
    description="Get metrics for cryptocurrency exchanges such as volume, trade count, etc."
)
