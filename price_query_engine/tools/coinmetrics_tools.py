"""
Coin Metrics API tools for the LangChain framework.
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Simple cache implementation
_API_CACHE = {}
# Shared client instance
_client = None


def get_client() -> CoinMetricsClient:
    """Get or create a shared Coin Metrics API client."""
    global _client
    if _client is None:
        api_key = os.environ.get("COINMETRICS_API_KEY")
        _client = CoinMetricsClient(api_key)
        logger.info("Initialized Coin Metrics API client")
    return _client


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
    - start_time: Start date in ISO format or natural language (e.g., "2023-01-01", "last week")
    - end_time: End date in ISO format or natural language (e.g., "2023-02-01", "today")
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
    - start_time: Start date in ISO format or natural language
    - end_time: End date in ISO format or natural language
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
    - start_time: Start date in ISO format or natural language
    - end_time: End date in ISO format or natural language
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
    start_time: str = Field(..., description="Start date in ISO format or natural language (e.g., 'yesterday', 'last week')")
    end_time: str = Field(..., description="End date in ISO format or natural language (e.g., 'today')")
    frequency: str = Field(default="1d", description="Data frequency - '1d' (daily), '1h' (hourly), etc.")

class MarketCandlesInput(BaseModel):
    markets: str = Field(..., description="Market identifier (e.g., 'binance-BTCUSDT-spot') or comma-separated list")
    start_time: str = Field(..., description="Start date in ISO format or natural language")
    end_time: str = Field(..., description="End date in ISO format or natural language")
    frequency: str = Field(default="1d", description="Data frequency - '1d' (daily), '1h' (hourly), etc.")

class ReferenceDataInput(BaseModel):
    data_type: str = Field(..., description="Type of reference data: 'assets', 'markets', 'metrics', or 'exchanges'")
    filters: dict = Field(default_factory=dict, description="Optional filters for the data")

class ExchangeMetricsInput(BaseModel):
    exchanges: str = Field(..., description="Exchange name (e.g., 'binance', 'coinbase') or comma-separated list")
    metrics: str = Field(..., description="Metric name or comma-separated list")
    start_time: str = Field(..., description="Start date in ISO format or natural language")
    end_time: str = Field(..., description="End date in ISO format or natural language")
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
