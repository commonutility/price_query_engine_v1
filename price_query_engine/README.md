# Price Query Engine

A natural language interface for accessing cryptocurrency pricing data through the Coin Metrics API. This tool allows you to query cryptocurrency data using plain English, leveraging LangChain and the Anthropic Claude model.

## Features

- Query cryptocurrency data using natural language
- Support for asset metrics, market candles, exchange data, and reference information
- Conversation memory to maintain context across queries
- Query history with search capabilities
- Interactive CLI mode for easy exploration

## Installation

Clone the repository and install:

```bash
git clone <repository-url>
cd Price_Query_Engine
pip install -e .
```

Or install directly:

```bash
pip install price-query-engine
```

## Requirements

- Python 3.7+
- Coin Metrics API key
- Anthropic API key (Claude)

## Environment Variables

Set the following environment variables:

```bash
export COINMETRICS_API_KEY="your-coin-metrics-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Usage

### Command Line Interface

#### Single Query

```bash
price-query "What was the price of Bitcoin last week?"
```

#### Interactive Mode

```bash
price-query --interactive
```

#### View Query History

```bash
price-query --history
```

#### Clear Memory or History

```bash
price-query --clear-memory
price-query --clear-history
```

### Python API

```python
from price_query_engine.engine import PriceQueryEngine

# Initialize the engine
engine = PriceQueryEngine()

# Process a query
result = engine.query("What is the trading volume of Ethereum on Binance for the past month?")

# Display the response
print(result["response"])

# Access the parameters used
print(result["parameters"])
```

## Example Queries

- "What was the price of Bitcoin last week?"
- "Show me Ethereum's trading volume on Binance for the past month"
- "Compare Bitcoin and Ethereum prices for the last 7 days"
- "What were the daily candles for BTC/USDT on Coinbase for the last 3 days?"
- "Which exchanges have the highest trading volume for Bitcoin?"
- "What's the market cap of the top 5 cryptocurrencies?"

## API Tools

The engine provides access to several Coin Metrics API endpoints:

- **AssetMetricsTool**: Get metrics for cryptocurrency assets (price, volume, etc.)
- **MarketCandlesTool**: Get OHLCV data for specific markets
- **ReferenceDataTool**: Get reference data about available assets, markets, etc.
- **ExchangeMetricsTool**: Get metrics for cryptocurrency exchanges

## Advanced Features

### Conversation Context

The engine maintains conversation context, allowing for follow-up questions:

```
> What was Bitcoin's price yesterday?
[Response with Bitcoin price data]

> How about Ethereum?
[Response with Ethereum price data, understanding from context to use yesterday's date]
```

### Query History

The engine keeps a history of your queries and their results, which can be accessed using:

```python
engine.get_recent_queries(5)  # Get the 5 most recent queries
engine.search_queries("bitcoin")  # Search for queries containing "bitcoin"
```

## License

[MIT License](LICENSE)
