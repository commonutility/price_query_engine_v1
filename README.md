# Price Query Engine for Coin Metrics API

A natural language interface for accessing cryptocurrency pricing data through the Coin Metrics API. This project wraps the Coin Metrics Python API Client with a LangChain-powered natural language processing framework, allowing users to query cryptocurrency data using plain English.

## Overview

The Price Query Engine provides a bridge between natural language queries and the structured data available through the Coin Metrics API. It uses the Anthropic Claude model to interpret user queries, determine the appropriate API calls, and present the results in a human-friendly format.

## Features

- **Natural Language Interface**: Query cryptocurrency data using plain English
- **Comprehensive Data Access**: Support for asset metrics, market candles, exchange data, and reference information
- **Conversation Memory**: Maintains context across queries for follow-up questions
- **Query History**: Stores and allows searching of past queries
- **Interactive CLI**: User-friendly command-line interface
- **Caching**: Reduces redundant API calls

## Installation

### Prerequisites

- Python 3.7+
- Coin Metrics API key
- Anthropic API key (Claude)

### Install from Source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Price_Query_Engine.git
cd Price_Query_Engine
```

2. Install the package:
```bash
pip install -e .
```

## Environment Setup

Set your API keys as environment variables:

```bash
export COINMETRICS_API_KEY="your-coin-metrics-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Usage

### Command Line Interface

The simplest way to use the Price Query Engine is through its command-line interface:

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

You can also use the Price Query Engine as a Python library:

```python
from price_query_engine.engine import PriceQueryEngine

# Initialize the engine
engine = PriceQueryEngine()

# Process a query
result = engine.query("What is the trading volume of Ethereum on Binance for the past month?")

# Display the response
print(result["response"])

# Access the parameters used for the query
print(result["parameters"])
```

## Example Queries

- "What was the price of Bitcoin last week?"
- "Show me Ethereum's trading volume on Binance for the past month"
- "Compare Bitcoin and Ethereum prices for the last 7 days"
- "What were the daily candles for BTC/USDT on Coinbase for the last 3 days?"
- "Which exchanges have the highest trading volume for Bitcoin?"
- "What's the market cap of the top 5 cryptocurrencies?"

## Repository Structure

```
price_query_engine/
├── cli/                 # Command-line interface
├── memory/              # Conversation and query history
├── tools/               # Coin Metrics API tool wrappers
├── engine.py            # Core query processing engine
├── __init__.py          # Package initialization
└── __main__.py          # Entry point for running as a module
```

## Running Examples

The repository includes an example script that demonstrates basic usage:

```bash
python examples/simple_query.py
```

## Requirements

This project depends on:
- langchain
- coinmetrics-api-client
- anthropic
- pandas
- python-dateutil

## License

[MIT License](LICENSE)
