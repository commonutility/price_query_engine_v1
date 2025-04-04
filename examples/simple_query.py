#!/usr/bin/env python3
"""
Example script demonstrating basic usage of the Price Query Engine.
"""
import os
import sys
from price_query_engine.engine import PriceQueryEngine

def main():
    """Run a simple query example."""
    # Check if API keys are set
    coinmetrics_api_key = os.environ.get("COINMETRICS_API_KEY")
    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    
    if not coinmetrics_api_key:
        print("Warning: COINMETRICS_API_KEY environment variable is not set.")
        print("Some functionality may be limited.")
    
    if not anthropic_api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please set this environment variable to use the Price Query Engine.")
        sys.exit(1)
    
    # Initialize the engine
    print("Initializing Price Query Engine...")
    engine = PriceQueryEngine(verbose=True)
    
    # Example queries
    example_queries = [
        "What was the price of Bitcoin yesterday?",
        "How about Ethereum?",
        "Show me the trading volume of BTC on Binance for the last week",
        "What exchanges have the highest Bitcoin volume?",
    ]
    
    # Process each query
    for i, query in enumerate(example_queries, 1):
        print(f"\n--- Query {i}: {query} ---")
        
        result = engine.query(query)
        
        if result["success"]:
            print(f"\nResponse: {result['response']}")
            
            if result["parameters"]:
                print("\nParameters used:")
                for key, value in result["parameters"].items():
                    print(f"  {key}: {value}")
        else:
            print(f"\nError: {result['response']}")
        
        print("\nPress Enter to continue to the next query, or Ctrl+C to exit...")
        input()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
