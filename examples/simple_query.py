#!/usr/bin/env python3
"""
Example script demonstrating basic usage of the Price Query Engine.
"""
import os
import sys
from price_query_engine.engine import PriceQueryEngine

def main():
    """Run an interactive query session with the Price Query Engine."""
    # Check if API keys are set
    coinmetrics_api_key = os.environ.get("COINMETRICS_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    
    if not coinmetrics_api_key:
        print("Warning: COINMETRICS_API_KEY environment variable is not set.")
        print("Some functionality may be limited.")
    
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set this environment variable to use the Price Query Engine.")
        sys.exit(1)
    
    # Initialize the engine
    print("Initializing Price Query Engine...")
    engine = PriceQueryEngine(verbose=True)
    
    print("\nWelcome to the Price Query Engine!")
    print("Type your questions about cryptocurrency data, or 'exit' to quit.")
    
    # Interactive query loop
    query_count = 1
    while True:
        try:
            # Get user input
            print(f"\n--- Query {query_count} ---")
            query = input("Your query: ")
            
            # Check if user wants to exit
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting the Price Query Engine.")
                break
            
            # Process the query
            if query.strip():
                result = engine.query(query)
                
                if result["success"]:
                    print(f"\nResponse: {result['response']}")
                    
                    if result["parameters"]:
                        print("\nParameters used:")
                        for key, value in result["parameters"].items():
                            print(f"  {key}: {value}")
                else:
                    print(f"\nError: {result['response']}")
                
                query_count += 1
            else:
                print("Please enter a valid query.")
                
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
