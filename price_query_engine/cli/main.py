"""
Command-line interface for the Price Query Engine.
"""
import os
import sys
import argparse
from typing import Optional, List, Dict, Any

from price_query_engine.engine import PriceQueryEngine

def setup_argparse() -> argparse.ArgumentParser:
    """Set up the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        description="Natural language interface for cryptocurrency pricing data"
    )
    
    # Main command options
    parser.add_argument(
        "query",
        nargs="?",
        help="Natural language query for cryptocurrency data"
    )
    
    # Mode options
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    
    parser.add_argument(
        "--history", "-H",
        action="store_true",
        help="Show query history"
    )
    
    parser.add_argument(
        "--clear-memory", "-C",
        action="store_true",
        help="Clear conversation memory"
    )
    
    parser.add_argument(
        "--clear-history", "-X",
        action="store_true",
        help="Clear query history"
    )
    
    # Configuration options
    parser.add_argument(
        "--api-key",
        help="Coin Metrics API key (defaults to COINMETRICS_API_KEY env var)"
    )
    
    parser.add_argument(
        "--anthropic-api-key",
        help="Anthropic API key (defaults to ANTHROPIC_API_KEY env var)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output (shows agent's thinking)"
    )
    
    return parser

def run_interactive_mode(engine: PriceQueryEngine) -> None:
    """Run the CLI in interactive mode."""
    print("\nPrice Query Engine Interactive Mode")
    print("Type 'exit', 'quit', or Ctrl+C to exit")
    print("Type 'history' to view recent queries")
    print("Type 'clear memory' to clear conversation memory")
    print("Type 'clear history' to clear query history")
    print("-------------------------------------------")
    
    try:
        while True:
            query = input("\n> ")
            
            if query.lower() in ("exit", "quit"):
                break
            elif query.lower() == "history":
                display_history(engine)
                continue
            elif query.lower() == "clear memory":
                engine.clear_memory()
                continue
            elif query.lower() == "clear history":
                engine.clear_history()
                continue
            elif not query.strip():
                continue
            
            # Process the query
            result = engine.query(query)
            
            # Display the response
            if result["success"]:
                print(f"\n{result['response']}")
            else:
                print(f"\nError: {result['response']}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nError: {str(e)}")

def display_history(engine: PriceQueryEngine, n: int = 5) -> None:
    """Display recent query history."""
    recent_queries = engine.get_recent_queries(n)
    
    if not recent_queries:
        print("\nNo query history found.")
        return
    
    print(f"\nRecent Queries ({len(recent_queries)}):")
    print("-------------------------------------------")
    
    for i, entry in enumerate(recent_queries, 1):
        timestamp = entry.get("timestamp", "Unknown time")
        query = entry.get("query", "")
        result = entry.get("result_summary", "")
        
        print(f"{i}. [{timestamp}] {query}")
        print(f"   Result: {result}")
        print("-------------------------------------------")

def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    try:
        # Initialize the engine
        engine = PriceQueryEngine(
            api_key=args.api_key,
            anthropic_api_key=args.anthropic_api_key,
            verbose=args.verbose
        )
        
        # Handle special modes
        if args.clear_memory:
            engine.clear_memory()
            return
            
        if args.clear_history:
            engine.clear_history()
            return
            
        if args.history:
            display_history(engine)
            return
            
        if args.interactive:
            run_interactive_mode(engine)
            return
            
        # Process a single query if provided
        if args.query:
            result = engine.query(args.query)
            print(result["response"])
            return
            
        # If no query and no special mode, show help
        parser.print_help()
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
