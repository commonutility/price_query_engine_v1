"""
Core engine for the Price Query Engine.
"""
import os
import sys
import time
from typing import List, Dict, Any, Optional, Union, Callable

import json
from langchain.agents import initialize_agent, AgentType
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import LLMResult, HumanMessage, BaseMessage, SystemMessage

from price_query_engine.memory import PersistentConversationMemory, QueryHistoryMemory
from price_query_engine.tools import (
    asset_metrics_tool,
    market_candles_tool,
    reference_data_tool,
    exchange_metrics_tool,
    get_metadata_registry
)
from price_query_engine.utils import MetadataRegistry

class ThinkingCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler that shows 'thinking' indicators when the LLM is working."""
    
    def __init__(self):
        super().__init__()
        self.thinking_started = False
        self.llm_finished = False
    
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
        """Show a 'thinking' indicator when the LLM starts processing."""
        print("\nThinking", end="", flush=True)
        self.thinking_started = True
        self.llm_finished = False
    
    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Clear the 'thinking' indicator when the LLM finishes."""
        if self.thinking_started:
            print("\n", end="", flush=True)
        self.thinking_started = False
        self.llm_finished = True
    
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print dots while the LLM is thinking."""
        if self.thinking_started and not self.llm_finished:
            print(".", end="", flush=True)
            time.sleep(0.1)  # Slow down the dots to make them visible

class PriceQueryEngine:
    """Engine for processing natural language queries about cryptocurrency pricing data."""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        llm_model: str = "o1",
        memory_file: Optional[str] = None,
        history_file: Optional[str] = None,
        verbose: bool = True
    ):
        """Initialize the query engine."""
        # Set up API keys
        self.api_key = api_key or os.environ.get("COINMETRICS_API_KEY")
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        
        # Validate API keys
        if not self.api_key:
            print("Warning: No Coin Metrics API key provided. Some functionality may be limited.")
        
        if not self.openai_api_key:
            raise ValueError(
                "No OpenAI API key provided. Set the OPENAI_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        # Set up memory
        self.memory = PersistentConversationMemory(
            memory_file=memory_file,
            auto_save=True,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up query history
        self.query_history = QueryHistoryMemory(history_file=history_file)
        
        # Set up LLM with thinking indicators
        self.thinking_callback = ThinkingCallbackHandler()
        self.llm = ChatOpenAI(
            model=llm_model,
            openai_api_key=self.openai_api_key,
            streaming=True,
            callbacks=[self.thinking_callback]
        )
        
        # Set up tools
        self.tools = [
            asset_metrics_tool,
            market_candles_tool,
            reference_data_tool,
            exchange_metrics_tool
        ]
        
        # Initialize metadata registry and get context
        self.metadata_registry = get_metadata_registry()
        metadata_context = self.metadata_registry.get_metadata_context()
        
        # Create system message with metadata context
        system_message = """You are a cryptocurrency data assistant that provides accurate information about crypto prices and market data.
        
You have access to the Coin Metrics API to retrieve cryptocurrency data.
        
When users ask about cryptocurrency data, use the appropriate tool to fetch the data.
        
{metadata_context}
        
Always try to understand what the user is asking for, and use the most appropriate tool to get the data.
If the user asks about a cryptocurrency by name (like "Bitcoin"), convert it to the appropriate ticker symbol (like "btc") when using the tools.

IMPORTANT: All dates in API calls must be supplied in ISO 8601 format: "YYYY-MM-DDTHH:MM:SS" (e.g., "2023-04-03T00:00:00").
Do not use relative time formats like "1d" or "yesterday" - convert these to actual dates in ISO 8601 format.
"""
        system_message = system_message.format(metadata_context=metadata_context)
        
        # Set up agent with system message
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            # agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=verbose,
            memory=self.memory,
            handle_parsing_errors=True,
            system_message=system_message
        )
        
        # Store parameters
        self.verbose = verbose
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """
        Process a natural language query about cryptocurrency pricing data.
        
        Args:
            query_text: The natural language query string
            
        Returns:
            Dict containing the response and any relevant metadata
        """
        try:
            # Process the query with the agent
            # Use invoke instead of run as run is deprecated
            result = self.agent.invoke({"input": query_text})
            
            # Extract response text from the result
            if isinstance(result, dict) and "output" in result:
                response = result["output"]
            else:
                response = str(result)
            
            # Extract parameters if available (from agent's last action)
            parameters = self._extract_parameters_from_last_action()
            
            # Save to query history
            self.query_history.add_query(
                query=query_text,
                parameters=parameters,
                result=response
            )
            
            return {
                "query": query_text,
                "response": response,
                "parameters": parameters,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            print(error_msg)
            return {
                "query": query_text,
                "response": error_msg,
                "parameters": {},
                "success": False
            }
    
    def _extract_parameters_from_last_action(self) -> Dict[str, Any]:
        """Extract parameters from the agent's last action if available."""
        try:
            # This is an implementation detail that might change with LangChain versions
            if hasattr(self.agent, "agent") and hasattr(self.agent.agent, "last_tool_inputs"):
                last_input = self.agent.agent.last_tool_inputs
                if isinstance(last_input, str):
                    try:
                        return json.loads(last_input)
                    except:
                        pass
            return {}
        except:
            return {}
    
    def clear_memory(self) -> None:
        """Clear the conversation memory."""
        self.memory.clear()
        print("Conversation memory cleared.")
    
    def clear_history(self) -> None:
        """Clear the query history."""
        self.query_history.clear()
        print("Query history cleared.")
    
    def get_recent_queries(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n most recent queries."""
        return self.query_history.get_recent_queries(n)
    
    def search_queries(self, search_term: str) -> List[Dict[str, Any]]:
        """Search for queries containing the search term."""
        return self.query_history.search_queries(search_term)
