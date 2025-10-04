"""
Tools for the ReAct Agent

Provides web search functionality using Tavily as a LangChain tool.
"""

import os
from typing import Optional
from langchain_core.tools import tool


@tool
def web_search(query: str) -> str:
    """Search the internet for current information.

    Use this when you need up-to-date information, news, or facts that you don't know.

    Args:
        query: The search query to look up on the web

    Returns:
        Search results as a formatted string
    """
    try:
        import requests

        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            return "Web search unavailable: TAVILY_API_KEY not set. Please configure your API key to use web search."

        response = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": 3
            },
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])

            if not results:
                return f"No results found for: {query}"

            # Format results concisely
            formatted = f"Search results for '{query}':\n\n"
            for i, result in enumerate(results[:3], 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')[:300]
                url = result.get('url', '')
                formatted += f"{i}. {title}\n   {content}...\n   Source: {url}\n\n"

            return formatted
        else:
            return f"Search failed with status {response.status_code}"

    except Exception as e:
        return f"Search error: {str(e)}"

@tool
def calculator(expression: str) -> str:
    """Evaluate mathematical expressions."""
    # Implementation
    return str(eval(expression))

def get_tools():
    """Get all available tools for the agent"""
    return [web_search, calculator]
