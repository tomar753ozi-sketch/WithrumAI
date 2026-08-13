"""
Whitrum Web Search Module
Enables the model to fetch information from the internet.
Founder: Oguzhan (Dr0xy-Drawn)
Copyright 2026 Whitrum AI.
"""

import requests
import json
from typing import Optional


class WhitrumWebSearch:
    """Web search capability for Whitrum AI."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.search_url = "https://api.search.brave.com/res/v1/web/search"
        self.headers = {
            "Accept": "application/json",
            "Accept-Encoding": "gzip",
            "X-Subscription-Token": api_key,
        } if api_key else {}

    def search(self, query: str, count: int = 5) -> list:
        """Search the web and return results."""
        if not self.api_key:
            return self._fallback_search(query)

        try:
            params = {"q": query, "count": count}
            response = requests.get(self.search_url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results = []
            for item in data.get("web", {}).get("results", []):
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("description", ""),
                })
            return results
        except Exception as e:
            return [{"error": str(e)}]

    def _fallback_search(self, query: str) -> list:
        """Fallback search using DuckDuckGo."""
        try:
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url, timeout=10)
            data = response.json()
            results = []
            if data.get("AbstractText"):
                results.append({
                    "title": data.get("Heading", ""),
                    "url": data.get("AbstractURL", ""),
                    "snippet": data.get("AbstractText", ""),
                })
            for topic in data.get("RelatedTopics", [])[:5]:
                if isinstance(topic, dict) and "Text" in topic:
                    results.append({
                        "title": topic.get("Text", "")[:50],
                        "url": topic.get("FirstURL", ""),
                        "snippet": topic.get("Text", ""),
                    })
            return results if results else [{"info": "No results found. Add API key for better search."}]
        except Exception as e:
            return [{"error": str(e)}]

    def get_context(self, query: str, max_chars: int = 2000) -> str:
        """Get search results as context string for the model."""
        results = self.search(query)
        context_parts = []
        total = 0
        for r in results:
            text = f"{r.get('title', '')}: {r.get('snippet', '')}"
            if total + len(text) > max_chars:
                break
            context_parts.append(text)
            total += len(text)
        return "\n".join(context_parts) if context_parts else "No relevant information found."


def create_search_tool(api_key: Optional[str] = None) -> WhitrumWebSearch:
    """Factory function to create web search tool."""
    return WhitrumWebSearch(api_key=api_key)
