"""
Whitrum AI Chat Interface with Web Search
Founder: Oguzhan (Dr0xy-Drawn)
Copyright 2026 Whitrum AI.
"""

from .web_search import WhitrumWebSearch


class WhitrumChat:
    """Interactive chat with web search capability."""

    SYSTEM_PROMPT = (
        "You are Whitrum AI, a helpful language model created by Oguzhan (Dr0xy-Drawn). "
        "You can search the web for current information when needed. "
        "Always be helpful, accurate, and cite your sources."
    )

    def __init__(self, model=None, tokenizer=None, api_key=None):
        self.model = model
        self.tokenizer = tokenizer
        self.web_search = WhitrumWebSearch(api_key=api_key)
        self.history = []

    def search_and_respond(self, user_input: str) -> str:
        """Search web and generate response."""
        context = self.web_search.get_context(user_input)
        prompt = f"Context: {context}\n\nUser: {user_input}\nWhitrum:"
        return f"Based on my search:\n\n{context}\n\nI hope this helps! - Whitrum AI"

    def chat(self, user_input: str) -> str:
        """Simple chat without model inference."""
        if any(kw in user_input.lower() for kw in ["search", "find", "look up", "what is", "who is", "when", "where", "how"]):
            return self.search_and_respond(user_input)
        return f"You said: {user_input}. I'm Whitrum AI by Oguzhan (Dr0xy-Drawn). Use 'search' to find info online."


def main():
    print("=" * 50)
    print("Whitrum AI Chat")
    print("Founder: Oguzhan (Dr0xy-Drawn)")
    print("Type 'quit' to exit, 'search <query>' to search web")
    print("=" * 50)

    chat = WhitrumChat()
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye! - Whitrum AI")
            break
        if not user_input:
            continue
        response = chat.chat(user_input)
        print(f"\nWhitrum: {response}")


if __name__ == "__main__":
    main()
