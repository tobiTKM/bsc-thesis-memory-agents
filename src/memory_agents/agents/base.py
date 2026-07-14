from ..llm.client import LLMClient, LLMResponse
from ..envs.sparc import parse_action

class BaselineAgent:
    def __init__(self, client, max_format_retries: int = 5):
        self.client = client
        self.max_format_retries = max_format_retries
        self.messages: list[dict] = []

    def reset(self, system_prompt: str) -> None:
        """Start a fresh episode with the given prompt."""
        self.messages = [{"role": "system", "content": system_prompt}]

    async def act(self, user_message: str):
        self.messages.append({"role": "user", "content": user_message})

        action, response = None, None
        for _ in range(self.max_format_retries):
            response = await self.client.chat(self.messages)
            action = parse_action(response.text)
            if action is not None:
                break

        self._trim()
        return action, response

    def _trim(self) -> None:
        """Keep only the system prompt (memoryless: the env re-sends the full
        grid state every step, so the agent needs no history)."""
        self.messages = self.messages[:1]
