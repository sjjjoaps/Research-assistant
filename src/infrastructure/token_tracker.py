"""
Token 用量追踪模块
从 LangChain AIMessage 的 response_metadata 中提取 token 用量，估算费用并格式化输出。
"""
from dataclasses import dataclass
from typing import Any


_DEFAULT_PRICE_PER_1K_TOKENS = 0.0005
_PRICE_PER_1K_TOKENS: dict[str, float] = {
    "qwen3.5-flash": 0.0005,
}


@dataclass
class TokenUsage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model_name: str
    estimated_cost_cny: float

    @classmethod
    def from_langchain_message(cls, message: Any, model_name: str) -> "TokenUsage":
        response_metadata = getattr(message, "response_metadata", {}) or {}
        token_usage = response_metadata.get("token_usage", {}) or {}

        prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
        total_tokens = int(token_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0)

        price_per_1k = _PRICE_PER_1K_TOKENS.get(model_name, _DEFAULT_PRICE_PER_1K_TOKENS)
        estimated_cost_cny = total_tokens / 1000 * price_per_1k

        return cls(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            model_name=model_name,
            estimated_cost_cny=estimated_cost_cny,
        )

    def display(self) -> str:
        return (
            f"Token: 提示词 {self.prompt_tokens}  生成 {self.completion_tokens}  "
            f"合计 {self.total_tokens}  估算费用 ¥{self.estimated_cost_cny:.5f}"
        )
