from langchain.agents import create_agent

from src.llm_client import get_llm


model = get_llm()


def get_weather(city: str) -> str:
    """获取指定城市的天气。"""
    return f"{city}总是阳光明媚！"


agent = create_agent(
    model=model,
    tools=[get_weather],
    system_prompt="你是一个乐于助人的助手",
)

print(
    agent.invoke(
        {"messages": [{"role": "user", "content": "旧金山的天气怎么样"}]}
    )
)
