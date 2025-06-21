import os
import dotenv

dotenv.load_dotenv()

from llm.providers import (
    PlatformConfig, PlatformType, 
    ProviderFactory
)

# 1. 创建配置
config = PlatformConfig(
    platform_type=PlatformType.OPENROUTER,
    api_key=os.getenv("OPENROUTER_API_KEY"),
    preferred_models=["gpt-4o", "gpt-4o-mini"],
    temperature=0.7,
    max_tokens=4096
)

# from llm.providers.types import PLATFORM_REGISTRY
# print(PLATFORM_REGISTRY[PlatformType.OPENROUTER])
# exit()
# 2. 创建提供商
provider = ProviderFactory.create_provider(config)

async def main():
    # 3. 初始化并使用
    await provider.initialize()

    # 4. 发送请求
    from llm.providers import ChatMessage, ChatRequest
    request = ChatRequest(
        messages=[ChatMessage(role="user", content="Hello!")],
        model="gpt-4o-mini"
    )

    response = await provider.chat(request)
    print(response.content)

    # 5. 流式请求
    async for chunk in provider.stream_chat(request):
        print(chunk.content, end="")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())