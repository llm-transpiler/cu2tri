import os
import dotenv

dotenv.load_dotenv()

from llm.providers import get_config_manager, get_provider_manager, PlatformConfig

# 配置管理
config_manager = get_config_manager()
config_manager.set_config("openrouter", PlatformConfig(
    platform_type="openrouter",
    api_key=os.getenv("OPENROUTER_API_KEY")
))
config_manager.save_config()

# 提供商管理
provider_manager = get_provider_manager()
provider_manager.load_from_config_manager()

async def main():
    # 初始化并使用
    await provider_manager.initialize_provider("openrouter")
    provider = provider_manager.get_initialized_provider("openrouter")
    print(await provider.list_models())

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())