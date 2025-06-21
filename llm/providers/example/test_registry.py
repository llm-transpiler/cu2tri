from llm.providers.registry import get_provider_registry
from llm.providers.types import PlatformType
from llm.providers.factory import get_provider_manager

registry = get_provider_registry()

print(registry.get_platform_info(PlatformType.OPENROUTER))

factory = get_provider_manager()
provider = factory.get_provider(PlatformType.OPENROUTER)

print(provider.list_models_sync())