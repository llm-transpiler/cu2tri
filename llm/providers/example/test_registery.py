from llm.providers import get_provider_registry, VENDOR_MODEL_REGISTRY
for vendor, models in VENDOR_MODEL_REGISTRY.items():
    print(type(vendor))