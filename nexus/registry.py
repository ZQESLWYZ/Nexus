from typing import Callable, TypeVar

T = TypeVar("T")
REGISTRY: dict[str, dict[str, T]] = {}


def register(group: str, name: str) -> Callable[[T], T]:
    def decorator(item: T) -> T:
        REGISTRY.setdefault(group, {})[name] = item
        return item

    return decorator


def get(group: str, name: str) -> T:
    return REGISTRY[group][name]
