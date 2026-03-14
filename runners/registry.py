"""Runner registry — maps model names to runner classes."""

RUNNERS = {}


def register(name):
    def decorator(cls):
        RUNNERS[name] = cls
        return cls
    return decorator


def get_runner(name):
    if name not in RUNNERS:
        raise ValueError(f"Unknown runner: {name}. Available: {list(RUNNERS.keys())}")
    return RUNNERS[name]()


def list_runners():
    return list(RUNNERS.keys())
