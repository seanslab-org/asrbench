"""Dataset registry — maps dataset names to loader classes."""

DATASETS = {}


def register_dataset(name):
    def decorator(cls):
        DATASETS[name] = cls
        return cls
    return decorator


def get_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASETS.keys())}")
    return DATASETS[name]()


def list_datasets():
    return list(DATASETS.keys())
