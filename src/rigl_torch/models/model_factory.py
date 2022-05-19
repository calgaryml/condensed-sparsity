from typing import Callable
from functools import partial
import torch.nn as nn


class ModelFactory(object):
    registered_models = {}
    # Nested dictionary in form {dataset: {model: Callable}}

    @classmethod
    def register_model(cls, model: str, dataset: str) -> Callable:
        print(f"assigning {model} to factory")

        def wrapper(model_loader: Callable) -> Callable:
            cls.registered_models[dataset] = {model: partial(model_loader)}
            return model_loader

        return wrapper

    @classmethod
    def get_model(cls, model, dataset, *args, **kwargs) -> nn.Module:
        return cls.registered_models[dataset][model](*args, **kwargs)
