from typing import Callable
import torch.nn as nn
import logging


class ModelFactory(object):
    registered_models = {}
    # Nested dictionary in form {dataset: {model: Callable}}
    logging.basicConfig(level=logging.INFO)
    # TODO -> Check above for interaction with hydra-core
    __logger = logging.getLogger(__file__)

    @classmethod
    def register_model_loader(cls, model: str, dataset: str) -> Callable:
        cls.__logger.info(
            f"Registering {model} for {dataset} dataset to ModelFactory..."
        )

        def wrapper(model_loader: Callable) -> Callable:
            if dataset not in cls.registered_models:
                cls.registered_models[dataset] = {}
            cls.registered_models[dataset][model] = model_loader
            return model_loader

        return wrapper

    @classmethod
    def load_model(
        cls, model, dataset, state_dict=None, *args, **kwargs
    ) -> nn.Module:
        cls.__logger.info(
            f"Loading model {model}/{dataset} using "
            f"{cls.registered_models[dataset][model]} with args: {args} and "
            f"kwargs: {kwargs}"
        )
        model = cls.registered_models[dataset][model](*args, **kwargs)
        if state_dict is not None:
            model.load_state_dict(state_dict)
        return model
