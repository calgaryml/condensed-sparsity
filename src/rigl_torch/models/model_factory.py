class ModelFactory(object):
    registered_models = {}
    # Nested dictionary in form {dataset: {model: Class}}

    @classmethod
    def register_model(cls, model: str, dataset: str) -> None:
        print(f"assigning {model} to factory")

        def wrapper(model_cls):
            cls.registered_models[dataset] = {model: model_cls}
            return model_cls

        return wrapper

    @classmethod
    def get_model(cls, model, dataset):
        return cls.registered_models[dataset][model]()
