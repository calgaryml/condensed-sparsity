from torchvision.models import mobilenet_v3_large, mobilenet_v3_small
from rigl_torch.models import ModelFactory

# NOTE: No dropout to avoid disconnected graph when sparse training


@ModelFactory.register_model_loader(model="mobilenet_large", dataset="imagenet")
def get_mobilenet_large(*args, **kwargs):
    return mobilenet_v3_large(*args, weights=None, dropout=0, **kwargs)


@ModelFactory.register_model_loader(model="mobilenet_small", dataset="imagenet")
def get_mobilenet_small(*args, **kwargs):
    return mobilenet_v3_small(*args, weights=None, dropout=0, **kwargs)


if __name__ == "__main__":
    model = ModelFactory.load_model("mobilenet_small", "imagenet")
    print(model)
