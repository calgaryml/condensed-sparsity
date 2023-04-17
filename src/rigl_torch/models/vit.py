from torchvision.models import vit_b_32
from rigl_torch.models import ModelFactory


@ModelFactory.register_model_loader(model="vit", dataset="imagenet")
def get_vit(*args, **kwargs):
    return vit_b_32(weights=None)


if __name__ == "__main__":
    vit_b = ModelFactory.load_model("vit", "imagenet")
    print(vit_b)
