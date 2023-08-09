from torchvision.models.detection import maskrcnn_resnet50_fpn

from rigl_torch.models import ModelFactory


@ModelFactory.register_model_loader(model="maskrcnn", dataset="coco")
def get_maskrcnn(*args, **kwargs):
    return maskrcnn_resnet50_fpn(
        weights=None, weights_backbone=None, trainable_backbone_layers=5
    )


if __name__ == "__main__":
    model = get_maskrcnn()
    print(model)
