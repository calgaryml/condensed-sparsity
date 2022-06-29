from torch.optim.lr_scheduler import _LRScheduler


class StepLrWithLinearWarmUp(_LRScheduler):
    def __init__(self):
        super.__init__()
        # TODO for imagenet
