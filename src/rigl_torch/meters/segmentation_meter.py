import torch


class SegmentationMeter(object):
    def __init__(self):
        self._max_bbox_mAP = -torch.inf
        self._bbox_mAP = -torch.inf
        self.bbox_mAPs = []
        self._max_mask_mAP = -torch.inf
        self._mask_mAP = -torch.inf
        self.mask_mAPs = []

    @property
    def mask_mAP(self) -> float:
        return self._mask_mAP

    @mask_mAP.setter
    def mask_mAP(self, mask_mAP: float) -> None:
        self._mask_mAP = mask_mAP
        self.mask_mAPs.append(self._mask_mAP)
        if self._mask_mAP >= self._max_mask_mAP:
            self._max_mask_mAP = self._mask_mAP

    @property
    def max_mask_mAP(self) -> float:
        return self._max_mask_mAP

    @property
    def bbox_mAP(self) -> float:
        return self._bbox_mAP

    @bbox_mAP.setter
    def bbox_mAP(self, bbox_mAP: float) -> None:
        self._bbox_mAP = bbox_mAP
        self.bbox_mAPs.append(self._bbox_mAP)
        if self._bbox_mAP >= self._max_bbox_mAP:
            self._max_bbox_mAP = self._bbox_mAP

    @property
    def max_bbox_mAP(self) -> float:
        return self._max_bbox_mAP


if __name__ == "__main__":
    meter = SegmentationMeter()
    meter.mask_mAP = 0.1
    print(meter.max_mask_mAP)
    meter.mask_mAP = 0.2
    print(meter.max_mask_mAP)
    meter.mask_mAP = 0.1
    print(meter.max_mask_mAP)
    print(meter.mask_mAP)
    meter.mask_mAP = 0.3
    print(meter.max_mask_mAP)
    print(meter.mask_mAP)
    print(meter.mask_mAPs)
