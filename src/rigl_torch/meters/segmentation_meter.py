import torch


class SegmentationMeter(object):
    def __init__(self):
        self._max_box_mAP = -torch.inf
        self._box_mAP = -torch.inf
        self.box_mAPs = []
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
    def box_mAP(self) -> float:
        return self._box_mAP

    @mask_mAP.setter
    def box_mAP(self, box_mAP: float) -> None:
        self._box_mAP = box_mAP
        self.box_mAPs.append(self._box_mAP)
        if self._box_mAP >= self._max_box_mAP:
            self._max_box_mAP = self._box_mAP

    @property
    def max_box_mAP(self) -> float:
        return self._max_box_mAP


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
