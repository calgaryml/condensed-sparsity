import torch


class TrainingMeter(object):
    def __init__(self):
        self._max_accuracy = -torch.inf
        self._accuracy = -torch.inf
        self.accuracies = []

    @property
    def accuracy(self) -> float:
        return self._accuracy

    @accuracy.setter
    def accuracy(self, accuracy: float) -> None:
        self._accuracy = accuracy
        self.accuracies.append(self._accuracy)
        if self._accuracy >= self._max_accuracy:
            self._max_accuracy = self._accuracy

    @property
    def max_accuracy(self) -> float:
        return self._max_accuracy


if __name__ == "__main__":
    meter = TrainingMeter()
    meter.accuracy = 0.1
    print(meter.max_accuracy)
    meter.accuracy = 0.2
    print(meter.max_accuracy)
    meter.accuracy = 0.1
    print(meter.max_accuracy)
    print(meter.accuracy)
    meter.accuracy = 0.3
    print(meter.max_accuracy)
    print(meter.accuracy)
    print(meter.accuracies)
