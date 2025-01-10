import torch
from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)

from ..data.obj_segmentation import LinearProbingNet


class OutputShapeTest(UnitTest):
    def __init__(self):
        self.H = 480
        self.W = 640
        self.channels = 32
        self.batch_size = 3
        self.network = LinearProbingNet(self.channels)
        self.input = torch.rand(self.batch_size, self.channels, self.H, self.W)
        self.output = None

    def test(self):
        self.output = self.network(self.input)
        return self.output.shape == (self.batch_size, 1, self.H, self.W)

    def define_success_message(self):
        return f"Congratulations: The output shape of the network is correct"

    def define_failure_message(self):
        return f"The output shape of the network is incorrect (expected {(self.batch_size, 1, self.H, self.W)}, got {self.output.shape})."


class OutputRangeTest(UnitTest):
    def __init__(self):
        self.H = 480
        self.W = 640
        self.channels = 32
        self.batch_size = 3
        self.network = LinearProbingNet(self.channels)
        self.input = torch.rand(self.batch_size, self.channels, self.H, self.W)
        self.output = None
        self.min = None
        self.max = None

    def test(self):
        self.output = self.network(self.input)
        self.min = torch.min(self.output)
        self.max = torch.max(self.output)

        return self.min >= 0 and self.max <= 1.0

    def define_success_message(self):
        return (
            f"Congratulations: The output returned gives a probability between 0 and 1"
        )

    def define_failure_message(self):
        return f"The output range of the network is incorrect (got a minimum of {self.min} and a maximum of {self.max}, expected only values between 0 and 1)."


class LinearProbingTest(ClassTest):
    def define_tests(self):
        return [
            OutputShapeTest(),
            OutputRangeTest(),
        ]

    def define_class_name(self):
        return "LinearProbingNet"


def test_linear_probing():
    test = LinearProbingTest()
    return test_results_to_score(test())
