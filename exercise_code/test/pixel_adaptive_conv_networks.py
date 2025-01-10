import torch
from torch.nn.modules.utils import _pair

from .base_tests import (
    UnitTest,
    MethodTest,
    CompositeTest,
    ClassTest,
    test_results_to_score,
)
from ..data.obj_segmentation.pixel_adaptive_conv_networks import packernel2d, pacconv2d


class KernelShapeTest(UnitTest):
    def __init__(self) -> None:
        self.batch_size = 1
        self.num_channels = 3
        self.input_size = (48, 64)
        self.kernel_size = 3
        self.center_idx = self.kernel_size * self.kernel_size // 2
        self.stride = 1
        self.padding = 1
        self.output_padding = 0
        self.dilation = 1
        self.kernel = None
        self.output_size = (
            self.batch_size,
            1,
            self.kernel_size,
            self.kernel_size,
        ) + tuple(
            [
                ((i + 2 * p - d * (k - 1) - 1) // s + 1)
                for (i, k, d, p, s) in zip(
                    self.input_size,
                    _pair(self.kernel_size),
                    _pair(self.dilation),
                    _pair(self.padding),
                    _pair(self.stride),
                )
            ]
        )
        self.kernel = None

    def test(self):
        self.kernel = packernel2d(
            torch.rand(
                self.batch_size,
                self.num_channels,
                self.input_size[0],
                self.input_size[1],
            ),
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        )
        return self.kernel.shape == self.output_size

    def define_success_message(self):
        return f"Congratulations: The kernel shape is correct."

    def define_failure_message(self):
        return f"The output shape of the kernel is incorrect. Expected: {self.output_size}, got: {self.kernel.shape}."


class CenterKernelWeightTest(UnitTest):
    def __init__(self) -> None:
        self.batch_size = 1
        self.num_channels = 3
        self.input_size = (48, 64)
        self.kernel_size = 3
        self.center_idx = self.kernel_size * self.kernel_size // 2
        self.stride = 1
        self.padding = 1
        self.output_padding = 0
        self.dilation = 1
        self.kernel = None

    def test(self):
        self.kernel = packernel2d(
            torch.rand(
                self.batch_size,
                self.num_channels,
                self.input_size[0],
                self.input_size[1],
            ),
            self.kernel_size,
            self.stride,
            self.padding,
            self.output_padding,
            self.dilation,
        ).flatten(
            2, 3
        )  # flatten again for test purposes
        return torch.all(
            torch.isclose(
                self.kernel[:, :, self.center_idx, :, :],
                torch.ones_like(self.kernel[:, :, self.center_idx, :, :]),
            )
        )

    def define_success_message(self):
        return f"Congratulations: The center kernel weight is correct."

    def define_failure_message(self):
        return f"The center kernel weight is incorrect. Expected it to be all one, got: {self.kernel[:, :, self.center_idx, :, :]}."


class KernelComputationTest(UnitTest):
    def __init__(self):
        self.batch_size = 4
        self.num_channels = 3
        self.H = 48
        self.W = 64

        self.kernel_size = 3
        self.stride = 1
        self.padding = 1
        self.output_padding = 0
        self.dilation = 1
        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.input = torch.load("exercise_code/test/kernel_input.pth")
        self.output = torch.load("exercise_code/test/kernel_output.pth")

    def test(self):
        return torch.all(
            torch.isclose(
                self.output,
                packernel2d(
                    self.input,
                    self.kernel_size,
                    self.stride,
                    self.padding,
                    self.output_padding,
                    self.dilation,
                ),
                1.0e-5,
            )
        )

    def define_success_message(self):
        return f"Congratulations: The kernel computation is correct."

    def define_failure_message(self):
        return f"Please rethink about the computation of the kernel."


class PACComputationTest(UnitTest):
    def __init__(self):
        self.batch_size = 4
        self.num_channels = 32
        self.H = 48
        self.W = 64

        self.kernel_size = _pair(3)
        self.stride = 1
        self.padding = 1
        self.output_padding = 0
        self.dilation = 1

        ########################################################################
        # TODO:                                                                #
        # Nothing to do here                                                   #
        ########################################################################

        pass

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        self.input = torch.load("exercise_code/test/conv_input.pth")
        self.kernel = torch.load("exercise_code/test/conv_kernel.pth")
        self.weights = torch.load("exercise_code/test/conv_weights.pth")
        self.bias = torch.load("exercise_code/test/conv_bias.pth")
        self.output = torch.load("exercise_code/test/conv_output.pth")

    def test(self):
        return torch.all(
            torch.isclose(
                self.output,
                pacconv2d(
                    self.input,
                    self.kernel,
                    self.weights,
                    self.bias,
                    self.stride,
                    self.padding,
                    self.dilation,
                ),
                1.0e-5,
            )
        )

    def define_success_message(self):
        return f"Congratulations: The convolution computation is correct."

    def define_failure_message(self):
        return f"Please rethink about the computation of the convolution."


class PacKernel2DTest(MethodTest):
    def define_tests(self):
        return [
            KernelShapeTest(),
            CenterKernelWeightTest(),
            KernelComputationTest(),
        ]

    def define_method_name(self):
        return "packernel2d"


class PixAdaptiveConvNetsTest(CompositeTest):
    def define_tests(self):
        return [
            PacKernel2DTest(),
            PACComputationTest(),
        ]

    def define_method_name(self):
        return "pacconv2d"


def test_pix_adaptive_conv_nets():
    test = PixAdaptiveConvNetsTest()
    return test_results_to_score(test())
