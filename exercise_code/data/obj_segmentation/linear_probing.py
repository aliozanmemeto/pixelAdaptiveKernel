import torch
import torch.nn as nn


class LinearProbingNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        ########################################################################
        # TODO:                                                                #
        # Define a ONE layer neural network with a 1x1 convolution and a       #
        # sigmoid non-linearity to do binary classification on an image        #
        # NOTE: the network receives a batch of feature maps of shape          #
        # B x H x W x input_dim and should output a binary classification of   #
        # shape B x H x W                                                      #
        ########################################################################


        self.conv = nn.Conv2d(in_channels=input_dim, out_channels=1, kernel_size=1)
        
        # Define the sigmoid non-linearity
        self.sigmoid = nn.Sigmoid()

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        # x is a batch of feature maps of shape B x H x W x feat_dim

        ########################################################################
        # TODO:                                                                #
        # Do the forward pass of you defined network                           #
        # prediction = ...                                                     #
        ########################################################################

        # Apply the convolutional layer
        x = self.conv(x)
        
        # Apply the sigmoid function
        prediction = self.sigmoid(x)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

        return prediction
