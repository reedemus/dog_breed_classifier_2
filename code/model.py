import torchvision.models as models
import torch.nn as nn

# Using feature extraction approach
# =================================
# Freeze the weights for all of the network except the final fully connected(FC) layer.
# This last FC layer is replaced with a new one with random weights and only this layer is trained.
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

class DogBreedClassifier():
    '''
    Pretrained Resnet model with the output features in the last layer set to 133 nodes, which is the number of dog breed classes
    '''
    def __init__(self):
        # ResNet 152-layer model
        self.model_transfer = models.resnet152(pretrained=True)

        # Freeze the pre-trained weights,biases of all layers at first so it doesn't get updated during re-training
        for param in self.model_transfer.parameters():
            param.requires_grad = False

        # Get the number of input features in the last FC layer
        # Reinitialize output features to number of dog breed classes
        self.input_features = self.model_transfer.fc.in_features
        DOG_BREEDS_NUM = 133
        self.model_transfer.fc = nn.Linear(self.input_features, DOG_BREEDS_NUM)

        print("ResNet-152 last fc layer:", models.resnet152().fc)
        print("Our fc layer:", self.model_transfer.fc)

        return self.model_transfer
