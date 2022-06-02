import torch.nn as nn
import torch.hub as Hub
from torchvision.models.resnet import ResNet, Bottleneck

# Using feature extraction approach
# =================================
# Freeze the weights for all of the network except the final fully connected(FC) layer.
# This last FC layer is replaced with a new one with random weights and only this layer is trained.
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html#initialize-and-reshape-the-networks

DOG_BREEDS_NUM = 133

class DogBreedClassifier(ResNet):
    '''Pretrained Resnet-152 model with the number of classes(i.e. output features in the last layer) set to 133 dog breed classes'''
    def __init__(self):
        # reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L788
        super().__init__(Bottleneck, [3, 8, 36, 3])

    @staticmethod
    def training_setup(model, pretrained=False):
        if pretrained:
            state_dict = Hub.load_state_dict_from_url('https://download.pytorch.org/models/resnet152-b121ed2d.pth')
            model.load_state_dict(state_dict)
            # Freeze the pre-trained weights,biases of all layers at first so it doesn't get updated during re-training
            for param in model.parameters():
                param.requires_grad = False
        
        # Get the number of input features in the last FC layer
        # Reinitialize output features to number of dog breed classes
        input_features = model.fc.in_features
        model.fc = nn.Linear(input_features, DOG_BREEDS_NUM)
        print("Customized output classes:", model.fc)
        return model
