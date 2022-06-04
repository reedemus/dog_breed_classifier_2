from torchvision.models.resnet import ResNet, Bottleneck

DOG_BREEDS_NUM = 133 # our desired output classes

class DogBreedClassifier(ResNet):
    '''Pretrained Resnet-152 model with the number of classes(i.e. output features in the last layer) set to 133 dog breed classes'''
    def __init__(self):
        # reference: https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py#L788
        super().__init__(Bottleneck, [3, 8, 36, 3], num_classes=DOG_BREEDS_NUM)
