from PIL import Image
import torchvision.transforms as transforms
import torch
import os
import numpy as np

# import model from model.py, by name
from model import DogBreedClassifier

# default content type is numpy array
NP_CONTENT_TYPE = 'application/x-image'

# Reference: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model

# Provided model load function
def model_fn(model_dir):
    '''
    Loads the PyTorch model from the `model_dir` directory.

    :param: model_dir = SageMaker's model directory
    Reference:
        https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html?highlight=model_fn#load-a-model
    '''
    print("Loading model...")
    # Determine the device and construct the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogBreedClassifier().get_model()

    # Load the stored model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    return model.to(device).eval()


# Provided input data loading
def input_fn(serialized_input_data, content_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Deserializing the input data.')
    if content_type == NP_CONTENT_TYPE:
        image = Image.open(serialized_input_data).convert(mode='RGB')
        IMAGE_SIZE = 224
        # preprocess the image using transform
        prediction_transform = transforms.Compose([
                                            transforms.Resize(256),
                                            transforms.CenterCrop(IMAGE_SIZE),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225] )
                                            ])
        image_tensor = prediction_transform(image).unsqueeze(0)
        return image_tensor.to(device)
    raise Exception('Requested unsupported ContentType in content_type: ' + content_type)


# Provided output data handling
def output_fn(prediction, content_type):
    print('Serializing the generated output.')
    if content_type == 'json': # TODO
        # eturn a json string dictionary
    raise Exception('Requested unsupported ContentType in Accept: ' + accept)


# Provided predict function TODO
def predict_fn(input_data, model):
    '''load the image and return the predicted breed

    :input_data: return value from input_fn of type torch.Tensor.
    :model: loaded model.
    :return: return string of class name of dog breed. The return value should be of the correct type to be 
             passed as the first argument to output_fn.
    '''
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input = input_data.to(device)
        model.eval()                                # Set to eval mode for inference
        output = model(input)
        idx = torch.argmax(output)
        return class_names[idx]