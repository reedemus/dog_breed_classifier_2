import torchvision.transforms as transforms
import torch
import os
import numpy as np
from PIL import Image

# import model from model.py, by name
from model import DogBreedClassifier

# Reference: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model

# Provided model load function
def model_fn(model_dir):
    '''Loads the PyTorch model

    Args:
        model_dir (str): folder path from SageMaker's model directory
    
    Returns:
        a pyTorch model
    
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
    '''Input function

    Args:
        serialized_input_data (any): raw input data received
        content_type (str): data type of serialized input data e.g. npy, x-image, json, etc.

    Returns:
        input data converted to Torch.tensor, which is the first argument of predict_fn.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Deserializing the input data.')
    if content_type == 'application/x-image':
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
    raise Exception('Requested unsupported ContentType: ' + content_type)


# Provided predict function TODO
def predict_fn(input_data, model):
    '''Make prediction of the dog breed given input data

    Args:
        input_data (Torch.Tensor): output from input_fn as input to the model
        model : PyTorch model loaded in memory by model_fn (output from model_fn)
    
    Returns:
        returns the class name of dog breed as a string, which is the first argument to output_fn.
    '''
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input = input_data.to(device)
        model.eval()                                # Set to eval mode for inference
        output = model(input)
        idx = torch.argmax(output)
        return class_names[idx]


# Provided output data handling
def output_fn(prediction, content_type):
    '''Custom output function to be returned from prediction

    Args:
        prediction (str): prediction result from predict_fn, which is class name of dog breed
        content_type (str): type which the output data needs to be serialized

    Returns:
        output data serialized
    '''
    print('Serializing the generated output.')
    if type(prediction) == str and content_type == 'application/json':
        # returns a json string dict
        return { 'result': prediction }
    raise Exception('Requested unsupported ContentType: ' + content_type)
