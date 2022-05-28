
import base64
import torch
import os
import json
import io
import torchvision.transforms as transforms
import numpy as np
from base64 import b64decode
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
        serialized_input_data (obj): the request data stream received
        content_type (str): data type of serialized input data

    Returns:
        returns a Torch.tensor, which is the first argument of predict_fn.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Deserializing the input data.')
    if content_type == 'image/jpeg':
        # create a binary stream using an in-memory bytes buffer and return the BytesIO object
        image_data = io.BytesIO(b64decode(serialized_input_data))

        image = Image.open(image_data).convert(mode='RGB')
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
        image_data.close()                 # free the memory buffer
        return image_tensor.to(device)
    raise Exception('Requested unsupported ContentType: ' + content_type)


# Provided predict function
def predict_fn(input_data, model):
    '''Make prediction of the dog breed given input data

    Args:
        input_data (Torch.Tensor): output from input_fn as input to the model
        model : PyTorch model loaded in memory by model_fn (output from model_fn)
    
    Returns:
        returns the class name of dog breed as a string, which is the first argument to output_fn.
    '''
    class_names = ['Affenpinscher', 'Afghan hound', 'Airedale terrier', 'Akita', 'Alaskan malamute', 'American eskimo dog', 'American foxhound', 'American staffordshire terrier', 'American water spaniel', 'Anatolian shepherd dog', 'Australian cattle dog', 'Australian shepherd', 'Australian terrier', 'Basenji', 'Basset hound', 'Beagle', 'Bearded collie', 'Beauceron', 'Bedlington terrier', 'Belgian malinois', 'Belgian sheepdog', 'Belgian tervuren', 'Bernese mountain dog', 'Bichon frise', 'Black and tan coonhound', 'Black russian terrier', 'Bloodhound', 'Bluetick coonhound', 'Border collie', 'Border terrier', 'Borzoi', 'Boston terrier', 'Bouvier des flandres', 'Boxer', 'Boykin spaniel', 'Briard', 'Brittany', 'Brussels griffon', 'Bull terrier', 'Bulldog', 'Bullmastiff', 'Cairn terrier', 'Canaan dog', 'Cane corso', 'Cardigan welsh corgi', 'Cavalier king charles spaniel', 'Chesapeake bay retriever', 'Chihuahua', 'Chinese crested', 'Chinese shar-pei', 'Chow chow', 'Clumber spaniel', 'Cocker spaniel', 'Collie', 'Curly-coated retriever', 'Dachshund', 'Dalmatian', 'Dandie dinmont terrier', 'Doberman pinscher', 'Dogue de bordeaux', 'English cocker spaniel', 'English setter', 'English springer spaniel', 'English toy spaniel', 'Entlebucher mountain dog', 'Field spaniel', 'Finnish spitz', 'Flat-coated retriever', 'French bulldog', 'German pinscher', 'German shepherd dog', 'German shorthaired pointer', 'German wirehaired pointer', 'Giant schnauzer', 'Glen of imaal terrier', 'Golden retriever', 'Gordon setter', 'Great dane', 'Great pyrenees', 'Greater swiss mountain dog', 'Greyhound', 'Havanese', 'Ibizan hound', 'Icelandic sheepdog', 'Irish red and white setter', 'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Irish wolfhound', 'Italian greyhound', 'Japanese chin', 'Keeshond', 'Kerry blue terrier', 'Komondor', 'Kuvasz', 'Labrador retriever', 'Lakeland terrier', 'Leonberger', 'Lhasa apso', 'Lowchen', 'Maltese', 'Manchester terrier', 'Mastiff', 'Miniature schnauzer', 'Neapolitan mastiff', 'Newfoundland', 'Norfolk terrier', 'Norwegian buhund', 'Norwegian elkhound', 'Norwegian lundehund', 'Norwich terrier', 'Nova scotia duck tolling retriever', 'Old english sheepdog', 'Otterhound', 'Papillon', 'Parson russell terrier', 'Pekingese', 'Pembroke welsh corgi', 'Petit basset griffon vendeen', 'Pharaoh hound', 'Plott', 'Pointer', 'Pomeranian', 'Poodle', 'Portuguese water dog', 'Saint bernard', 'Silky terrier', 'Smooth fox terrier', 'Tibetan mastiff', 'Welsh springer spaniel', 'Wirehaired pointing griffon', 'Xoloitzcuintli', 'Yorkshire terrier']
    with torch.no_grad():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        input = input_data.to(device)
        model.eval() # Set to eval mode for inference
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
