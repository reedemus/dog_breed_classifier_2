import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# the following import is required for training to be robust to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# imports the model in model.py by name
from model import DogBreedClassifier

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
    model = DogBreedClassifier()

    # Load the stored model parameters
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode, could use no_grad
    return model.to(device).eval()


# model saving function
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html?highlight=model_fn#save-the-model
def save_model(model, model_dir):
    print("Saving the model...")
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        # save state dictionary
        torch.save(model.state_dict(), f)


def _get_data_loader(batch_size=32, training_dir='train', validation_dir='valid', testing_dir='test'):
    '''
    Create three separate data loaders for the training, validation, and test datasets
    
    :param: batch size = no of samples
    :param: training_dir = folder path of training dataset
    :param: validation_dir = folder path of validation dataset
    :param: testing_dir = folder path of test dataset
    :return: dictionary of the three dataloaders
    '''
    # Note: pretrained models from ImageNet requires input images to be shape (3 x h x w) with h,w >= 224 and 
    #       normalized to mean,std dev as per ImageNet (mean = [0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225])
    #
    # Reference: https://pytorch.org/vision/stable/models.html
    
    ## Specify appropriate transforms, and batch_sizes
    train_dir = training_dir
    valid_dir = validation_dir
    test_dir = testing_dir
    # ResNet input image size
    IMG_SIZE = 224
    # mean and std deviation of models trained on Imagenet dataset
    mean = [0.485, 0.456, 0.406]
    std_dev = [0.229, 0.224, 0.225]

    # Data augmentation to create a variety of test images so the model learn to generalize better.
    # Output is a tensor.
    preprocess_train = transforms.Compose([
                                        transforms.RandomResizedCrop(IMG_SIZE),
                                        transforms.RandomRotation(20),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize( mean, std_dev)
                                        ])

    # Data augmentation is not performed on validation and test datasets because the goal is not to create more data,
    # but to resize and crop the images to the same size as the input image.
    # Output is a tensor.
    preprocess_valid_test = transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(IMG_SIZE),
                                        transforms.ToTensor(),
                                        transforms.Normalize( mean, std_dev)
                                        ])

    train_dataset = datasets.ImageFolder(train_dir, transform=preprocess_train)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=preprocess_valid_test)
    test_dataset = datasets.ImageFolder(test_dir, transform=preprocess_valid_test)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    loaders_dict = { 'train':train_loader, 'valid':valid_loader, 'test':test_loader }
    return loaders_dict


def train(model, loaders, n_epochs, optimizer, criterion, use_cuda):
    """
    This is the training method that is called by the PyTorch training script. After
    training, the model with the best validation accuracy is saved.
    The parameters passed are as follows:
    model        - The PyTorch model that we wish to train.
    loaders      - The PyTorch DataLoader that should be used during training.
    n_epochs     - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    use_cuda     - uses gpu(True) or cpu(False).
    """
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    
    for epoch in range(1, n_epochs+1):
        # initialize variables to monitor training and validation loss
        train_loss = 0.0
        valid_loss = 0.0
        
        ###################
        # train the model #
        ###################
        model.train()
        for batch_idx, (data, target) in enumerate(loaders['train']):
            # move to GPU
            if use_cuda:
                data, target = data.cuda(), target.cuda()
            
            # zero the gradients accumulated from the previous backward propagation steps,
            # make prediction, calculate the training loss, perform backpropagation, 
            # and finally update model weights and biases.
            optimizer.zero_grad()
            output = model(data)
            loss = criterion( output, target)
            loss.backward()
            optimizer.step()
            
            # record the average training loss
            train_loss = train_loss + ((1 / (batch_idx + 1)) * (loss.data - train_loss))
        
        ######################    
        # validate the model #
        ######################
        # switch model to evalution mode and disable gradient calculations to reduce memory usage
        # and speed up computations since no backpropagation is needed in evaluation.
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(loaders['valid']):
                # move to GPU
                if use_cuda:
                    data, target = data.cuda(), target.cuda()

                # get prediction from our model, calculate the validation loss
                output = model(data)
                loss = criterion( output, target)

                ## update the average validation loss
                valid_loss = valid_loss + ((1 / (batch_idx + 1)) * (loss.data - valid_loss))

            # print training/validation statistics 
            print('Epoch {}: \tTraining Loss: {:.3f} \tValidation Loss: {:.3f}'.format(
                epoch, train_loss, valid_loss))

            # save the model if validation loss has decreased
            if valid_loss < valid_loss_min:
                valid_loss_min = valid_loss
                save_model(model, args.model_dir)


if __name__ == '__main__':
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()
        
    # Training Parameters, given
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                        help="learning rate (default: 0.01)")    
    
    # SageMaker parameters, like the directories for training data and saving models; set automatically
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--valid-dir', type=str, default=os.environ['SM_CHANNEL_VALID'])
    parser.add_argument('--test-dir', type=str, default=os.environ['SM_CHANNEL_TEST'])
    
    # args holds all passed-in arguments
    args = parser.parse_args()
    
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print("Using GPU.")
    else:
        print("Using CPU.")
    
    # Gets the train, validation and test data loaders
    data_loaders = _get_data_loader(args.batch_size, args.train_dir, args.valid_dir, args.test_dir)
    
    # Build the model
    model = DogBreedClassifier()
    if use_cuda:
        model.cuda()

    # Define an optimizer and loss function for training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Trains and save the model
    train(model, data_loaders, args.epochs, optimizer, criterion, use_cuda)
