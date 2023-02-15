# import libraries
import argparse
import json
import torch

from torchvision import transforms, datasets, models
from workspace_utils import *
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from train import FlowerClassifier
import matplotlib.pyplot as plt

# Define functions

# function to get inputs from the terminal
def get_predict_inputs():
    # Create the ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Predicts according to the trained model')

    # Add arguments
    parser.add_argument('img_dir', type=str, help='Image path')
    parser.add_argument('checkpoint', type=str,
                        help='saved checkpoint directory')

    parser.add_argument('--top_k', type=int, default=5,
                        help="no of most propable predictions")

    parser.add_argument('-cn', '--category_names',
                        type=str, default='cat_to_name.json')
    parser.add_argument('--gpu', default='cuda', type=str,help="Choose the CPU or GPU for computation")

    # Parse the arguments
    args = parser.parse_args()

    return args

# Choosing the model architecture
model_arch = {'densenet201': models.densenet201(pretrained=True),
              'densenet161': models.densenet161(pretrained=True)}

# function that loads a checkpoint and rebuilds the model
def load_checkpoint(filepath, idx):
    checkpoint = torch.load(filepath)

    architecture = checkpoint['arch']
    input_size = checkpoint['input_size']
    output_size = checkpoint['output_size']
    hidden_layers = checkpoint['hidden_layers']
    state_dict = checkpoint['model_data'].iloc[idx, 4]
    class_to_idx = checkpoint['class_to_idx']

    model = model_arch[architecture]
    classifier = FlowerClassifier(input_size, output_size, hidden_layers)

    model.classifier = classifier
    model.load_state_dict(state_dict)
    model.class_to_idx = class_to_idx

    return model

# function to process images to be used for prediction
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # TODO: Process a PIL image for use in a PyTorch model
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
            0.229, 0.224, 0.225])
    ])
    return preprocess(image)

# function to view the image
def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    return ax

def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    with Image.open(image_path) as im:
        image = process_image(im)
    image.unsqueeze_(0)
    model.eval()
    class_to_idx = model.class_to_idx
    idx_to_class = {idx : class_ for class_, idx in model.class_to_idx.items()}
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        probs, idxs = ps.topk(topk)
    idxs = idxs[0].tolist()
    classes = [idx_to_class[idx] for idx in idxs]
        
    # print('Probabilities: {}\nClasses: {}'.format(probs, classes))
    return probs, classes

# Main program
if __name__ == "__main__":
    inputs = get_predict_inputs()

    img_path = inputs.img_dir
    checkpoint = inputs.checkpoint
    top_k = inputs.top_k
    category_names = inputs.category_names
    processor = inputs.gpu

    # Use GPU if it's available
    if processor == 'gpu' and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    device = torch.device(device)

    # Label mapping
    with open(category_names,'r') as f:
        cat_to_name = json.load(f)

    # Load saved model
    imported_model = load_checkpoint(checkpoint, idx=3)

    probs, classes = predict(img_path, imported_model, top_k)
    names = [cat_to_name[class_] for class_ in classes]
    print(f"Most propable class is {names[0]} with a propability of {int(probs[0])}")

    print(f"Other propable classes are :\n")
    for i, val in enumerate(names[1:]):
        print(f"\t{names[i]} : propability = {int(probs[i])}")

    print("\nDone prediction!")

