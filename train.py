# import libraries
import argparse
import json
import time
import torch
import torch
from torchvision import transforms, datasets, models
from workspace_utils import *
# from flower_classifier import FlowerClassifier
from PIL import Image
from torch import nn, optim
import torch.nn.functional as F
import numpy as np
import pandas as pd

# define functions
# function to get inputs


def get_train_inputs():
    """
    returns:
        argparse object with argument inputs
    """

    # Create the ArgumentParser object
    parser = argparse.ArgumentParser(
        description='Train a classifier using a pretrained network')

    # Add arguments
    parser.add_argument('data_dir', type=str, help='dataset directory')
    parser.add_argument('--save_dir', type=str,
                        default='', help='checkpoint saving directory')

    # Choose architecture
    parser.add_argument('--arch', type=str, default='densenet201',
                        help="Pretrained model architecture")

    # Set hyperparameters
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.03)
    parser.add_argument('-e', '--epochs', type=int, default=10)
    parser.add_argument('-hu', '--hidden_units', type=int,
                        default=[1024, 512, 256])
    parser.add_argument('--gpu', default='cuda', type=str)

    # Parse the arguments
    args = parser.parse_args()

    return args

# function to load the datset


def load_dataset(data_dir):
    """
    parameters:
        data_dir : relative path to data directory

    returns:
        a tuple with traininig, validation and test loader
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    # Done: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                            transforms.RandomResizedCrop(224),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.RandomVerticalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])])

    test_valid_transforms = transforms.Compose([transforms.Resize(255),
                                                transforms.CenterCrop(224),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                                         [0.229, 0.224, 0.225])])
    # Done: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_valid_transforms)

    # Done: Using the image datasets and the trainforms, define the dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
    
    return test_data.class_to_idx, train_loader, valid_loader, test_loader

# function to train the model


def train(model, epochs, criterion, optimizer, train_loader, valid_loader, manual_seed=42):
    """
        Train the given model.


        Parameters
        ----------
        model : model for the classification problem

        epochs : no of complete iterations over the entire dataset

        criterion : loss function / cost function to see how much our model has been deviated from the real values
                examples :: Categorical Cross-Entropy Loss , Negative Log-Likelihood Loss

        optimizer : The algorithm that is used to update the parameters of the model
                examples :: Stochastic Gradient Descent (SGD) , Adam algorithm

        train_loader : loader for the training dataset

        valid_loader : loader fot the validation dataset


        Returns
        -------
        performance_list : a list with details of the model during each epoch

    """

    performance_list = []

    torch.manual_seed(manual_seed)

    start = time.time()

    with active_session():

        for e in range(1, epochs+1):

            train_total_loss = 0
            valid_total_loss = 0

            for images, labels in train_loader:

                # move images and labels data from GPU to CPU, back and forth.
                images = images.to(device)
                labels = labels.to(device)

                log_ps = model(images)
                loss = criterion(log_ps, labels)
                train_total_loss += loss.item()

                # To avoid accumulating gradients
                optimizer.zero_grad()

                # Back propagation
                loss.backward()
                optimizer.step()

            model.eval()

            accuracy = 0

            with torch.no_grad():
                for images, labels in valid_loader:

                    images = images.to(device)
                    labels = labels.to(device)

                    log_ps = model.forward(images)
                    loss = criterion(log_ps, labels)
                    valid_total_loss += loss.item()

                    top_p, top_class = log_ps.topk(1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += 100 *  torch.mean(equals.type(torch.FloatTensor)).item()

                accuracy = accuracy / len(valid_loader)
                train_loss = train_total_loss / len(train_loader)
                valid_loss = valid_total_loss / len(valid_loader)

                performance_list.append(
                    (e, train_loss, valid_loss, accuracy, model.state_dict()))

                end = time.time()
                epoch_duration = end - start
                print(f"Epoch: {e}, Train loss: {train_loss:.3f}, Valid loss: {valid_loss:.3f}, Accuracy: {accuracy:.3f}%, time duration per epoch: {epoch_duration:.3f}s")
            model.train()

    return performance_list


# define classifier
class FlowerClassifier(nn.Module):

    """
    Fully connected / Dense network to be used in the transfered model
    """

    def __init__(self, input_size, output_size, hidden_layers, dropout_p=0.3):
        """
        Parameters
        ----------

            input_size : no of units in the input layer (usually the pretrained classifier's 
            features_in value)

            output_size : no of units (no of classes that we have to classify the dataset)

            hidden_layers : a list with no of units in each hidden layer

            dropout_p : dropout probability (to avoid overfitting)        

        """

        super().__init__()

        self.hidden_layers = nn.ModuleList(
            [nn.Linear(input_size, hidden_layers[0])])

        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2)
                                  for h1, h2 in layer_sizes])

        self.output = nn.Linear(hidden_layers[-1], output_size)

        # add a dropout propability to avoid overfitting
        self.dropout = nn.Dropout(p=dropout_p)

    def forward(self, x):

        for each in self.hidden_layers:
            x = self.dropout(F.relu(each(x)))

        x = self.output(x)
        x = F.log_softmax(x, dim=1)

        return x


# function to train the model

# main program
# Label mapping
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

inputs = get_train_inputs()


data_dir = inputs.data_dir
save_dir = inputs.save_dir
architecture = inputs.arch
learning_rate = inputs.learning_rate
epochs = inputs.epochs
hidden_layers = inputs.hidden_units
gpu = inputs.gpu


class_to_idx, train_loader, valid_loader, test_loader = load_dataset(
    data_dir=data_dir)

output_layer_size = len(cat_to_name)
# print(output_layer_size)


# Use GPU if it's available
device = torch.device(gpu)
print(f'Using {device} for computation : ')
print(
    f"settings :\n\tmodel_architecutre : {architecture}\n\tlearning rate : {learning_rate}\n\tepochs : {epochs}\n\t")

# Choosing the model architecture
model_arch = {'densenet201': models.densenet201(pretrained=True),
              'densenet161': models.densenet161(pretrained=True)}
model = model_arch[architecture]

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

hidden_layer_units = hidden_layers
input_layer_size = model.classifier.in_features

# exit()
my_classifier = FlowerClassifier(
    input_layer_size, output_layer_size, hidden_layer_units, dropout_p=0.2)
model.classifier = my_classifier

model.to(device)

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()

print()
print("Started training the model : \n")
model_list = train(model=model, epochs=epochs, criterion=criterion,
                   optimizer=optimizer, train_loader=train_loader, valid_loader=valid_loader)

# Save the model
# dataframe to store performance of the model, so that we can change the hyperparameters
model_performance_architecture_df = pd.DataFrame(model_list)
model_performance_architecture_df.rename(
    columns={
        0: 'epoch',
        1: 'train_loss',
        2: 'valid_loss',
        3: 'valid_accuracy',
        4: 'state_dict'
    }, inplace=True)

model_performance_df = model_performance_architecture_df.iloc[:, :4]

# TODO: Save the checkpoint
checkpoint = {'arch': architecture,
              'input_size': input_layer_size,
              'output_size': output_layer_size,
              'hidden_layers': [each.out_features for each in model.classifier.hidden_layers],
              'model_data': pd.DataFrame(model_list),
              'class_to_idx': class_to_idx}

save_dir_path =  "checkpoint.pth"
torch.save(checkpoint, save_dir_path)
print(f"Checkpoint saved to {save_dir_path}")
