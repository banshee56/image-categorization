from create_dataset import create_dataset
from cnn_categorization_base import cnn_categorization_base
from cnn_categorization_improved import cnn_categorization_improved
from train import train
from torch import random, save
from argparse import ArgumentParser
import matplotlib.pyplot as plt


from time import time
import torch
from torchvision import transforms

# seed the random number generator. Remove the line below if you want to try different initializations
random.manual_seed(0)


def cnn_categorization(model_type="base",
                       data_path="image_categorization_dataset.pt",
                       contrast_normalization=False, whiten=False):
    """
    Invokes the dataset creation, the model construction and training functions

    Arguments
    --------
    model_type: (string), the type of model to train. Use 'base' for the base model and 'improved for the improved model. Default: base
    data_path: (string), the path to the dataset. This argument will be passed to the dataset creation function
    contrast_normalization: (boolean), specifies whether or not to do contrast normalization
    whiten: (boolean), specifies whether or not to whiten the data.

    """
    # Do not change the output path
    # but you can uncomment the exp_dir if you do not want to save the model checkpoints
    output_path = "{}_image_categorization_dataset.pt".format(model_type)
    exp_dir = "./{}_models".format(model_type)

    train_ds, val_ds = create_dataset(data_path, output_path, contrast_normalization, whiten)

    # specify the network architecture and the training policy of the models under
    # the respective blocks
    if model_type == "base":
        # create netspec_opts
        layer_type =    ['conv', 'bn', 'relu', 'conv', 'bn', 'relu', 'conv',  'bn', 'relu', 'pool', 'conv']
        kernel_size =   [3,     0,      0,      3,      0,      0,      3,      0,      0,      8,      1]
        stride =        [1,     0,      0,      2,      0,      0,      2,      0,      0,      1,      1]
        num_filers =    [16,    16,     0,      32,     32,     0,      64,     64,     0,      0,      16]

        netspec_opts = {'kernel_size': kernel_size,
                        'num_filters': num_filers, 
                        'stride': stride, 
                        'layer_type': layer_type}

        # create train_opts
        train_opts = {'lr': 0.1, 
                    'weight_decay': 0.0001, 
                    'batch_size': 128, 
                    'momentum': 0.9, 
                    'num_epochs': 25, 
                    'step_size': 20, 
                    'gamma': 0.1}

        # create model base on tetspect_opts
        model = cnn_categorization_base(netspec_opts)

    elif model_type == "improved":
        # create netspec_opts
        # adding another layer with 128 filters and stride 1 so size doesn't change before pooling
        layer_type =    ['conv', 'bn', 'relu', 'conv', 'bn', 'relu', 'conv',  'bn', 'relu',   'conv',  'bn', 'relu', 'pool', 'conv']
        kernel_size =   [3,     0,      0,      3,      0,      0,      3,      0,      0,      3,      0,      0,      8,      1]
        stride =        [1,     0,      0,      2,      0,      0,      2,      0,      0,      1,      0,      0,      1,      1]
        num_filers =    [16,    16,     0,      32,     32,     0,      64,     64,     0,      128,    128,    0,      0,      16]

        netspec_opts = {'kernel_size': kernel_size,
                        'num_filters': num_filers, 
                        'stride': stride, 
                        'layer_type': layer_type}

        # create train_opts
        train_opts = {'lr': 0.1, 
                    'weight_decay': 0.0001, 
                    'batch_size': 128, 
                    'momentum': 0.9, 
                    'num_epochs': 25, 
                    'step_size': 10, 
                    'gamma': 0.1}

        # create improved model
        model = cnn_categorization_improved(netspec_opts)

        # data augmentation
        original_tr = train_ds.tensors[0].clone()
        new_training = original_tr.clone()
        new_labels = train_ds.tensors[1]

        # randomized crop
        crops = transforms.RandomResizedCrop(32)(original_tr)               # crop the images in sequence
        new_training = torch.cat((new_training, crops))                     # add to training set
        new_labels = torch.cat((new_labels, train_ds.tensors[1]))           # add corresponding labels

        # horizontal mirroring
        flips = transforms.RandomHorizontalFlip(1)(original_tr)
        new_training = torch.cat((new_training, flips))                     
        new_labels = torch.cat((new_labels, train_ds.tensors[1]))             

        # add new data to training set
        train_ds.tensors = tuple((new_training, new_labels))        # change back to tuple
    else:
        raise ValueError(f"Error: unknown model type {model_type}")

    # uncomment the line below if you wish to resume training of a saved model
    # model.load_state_dict(load(PATH to state))

    # train the model
    train(model, train_ds, val_ds, train_opts, exp_dir)

    # save model's state and architecture to the base directory
    state_dictionary_path = f"{model_type}_state_dict.pt"
    save(model.state_dict(), state_dictionary_path)
    model = {"state":state_dictionary_path, "specs": netspec_opts}
    save(model, "{}-model.pt".format(model_type))

    plt.savefig(f"{model_type}-categorization.png")
    plt.show()


if __name__ == '__main__':
    # Change the default values for the various parameters to your preferred values
    # Alternatively, you can specify different values from the command line
    # For example, to change model type from base to improved
    # type <cnn_categorization.py --model_type improved> at a command line and press enter
    args = ArgumentParser()
    args.add_argument("--model_type", type=str, default="improved", required=False,
                      help="The model type must be either base or improved")
    args.add_argument("--data_path", type=str, default="image_categorization_dataset.pt",
                      required=False, help="Specify the path to the dataset")
    args.add_argument("--contrast_normalization", type=bool, default=False, required=False,
                      help="Specify whether or not to do contrast_normalization")
    args.add_argument("--whiten", type=bool, default=False, required=False,
                      help="Specify whether or not to whiten value")

    args, _ = args.parse_known_args()
    cnn_categorization(**args.__dict__)
