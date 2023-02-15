from torch import nn
from cnn_categorization_base import computePadding

def cnn_categorization_improved(netspec_opts):
    """
    Constructs a network for the improved categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the improved network's architecture.

    Returns
    -------
    A categorization model which can be trained by PyTorch

    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()

    # add layers as specified in netspec_opts to the network
    kernel_size, num_filters, stride, layer_type = netspec_opts['kernel_size'], netspec_opts['num_filters'], netspec_opts['stride'], netspec_opts['layer_type']
    prev_conv_layer = 0
    cl = 1  # convolutional layer number
    bl = 1  # batch normalization layer number
    rl = 1  # relU layer number 
    pl = int(layer_type.index('pool')/3)  # pooling layer number

    for l in range(len(kernel_size)):
        # Convolutional layer        
        if layer_type[l] == 'conv':
            padding = computePadding(kernel_size[l])
            if l == len(kernel_size) - 1:   # last layer is prediction layer
                name = 'pred'
                padding = 0
            else:
                name = 'conv_'+str(cl)
            
            if l - 1 < 0:
                net.add_module(name, nn.Conv2d(3, num_filters[l], kernel_size[l], stride[l], padding))
            else:
                net.add_module(name, nn.Conv2d(num_filters[prev_conv_layer], num_filters[l], kernel_size[l], stride[l], padding=padding))
            prev_conv_layer = l
            cl += 1
        
        # Batch Normalization layer
        elif layer_type[l] == 'bn':
            net.add_module('bn_'+str(bl), nn.BatchNorm2d(num_filters[l-1]))
            bl += 1
        
        # Rectified Linear Unit Layer
        elif layer_type[l] == 'relu':
            net.add_module('relu_'+str(rl), nn.ReLU())
            rl += 1

        # Pooling Layer
        elif layer_type[l] == 'pool':
            net.add_module('pool_'+str(pl), nn.AvgPool2d(kernel_size[l], stride[l], 0))
            pl +=1

    return net
