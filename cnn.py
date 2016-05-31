from __future__ import print_function

import numpy as np
import lasagne

def build(image_x_size, image_y_size, n_classes, input_var=None):
    weight_init_mean = 0
    weight_init_std = 0.01

    network = lasagne.layers.InputLayer(shape=(None, 1, image_x_size, image_y_size),
                                        input_var=input_var)

    # IMG_NET: 11x11 conv 55x55x48 + pool
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=48, filter_size=(11, 11),
            stride=(4, 4),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(weight_init_std, weight_init_mean),
            b=lasagne.init.Constant(0.))
    network = lasagne.layers.LocalResponseNormalization2DLayer(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # IMG_NET: 5x5 conv 27x27x128 + pool
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(weight_init_std, weight_init_mean))
    network = lasagne.layers.LocalResponseNormalization2DLayer(network)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # IMG_NET: 3x3 conv 13x13x192
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=192, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(weight_init_std, weight_init_mean))

    # IMG_NET: 3x3 conv 13x13x192
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=192, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)

    # IMG_NET: 3x3 conv 13x13x128 + pool
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # IMG_NET: dense 2048
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(weight_init_std, weight_init_mean))

    # IMG_NET: dense 2048
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=2048,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.Normal(weight_init_std, weight_init_mean))

    # Output Layer
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=n_classes,
            nonlinearity=lasagne.nonlinearities.softmax,
            W=lasagne.init.Normal(weight_init_std, weight_init_mean))

    return network