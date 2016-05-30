# general  imports
import sys
import time
import csv
import json
import numpy as np
from scipy import misc

# neural network imports
import theano
import theano.tensor as T
import lasagne

# our imports
import cnn
import mlp

# ################## CONSTANTS ##################
N_CLASSES = 9  # number of output units

IMG_DIR = 'photos_resized/photos_resized/'
META_DATA_FILE = 'meta/image_meta.csv'
IMG2SHOP_FILE = 'meta/photo_id_to_business_id.json'
IMG_NAMES_FILE = 'meta/img_names.txt'
IMG_Y_SIZE = 224
IMG_X_SIZE = 224
BATCH_SIZE = 192 # Batch size

# ################## Network ##################
def dictionary(META_DATA_FILE):
    print "Loading meta data"
    dic = {}
    with open(META_DATA_FILE, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 4:
                dic[row[0]] = row[4]
    return dic

# returns a cnn network with params initialized from saved params file pointed to by path
def load_cnn(path):
    input_var = T.tensor4('inputs')
    network = cnn.build(IMG_X_SIZE, IMG_Y_SIZE, N_CLASSES, input_var)
    with np.load(path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)
    return network

# returns a dict which if you query it with a img id it returns the according shop id
def load_img2shop(file):
    print "Loading img2shop data"
    img2shop_dict = {}
    with open(file) as data_file:
        json_list = json.load(data_file)
        for item in json_list:
            img_id = item['photo_id']
            shop_id = item['business_id']
            img2shop_dict[img_id] = shop_id
    return img2shop_dict

def load_img_names():
    print "Loading img names"
    img_names = []
    with open(IMG_NAMES_FILE) as img_name_file:
        img_names = img_name_file.readlines()
    return img_names

def load_dataset():
    dic = dictionary(META_DATA_FILE)
    
    
    images = load_img_names()

    X_imgs = []
    y_imgs = []
    
    count = 0

    for img_id in images:
        file_path = IMG_DIR + img_id[:-1] + '.jpg' # -1 to remove "\n" at end of line
        try:
            face = misc.imread(file_path)
            face = face.reshape(-1, 1, IMG_X_SIZE, IMG_Y_SIZE)
            
            img_id = img_id[:22]
            if img_id in dic:
                X_imgs.append(face / np.float32(256))
                y_imgs.append(dic[img_id])
            else:
                pass
                
        except Exception as e:
            print('No image for %s found in %s' % (img_id, file_path))
            #pass

        if len(X_imgs) >= 100 * BATCH_SIZE:
            break

    print ("loaded imgs: all %s with %s targets" % ((len(X_imgs)), len(y_imgs)))

    # test_size == valid_size == train_size / 2
    n_imgs = len(X_imgs)
    train_size = n_imgs / 2
    test_size = n_imgs / 4
    valid_size = n_imgs - train_size - test_size # use all left over imgs

    # create sets from the back of the imgs list, since this is more efficient in python
    X_train = np.array(X_imgs[-train_size:], dtype=theano.config.floatX)
    X_train = np.squeeze(X_train, axis=(1,))
    y_train = np.array(y_imgs[-train_size:], dtype=np.int32)
    del X_imgs[-train_size:]
    del y_imgs[-train_size:]

    X_test = np.array(X_imgs[-test_size:], dtype=theano.config.floatX)
    X_test = np.squeeze(X_test, axis=(1,))
    y_test = np.array(y_imgs[-test_size:], dtype=np.int32)
    del X_imgs[-test_size:]
    del y_imgs[-test_size:]

    X_valid = np.array(X_imgs[-valid_size:], dtype=theano.config.floatX)
    X_valid = np.squeeze(X_valid, axis=(1,))
    y_valid = np.array(y_imgs[-valid_size:], dtype=np.int32)
    del X_imgs[-valid_size:]
    del y_imgs[-valid_size:]

    assert len(X_imgs) == 0 and len(y_imgs) == 0 # checks if all imgs are properly used

    return X_train, y_train, X_valid, y_valid, X_test, y_test

# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.
def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]

# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main(model='cnn', num_epochs=2):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()


    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model " + model + " and compiling functions...")
    if model == 'mlp':
        network = mlp.build(input_var)
    elif model.startswith('custom_mlp:'):
        depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
        network = mlp.build_custom(input_var, int(depth), int(width),
                                   float(drop_in), float(drop_hid))
    elif model == 'cnn':
        # network = cnn.build(IMG_X_SIZE, IMG_Y_SIZE, N_CLASSES, input_var)
        print("Load network from file")
        network = load_cnn('model.npz')
    else:
        print("Unrecognized model type %r." % model)
        return


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    print(prediction)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    if (False):
        print("Starting training...")
        # We iterate over epochs:
        for epoch in range(num_epochs):
            # In each epoch, we do a full pass over the training data:
            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, y_train, BATCH_SIZE, shuffle=True):
                inputs, targets = batch
                train_err += train_fn(inputs, targets)
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print("Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time))
            print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
            print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
            print("  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, BATCH_SIZE, shuffle=True):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    # Optionally, you could now dump the network weights to a file like this:
    # np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
    #
    # And load them again later on like this:
    # with np.load('model.npz') as f:
    #     param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    # lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on MNIST using Lasagne.")
        print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
        print()
        print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
        print("       'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
        print("       with DEPTH hidden layers of WIDTH units, DROP_IN")
        print("       input dropout and DROP_HID hidden dropout,")
        print("       'cnn' for a simple Convolutional Neural Network (CNN).")
        print("EPOCHS: number of training epochs to perform (default: 500)")
    else:
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['model'] = sys.argv[1]
        if len(sys.argv) > 2:
            kwargs['num_epochs'] = int(sys.argv[2])
        main(**kwargs)
