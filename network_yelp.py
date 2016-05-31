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
from lasagne.regularization import regularize_layer_params, regularize_layer_params_weighted, l1 


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
WEIGHT_DECAY = 0.0005
MAX_IMGS = 100000

# ################## Network ##################
def dictionary(META_DATA_FILE):
    #print "Loading data"
    dic = {}
    with open(META_DATA_FILE, 'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) > 4:
                dic[row[0]] = row[4]
    return dic

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


def images_to_mem(image_ids):

    print "Putting images into memory for %s images" % (len(image_ids))
    X_imgs = []
    y_imgs = []
    dic = dictionary(META_DATA_FILE)

    for img_id in image_ids:
        file_path = IMG_DIR + img_id[:-1] + '.jpg' # -1 to remove "\n" at end of line
        try:
            face = misc.imread(file_path)

            #### DATA AUGMENTATION ####
            # as per paper add the (horizontal) mirror image of each img
            face_lr = np.fliplr(face)
            #face = np.memmap(file_path, dtype=np.uint8, shape=(224, 224, 1))

            face = face.reshape(-1, 1, IMG_X_SIZE, IMG_Y_SIZE)
            face_lr = face_lr.reshape(-1, 1, IMG_X_SIZE, IMG_Y_SIZE)

            img_id = img_id[:22]
            if img_id in dic:
                X_imgs.append(face / np.float32(256))
                X_imgs.append(face_lr / np.float32(256))
                y_imgs.append(dic[img_id])
                y_imgs.append(dic[img_id])

        except Exception as e:
            print('No image for %s found in %s' % (img_id, file_path))
            pass


    X_imgs = np.array(X_imgs, dtype=theano.config.floatX)
    X_imgs = np.squeeze(X_imgs, axis=(1,))
    y_imgs = np.array(y_imgs, dtype=np.int32)

    print("loaded %s images" % len(X_imgs))
    return X_imgs, y_imgs

def load_dataset():

    image_ids = load_img_names()
    image_ids = image_ids[:MAX_IMGS]
    n_imgs = len(image_ids)


    # test_size == valid_size == train_size / 2
    train_size = int(n_imgs * 0.8)
    test_size = int(n_imgs * 0.1)
    val_size = n_imgs - train_size - test_size # use all left over imgs

    train_ids=image_ids[:train_size]
    test_ids=image_ids[train_size:train_size + test_size]
    val_ids=image_ids[train_size+test_size:]

    X_valid, y_valid = images_to_mem(val_ids)
    X_test, y_test = images_to_mem(test_ids)
    X_train, y_train = images_to_mem(train_ids)
    
    #### PREPROCESS TRAIN IMGS ####
    # as specified in the paper subtract the mean activity from the pixels
    avg = np.mean(X_train)
    X_train = X_train - avg

    del train_ids
    del val_ids
    assert len(X_train) == len(y_train)

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
def main(model='cnn', num_epochs=200):
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
        network = cnn.build(IMG_X_SIZE, IMG_Y_SIZE, N_CLASSES, input_var)
    else:
        print("Unrecognized model type %r." % model)
        return


    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    print(prediction)

    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()

    # Also add weight decay to the cost function
    weight_decay = regularize_layer_params(network, l1) * WEIGHT_DECAY
    loss += weight_decay

    # learning rate params
    # as per paper we start of with a learn rate of 0.01,
    # if the error on the validation set does not improve significantly,
    # thus is lower than LEARN_TRHESH, it is divided by 10 (or in our case,
    # multiplied by LEARN_CHANGE
    LEARN_RATE = theano.shared(np.array(0.01, dtype=theano.config.floatX))
    LEARN_THRESH = theano.shared(np.array(0.0, dtype=theano.config.floatX))
    LEARN_CHANGE = theano.shared(np.array(0.1, dtype=theano.config.floatX))

    # as per paper we use an initial learning rate of 0.01
    # we also use a momentum of 0.9, but in contrast with the paper we use
    # nesterov momentum instead of normal momentum
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=LEARN_RATE, momentum=0.9)

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

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    valid_error_best = float('inf')
    valid_error_prev = float('inf')
    params_best = lasagne.layers.get_all_param_values(network)
    degrading_patience = sys.maxint
    degrading_count = 0
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
        valid_error = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, BATCH_SIZE, shuffle=True):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            valid_error += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(valid_error / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        ##### EARLY STOPPING ####
        # if the learning rate becomes consistently worse over a time period specified by "patience",
        # the training is stopped to avoid over fitting
        if valid_error <= valid_error_prev:
            valid_error_prev = valid_error
            if valid_error < valid_error_best:
                valid_error_best = valid_error
            degrading_count = 0

            ### LEARN RATE CHANGE ###
            compare = theano.tensor.lt((valid_error_prev - valid_error), LEARN_THRESH.eval())
            if compare.eval():
                LEARN_RATE = LEARN_CHANGE * LEARN_RATE
                print "marginal improvement:" + str(valid_error_prev - valid_error) + ", change learn rate: " + str(
                    LEARN_RATE.eval())
        else:
            # as per paper, decrease learning rate as the error on the validation set is
            # only marginal
            # save network params if they are the best until now, since network is degrading
            if valid_error <= valid_error_best:
                params_best = lasagne.layers.get_all_param_values(network)
            if degrading_count < degrading_patience:
                degrading_count += 1
                print('Valid error is degrading, tries left:', degrading_patience - degrading_count)
            else:
                print("Early stopping, best validation error:", valid_error_best,
                      " current validation err:", valid_error)
                lasagne.layers.set_all_param_values(network, params_best)
                break

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
    np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
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
