# general  imports
import sys
import csv
import json
import numpy as np
from scipy import misc
import cv2

# neural network imports
import theano

# ################## CONSTANTS ##################
N_CLASSES = 9  # number of output units

IMG_DIR = '/home/arthur/Master/SDDM/data/yelp_dataset_photos/photos_resized/'
META_DATA_FILE = 'meta/image_meta.csv'
IMG2SHOP_FILE = 'meta/photo_id_to_business_id.json'
IMG_NAMES_FILE = 'meta/img_names.txt'
IMG_Y_SIZE = 224
IMG_X_SIZE = 224
BATCH_SIZE = 192  # Batch size
WEIGHT_DECAY = 0.0005
MAX_IMGS = 100


# ################## Network ##################
def dictionary(META_DATA_FILE):
    # print "Loading data"
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


def images_to_mem(image_idx):
    print "Putting images into memory for %s images" % (len(image_idx))
    X_imgs = []
    y_imgs = []
    dic = dictionary(META_DATA_FILE)

    for img_id in image_idx:
        file_path = IMG_DIR + img_id[:-1] + '.jpg'  # -1 to remove "\n" at end of line
        try:
            face = misc.imread(file_path)

            #### DATA AUGMENTATION ####
            # as per paper add the (horizontal) mirror image of each img
            face_lr = np.fliplr(face)
            # face = np.memmap(file_path, dtype=np.uint8, shape=(224, 224, 1))

            face = face.reshape(-1, 1, IMG_X_SIZE, IMG_Y_SIZE)
            face_lr = face_lr.reshape(-1, 1, IMG_X_SIZE, IMG_Y_SIZE)

            img_id = img_id[:22]
            if img_id in dic:
                X_imgs.append(face / np.float32(256))
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
    val_size = n_imgs - train_size - test_size  # use all left over imgs

    train_ids = image_ids[:train_size]
    test_ids = image_ids[train_size:train_size + test_size]
    val_ids = image_ids[train_size + test_size:]

    X_valid, y_valid = images_to_mem(val_ids)
    X_test, y_test = images_to_mem(test_ids)
    X_train, y_train = images_to_mem(train_ids)

    del train_ids
    del val_ids
    assert len(X_train) == len(y_train)

    return X_train, y_train, X_valid, y_valid, X_test, y_test



# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.
def main(model='cnn', num_epochs=200):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()
    results = [0] * 10
    tgt = [0] * 10
    usr_val = [0] * 10
    ok = [0] * 10

    for i in range(0, 10):
        img = np.array(X_val[i][0] * 255, dtype=np.uint8)
        cv2.imshow('image', img)
        input = cv2.waitKey(0)
        usr_val[i] = int(chr(input & 255))
        tgt[i] = y_val[i]
        results[i] = int(tgt[i]) - int(usr_val[i])
        ok[i] = int(tgt[i]) == int(usr_val[i])
    cv2.destroyAllWindows()

    print "user:" + str(usr_val)
    print "targ:" + str(tgt)
    print "resu:" + str(results)
    correct = 0
    for i in range(0, 10):
        if ok[i]:
            correct += 1
    print "ok:" + str(ok)
    print "acc:" + str(correct / 10.0)


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
