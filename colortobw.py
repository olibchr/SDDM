import sys
import time
import csv

import numpy as np

from PIL import Image

from os import listdir
from os.path import isfile, join
from scipy import misc


# ################## Params ##################
N_CLASSES = 2  # number of output units
# dirname = 'photos_resized/photos_resized/'
dirname = 'data'
labelfile = 'meta/image_meta.csv'



images = [f for f in listdir(dirname) if isfile(join(dirname, f))]


for file in images:
    imagename = file[:-4]

    if file[-3:] == "jpg":
        col = Image.open(dirname+file)
        gray = col.convert('L')
        gray.save(dirname+file)

