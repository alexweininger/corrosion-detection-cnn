# program to create synthetic images using the real images by rotating, mirroring, etc.

from __future__ import print_function

import os
from os.path import splitext

from PIL import Image

# out_path = '../images/Synthetic_Images/'
# img_path = '../images/Original_Images/IR_Images/'

out_path = 'data/train/non-corroded/'
img_path = 'data/train/notcorroded/'

# for each image in the images dir, create a few syntheic ones

def img_out(img_name, trans):
    name, ext = os.path.splitext(img_name)
    return out_path + name + '_' + trans + ext

for img_in in os.listdir(img_path):
    try:
        with Image.open(img_path + img_in) as im:
            out = im.transpose(Image.FLIP_LEFT_RIGHT)
            out.save(img_out(img_in, 'FLR'))
            out = im.transpose(Image.FLIP_TOP_BOTTOM)
            out.save(img_out(img_in, 'FTB'))
            out = im.transpose(Image.ROTATE_90)
            out.save(img_out(img_in, 'ROT90'))
            out = im.transpose(Image.ROTATE_180)
            out.save(img_out(img_in, 'ROT180'))
            out = im.transpose(Image.ROTATE_270)
            out.save(img_out(img_in, 'ROT270'))

    except IOError:
        print('error', img_path)
        pass