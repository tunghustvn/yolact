#!/usr/bin/env python

import os
import numpy as np
import sys
import cv2
import scipy.misc
from PIL import Image
from resizeimage import resizeimage
from collections import namedtuple

'''
path_to_images = os.getcwd()
images = os.listdir(path_to_images)
# print(images)

resized_path = '/media/tungngo/DATA/Chuyen_mon/AI/dataset/city_scape/leftImg8bit_trainvaltest/leftImg8bit/train/aachen/resized0603/'

for image in images:

    img = cv2.imread(image, cv2.IMREAD_UNCHANGED) 

    width = 1024
    height = 512

    dim = (1024,512)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
    # cv2.imshow("Resized image", resized)
    cv2.imwrite(resized_path+image,resized)


# cv2.waitKey(0)
cv2.destroyAllWindows()
'''

def resize_image(data_dir,out_dir):
    ### Sets of data (folder)
    sets = [
        'leftImg8bit_val',
        'leftImg8bit_train',
        'leftImg8bit_test',
    ]

    ### Annotations directory
    img_dirs = [
        'leftImg8bit_trainvaltest/leftImg8bit/val',
        'leftImg8bit_trainvaltest/leftImg8bit/train',
        'leftImg8bit_trainvaltest/leftImg8bit/test',
    ]


    for data_set, img_dir in zip(sets, img_dirs):
        print('Starting %s' % data_set)
        img_dir = os.path.join(data_dir, img_dir)
        print(img_dir)
        # print(2)

        for root, _, files in os.walk(img_dir):
            # print(1)
            for filename in files:
                # print(os.path.join(root,filename))
                if 'png' not in filename:
                    continue
                with open(os.path.join(root,filename),'r+b') as f:
                    with Image.open(f) as image:
                        cover = resizeimage.resize_cover(image, [1024, 512])
                        out_img = data_set + '/' + filename
                        cover.save(os.path.join(out_dir,out_img), image.format)

if __name__ == '__main__':
    data_dir = '/media/tungngo/DATA/Chuyen_mon/AI/dataset/city_scape'
    out_dir = '/media/tungngo/DATA/Chuyen_mon/AI/dataset/city_scape/leftImg8bit_trainvaltest/leftImg8bit/resized_images'
    resize_image(data_dir, out_dir)