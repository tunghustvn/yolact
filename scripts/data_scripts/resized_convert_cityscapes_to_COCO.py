#!/usr/bin/env python


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
# import h5py
import json
import os
import scipy.misc
import sys
import cv2

import numpy as np

from PIL import Image
from collections import namedtuple

# import cityscapesscripts.evaluation.instances2dict_with_polygons as cs

from instance import *
import cv2_util

def cal_ratio(desire_size, old_size):

    global r_img

    # Ratio to resize image
    r_img = float(desire_size)/old_size

    return r_img


#-------------------------------------------------
# Definitions
#-------------------------------------------------

# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )

#-------------------------------------------------
# A list of all labels
#-------------------------------------------------



labels = [
    #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
    Label(  'unlabeled'            ,  0 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'ego vehicle'          ,  1 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'rectification border' ,  2 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'out of roi'           ,  3 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'static'               ,  4 ,      255 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
    Label(  'dynamic'              ,  5 ,      255 , 'void'            , 0       , False        , True         , (111, 74,  0) ),
    Label(  'ground'               ,  6 ,      255 , 'void'            , 0       , False        , True         , ( 81,  0, 81) ),
    Label(  'road'                 ,  7 ,        0 , 'flat'            , 1       , False        , False        , (128, 64,128) ),
    Label(  'sidewalk'             ,  8 ,        1 , 'flat'            , 1       , False        , False        , (244, 35,232) ),
    Label(  'parking'              ,  9 ,      255 , 'flat'            , 1       , False        , True         , (250,170,160) ),
    Label(  'rail track'           , 10 ,      255 , 'flat'            , 1       , False        , True         , (230,150,140) ),
    Label(  'building'             , 11 ,        2 , 'construction'    , 2       , False        , False        , ( 70, 70, 70) ),
    Label(  'wall'                 , 12 ,        3 , 'construction'    , 2       , False        , False        , (102,102,156) ),
    Label(  'fence'                , 13 ,        4 , 'construction'    , 2       , False        , False        , (190,153,153) ),
    Label(  'guard rail'           , 14 ,      255 , 'construction'    , 2       , False        , True         , (180,165,180) ),
    Label(  'bridge'               , 15 ,      255 , 'construction'    , 2       , False        , True         , (150,100,100) ),
    Label(  'tunnel'               , 16 ,      255 , 'construction'    , 2       , False        , True         , (150,120, 90) ),
    Label(  'pole'                 , 17 ,        5 , 'object'          , 3       , True         , False        , (153,153,153) ),
    Label(  'polegroup'            , 18 ,      255 , 'object'          , 3       , False        , True         , (153,153,153) ),
    Label(  'traffic light'        , 19 ,        6 , 'object'          , 3       , True         , False        , (250,170, 30) ),
    Label(  'traffic sign'         , 20 ,        7 , 'object'          , 3       , True         , False        , (220,220,  0) ),
    Label(  'vegetation'           , 21 ,        8 , 'nature'          , 4       , False        , False        , (107,142, 35) ),
    Label(  'terrain'              , 22 ,        9 , 'nature'          , 4       , False        , False        , (152,251,152) ),
    Label(  'sky'                  , 23 ,       10 , 'sky'             , 5       , False        , False        , ( 70,130,180) ),
    Label(  'person'               , 24 ,       11 , 'human'           , 6       , True         , False        , (220, 20, 60) ),
    Label(  'rider'                , 25 ,       12 , 'human'           , 6       , True         , False        , (255,  0,  0) ),
    Label(  'car'                  , 26 ,       13 , 'vehicle'         , 7       , True         , False        , (  0,  0,142) ),
    Label(  'truck'                , 27 ,       14 , 'vehicle'         , 7       , True         , False        , (  0,  0, 70) ),
    Label(  'bus'                  , 28 ,       15 , 'vehicle'         , 7       , True         , False        , (  0, 60,100) ),
    Label(  'caravan'              , 29 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0, 90) ),
    Label(  'trailer'              , 30 ,      255 , 'vehicle'         , 7       , True         , True         , (  0,  0,110) ),
    Label(  'train'                , 31 ,       16 , 'vehicle'         , 7       , True         , False        , (  0, 80,100) ),
    Label(  'motorcycle'           , 32 ,       17 , 'vehicle'         , 7       , True         , False        , (  0,  0,230) ),
    Label(  'bicycle'              , 33 ,       18 , 'vehicle'         , 7       , True         , False        , (119, 11, 32) ),
    Label(  'license plate'        , -1 ,       -1 , 'vehicle'         , 7       , False        , True         , (  0,  0,142) ),
]

#-------------------------------------------------
# Create dictionaries for a fast lookup
#-------------------------------------------------

# id to label object
id2label        = { label.id      : label for label in labels           }

def parse_args():
    ''' Set arguments '''
    parser = argparse.ArgumentParser(description='Convert dataset')
    parser.add_argument(
        '--dataset', help="cocostuff, cityscapes", default='cityscapes_instance_only', type=str)
    parser.add_argument(
        '--outdir', help="output dir for json files", default=None, type=str)
    parser.add_argument(
        '--datadir', help="data dir for annotations to be converted",
        default=None, type=str)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def poly_to_box(poly):
    """Convert a polygon into a tight bounding box."""
    x0 = min(min(p[::2]) for p in poly)
    x1 = max(max(p[::2]) for p in poly)
    y0 = min(min(p[1::2]) for p in poly)
    y1 = max(max(p[1::2]) for p in poly)
    box_from_poly = [x0, y0, x1, y1]

    return box_from_poly

def xyxy_to_xywh(xyxy_box):
    xmin, ymin, xmax, ymax = xyxy_box
    TO_REMOVE = 1
    xywh_box = (xmin, ymin, xmax - xmin + TO_REMOVE, ymax - ymin + TO_REMOVE)
    return xywh_box

def instances2dict_with_polygons(imageFileList):
    imgCount = 0
    instanceDict = {}

    for imageFileName in imageFileList:

        #Load image
        img = Image.open(imageFileName) ### need to run: pip install Pillow

        imgNp = np.array(img)

        # Initialize label categories
        instances = {}
        for label in labels:
            instances[label.name] = []

        # Loop through all instance ids in instance image
        for instanceId in np.unique(imgNp): ### Quy 1 day so ve cac so co trong day VD 1,2,3,4,3,1,5 => 1,2,3,4,5
            if instanceId < 1000:
                continue
            instanceObj = Instance(imgNp, instanceId)
            instanceObj_dict = instanceObj.toDict()

            #instances[id2label[instanceObj.labelID].name].append(instanceObj.toDict())
            if id2label[instanceObj.labelID].hasInstances:
                mask = (imgNp == instanceId).astype(np.uint8)
                contour, hier = cv2_util.findContours(
                    mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                polygons = [c.reshape(-1).tolist() for c in contour]
           #     polygon = [ele*r_img for ele in polygons[0]]
           #     polygons[0] = polygon
                instanceObj_dict['contours'] = polygons

            instances[id2label[instanceObj.labelID].name].append(instanceObj_dict)

        imgKey = os.path.abspath(imageFileName)
        instanceDict[imgKey] = instances
        imgCount += 1

    return instanceDict

def obj_cls2cat_id(object_cls):
    switcher={

        
        # 'rider': ,
        # 'car':          3,
        # 'truck':        10,
        # 'bus':          6,
        # 'train':        7,
        # 'motorcycle':   4,
        # 'bicycle':      2,
        # 'traffic light':  9,
        # 'traffic sign':   8,
        'pole':           5,         ### street sign
        # 'person':       1,
    }
    return switcher.get(object_cls,"Invalid id")


def convert_cityscapes_instance_only(
        data_dir, out_dir):
    """Convert from cityscapes format to COCO instance seg format - polygons"""
    # gtCoarse dataset
    '''
    sets = [
        'gtCoarse_val',
        'gtCoarse_train',
        'gtCoarse_train_extra',
    ]
    ann_dirs = [
        'gtCoarse/val',
        'gtCoarse/train',
        'gtCoarse/train_extra',
    ]
    '''

    # gtFine dataset
    
    sets = [
        'gtFine_val',
        'gtFine_train',
        'gtFine_test',
    ]
    ann_dirs = [
        'gtFine_trainvaltest/gtFine/val',
        'gtFine_trainvaltest/gtFine/train',
        'gtFine_trainvaltest/gtFine/test',
    ]
    

    json_name = 'instancesonly_filtered_%s.json'
    ends_in = '%s_polygons.json'
    img_id = 0
    ann_id = 0
    cat_id = 1
    category_dict = {}

    category_instancesonly = [
        # 'person',
        # 'rider',
        # 'car',
        # 'truck',
        # 'bus',
        # 'train',
        # 'motorcycle',
        # 'bicycle',
        'pole',
        # 'traffic light',
        # 'traffic sign',
    ]

    path_to_json_files = os.getcwd()

    # ann_dict = {}
    # images = []
    # annotations = []


    for data_set, ann_dir in zip(sets, ann_dirs):
        print('Starting %s' % data_set)
        print('Starting %s' % ann_dir)
        # data_set = 'gtFine_train'

        ann_dict = {}
        images = []
        annotations = []
        ann_dir = os.path.join(data_dir, ann_dir)
        print(ann_dir)

        for root, _, files in os.walk(ann_dir):
            for filename in files:
                if filename.endswith(ends_in % data_set.split('_')[0]):
                    if len(images) % 50 == 0:
                        print("Processed %s images, %s annotations" % (
                            len(images), len(annotations)))
                    json_ann = json.load(open(os.path.join(root, filename)))
                    image = {}
                    image['id'] = img_id
                    img_id += 1

                    image['width'] = json_ann['imgWidth'] # *r_img
                    image['height'] = json_ann['imgHeight'] # *r_img
                    image['file_name'] = filename[:-len(
                        ends_in % data_set.split('_')[0])] + 'rightImg8bit.png'
                        # ends_in % data_set.split('_')[0])] + 'leftImg8bit.png'
                    image['seg_file_name'] = filename[:-len(
                        ends_in % data_set.split('_')[0])] + \
                        '%s_instanceIds_test.png' % data_set.split('_')[0]
                    images.append(image)

                    fullname = os.path.join(root, image['seg_file_name'])
                    # print(fullname)
                    # print('\n')
                    objects = instances2dict_with_polygons([fullname])[fullname]
                    # print(objects)
                    # print('\n')

                    for object_cls in objects:
                        if object_cls not in category_instancesonly:
                            continue  # skip non-instance categories

                        cat_id = obj_cls2cat_id(object_cls)
                        # print(cat_id)


                        for obj in objects[object_cls]:



                            if obj['contours'] == []:
                                print('Warning: empty contours.')
                                continue  # skip non-instance categories

                            len_p = [len(p) for p in obj['contours']]
                            if min(len_p) <= 4:
                                print('Warning: invalid contours.')
                                continue  # skip non-instance categories
        
                            ann = {}
                            ann['id'] = ann_id
                            ann_id += 1
                            ann['image_id'] = image['id']
                            ann['segmentation'] = obj['contours']

                            xyxy_box = poly_to_box(ann['segmentation'])
                            xywh_box = xyxy_to_xywh(xyxy_box)
                            if (xywh_box[2] < 10) and (xywh_box[2] > 250):
                                continue
                            ann['bbox'] = xywh_box

                            if object_cls not in category_dict:

                                # switch()

                                category_dict[object_cls] = cat_id
                                # cat_id += 1
                            ann['category_id'] = category_dict[object_cls]
                            ann['iscrowd'] = 0
                            ann['area'] = obj['pixelCount'] # *r_img

                            annotations.append(ann)

        ann_dict['images'] = images

        categories = [{"id": category_dict[name], "name": name} for name in
                          category_dict]
        ann_dict['categories'] = categories
        ann_dict['annotations'] = annotations

        print("Num categories: %s" % len(categories))
        print("Num images: %s" % len(images))
        print("Num annotations: %s" % len(annotations))


        # outfile = open('output_test_resized.json','w')
        # json.dump(ann_dict, outfile)
        with open(os.path.join(out_dir, json_name % data_set), 'w') as outfile:
            outfile.write(json.dumps(ann_dict))

if __name__ == '__main__':

    desire_size = 2048
    old_size = 2048
    cal_ratio(desire_size, old_size)

    args = parse_args()

    convert_cityscapes_instance_only(args.datadir, args.outdir)
