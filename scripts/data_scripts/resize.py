import os
import numpy as np
import cv2
from PIL import Image

path_to_images = os.getcwd()
images = []
# images = os.listdir(path_to_images)
# print(images)

ends_in = '%s_polygons.json'

resized_path = '/media/tungngo/DATA/Chuyen_mon/AI/dataset/city_scape/gtFine_trainvaltest/gtFine/train/aachen/resized0603/'

data_set = 'gtFine_train'
for root, _, files in os.walk(path_to_images):
    for filename in files:
        if '_labelIds.png' in filename:
            # image = {}
            image = filename
            images.append(image)

print(images)

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