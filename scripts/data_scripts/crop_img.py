import json
import cv2


with open('bbox_detections.json') as bbox_file:
    bbox = json.load(bbox_file)

with open('instancesonly_filtered_gtFine_val.json') as anno_file:
    anno = json.load(anno_file)
    # print(bbox[0])
    for b in bbox:
        count = 0 
        print(b['image_id'])
        for a in anno['images']:
            if a['id'] == b['image_id']:
                # if b['category_id']
                bounding_box = b['bbox']
                print(bounding_box[0])
                print(a['file_name'])
                img = cv2.imread('/media/tungngo/DATA/Chuyen_mon/AI/dataset/city_scape/leftImg8bit_trainvaltest/leftImg8bit/original/val/'+a['file_name'])
                # crop_image = img[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+ bounding_box[2])]
                crop_image = img[int(bounding_box[1]):int(bounding_box[1]+bounding_box[3]),int(bounding_box[0]):int(bounding_box[0]+ bounding_box[2])]
                cv2.imwrite('/home/tungngo/AI/video_motorcycle/result_crop/'+str(count)+a['file_name'],crop_image)
                count+=1