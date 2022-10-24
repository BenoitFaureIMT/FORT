#Dependencies
import numpy as np

#Module for all utility functions
#Catalog:
#   Conversion:
#       - xyxy_to_xywh take in bbox with xyxy and return bbox with xy center and wh
#   Calculations:
#       - IoU take two bbox xyxy and xywh and return IoU

####################################CONVERSION####################################
def xyxy_to_xywh(bbox):
    bbox[2], bbox[3] = bbox[2] - bbox[0], bbox[3] - bbox[1]
    bbox[0] += bbox[2]/2
    bbox[1] += bbox[3]/2
    return bbox

####################################CALCULATION################################
def IoU(box1, box2) :
    x1 = max(box1[0] - box1[2]/2, box2[0] - box2[2]/2)
    y1 = max(box1[1] - box1[3]/2, box2[1] - box2[3]/2)
    x2 = min(box1[0] + box1[2]/2, box2[0] + box2[2]/2)
    y2 = min(box1[1] + box1[3]/2, box2[1] + box2[3]/2)
    width = (x2 - x1)
    height = (y2 - y1)
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    area_combined = area1 + area2 - area_overlap
    iou = area_overlap / area_combined
    return iou