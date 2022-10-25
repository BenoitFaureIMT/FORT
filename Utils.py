#Dependencies
import numpy as np
import cv2

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

def clamp01(inp):
    return  0 if (inp < 0) else (1 if (inp > 1) else inp)

####################################CALCULATION####################################
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

######################################DRAWING######################################
def draw_txt(img, label, x1, y1, color = (0, 0, 255), text_color = (255, 255, 255)):
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

    cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)
    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

def display_both(targs, out, img):
    img_h = img.shape[0]
    img_w = img.shape[1]

    for i in range(len(out)):
        x,y,x2,y2 = out[i][:4]
        cv2.rectangle(img, (int(x), int(y)), (int(x2), int(y2)), (0,255,0), 1)
    
    for t in targs:
        x,y,w,h = t.state[0]*img_w,t.state[1]*img_h,t.state[2]*img_w,t.state[3]*img_h
        cv2.rectangle(img, (int(x - w/2), int(y - h/2)), (int(x + w/2), int(y + h/2)), (0,0,255) if t.missed_detection else (255, 0, 0), 1)
        draw_txt(img, str(t.id), int(x - w/2), int(y + h/2), color = (255, 0, 0))
    
    return img