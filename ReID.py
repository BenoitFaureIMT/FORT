from math import floor
import numpy as np
import cv2
import time

import torch
from PIL import Image
from torchvision import transforms
from Logger import Logger

class ResNeXt50(object):
    def __init__(self, device, input_shape = (224, 224)):
        self.device = device

        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnext50_32x4d', pretrained=True)
        self.model.eval().to(self.device)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])

        self.input_shape = input_shape
    
    def extract_sub_image(self, img, bbox): #bbox -> [x1,y1,x2,y2] (x1, y1) top left, (x2, y2) bottom right in pixel coords
        return img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
    
    def extract_features(self, img):
        #Resize and pad image to fit in HEAD of ResNext50
        ratio = min(self.input_shape[0] / img.shape[0], self.input_shape[1] / img.shape[1])
        new_shape = (int(img.shape[0] * ratio), int(img.shape[1] * ratio))
        img = cv2.resize(img, new_shape, cv2.INTER_LINEAR)
        delta_w, delta_h = self.input_shape[0] - new_shape[0], self.input_shape[1] - new_shape[1]
        img = cv2.copyMakeBorder(img, delta_h//2, delta_h - delta_h//2, delta_w//2, delta_w - delta_w//2, cv2.BORDER_CONSTANT, value= (0, 0, 0))

        img = transforms.Compose([transforms.ToTensor()])(img) # ToTensor already normalises
        img = img.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(img).flatten(start_dim=1).to("cpu") #TODO alot of conversions btw cpu and gpu, need to optimize
        return np.array((output / np.linalg.norm(output))[0])
    
    def get_features(self, img, bbox):
        t = time.perf_counter()
        im = self.extract_sub_image(img, bbox)
        dt1 = time.perf_counter() - t
        a = self.extract_features(im)
        dt2 = time.perf_counter() - t
        Logger.add_to_log("ReID : ", int(dt2 * 1000), " ms | Extraction : ", int(dt1 * 1000), " ms | Network : ", int((dt2 - dt1) * 1000), " ms")
        return a