import numpy as np
import cv2


from models.experimental import attempt_load
from utils.general import non_max_suppression
import torch

class YOLOv7(object):
    def __init__(self, weights = "YoloV7x-m-c.pt"):
        self.device_choice = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = torch.device(self.device_choice)
        #Load model
        self.model = attempt_load(weights, map_location=self.device)
        #FP16 (seems to be if device != cpu)
        self.half = self.device_choice != 'cpu'
        if self.half:
            self.model.half()

        #Non max
        self.conf_thres = 0.25
        self.iou_thres = 0.45
    
    def warm_up(self):
        #Warmup
        if self.half:
            self.model(torch.zeros(1, 3, 640, 640).to(self.device).type_as(next(self.model.parameters()))) #TODO 640 to variable image size

    def run_net(self, img): #img in is a torch tensor (cpy or cuda)

        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0) #WTF?

        pred = self.model(img)[0] #What about augment, I dont know...
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
        
        return pred[0].cpu().detach().numpy()

class LabelReader(object):
    def __init__(self, labelFile, device = 'gpu'): #Labels supposed to be -> frame#, yolov7 format detection
        with open(labelFile, 'r') as f:
            reads = f.readlines()
            self.labels = []
            hold = []
            c = 0
            for line in reads:
                decon = line.split(' ')
                if c == int(decon[0]):
                    hold.append([int(x) for x in decon[1:]])
                else:
                    self.labels.append(hold)
                    hold = []
                    c += 1
        self.labels.reverse()
        self.device = device
    
    def warm_up(self):
        return None
    
    def run_net(self, img):
        return np.array(self.labels.pop())