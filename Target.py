#Dependencies
import numpy as np

from Utils import xyxy_to_xywh

#Class containing the definition of a target/tracklet
class target(object):
    def __init__(self, bbox, features, id): #bbox -> [x, y, w, h, ...]
        #Initialize association info
        self.id = id
        self.age = 1

        #Initialize ReID info
        self.features = features

        #Initialize tracking info
        #   Initialize state info
        self.state = np.expand_dims(np.append(bbox[:4], [0, 0, 0, 0]), axis=0).T #TODO : is :4 necessary? + check if conversion still viable
        self.pred_state = self.state
        #   Other
        self.last_detected_state = self.state
        self.missed_detection = False
        self.time_since_last_detection = 0
    
    def update_feature(self, n_feature, alpha):
        self.features = self.features * alpha + n_feature * (1 - alpha)