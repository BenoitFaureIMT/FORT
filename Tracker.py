#Dependencies
import numpy as np
from scipy.spatial.distance import cdist
import lap

from ReID import ResNeXt50
# from Filter_con_gradient import NNFilter
from Filter import NNFilter
from Target import target
from Utils import IoU
from Logger import Logger

#Class containing the tracker logic
class Cyclop(object):
    def __init__(self, filter_weight_path = "filter_weights.npy", reid = None, age_max = 6, alpha = 0.6, IoU_threshold = 0.4, cosine_threshold = 0.25, cost_threshold = 0.4, distance_threshold = 0.2): #screen_w, screen_h,
        #Initialise ReID
        self.reid = ResNeXt50('cpu') if reid == None else reid
        
        #Initialise Filter
        self.filter = NNFilter(filter_weight_path)

        #Initialise targets
        self.targs = np.array([])
        self.age_max = age_max
        self.next_id = 0

        #Initialise EMA
        self.alpha = alpha

        #Initialise Coefficients for filtering TODO : change value and location of variables
        self.IoU_threshold = IoU_threshold #Max IoU cost value (1 - IoU) allowed
        self.cosine_threshold = cosine_threshold #Max cosine dist allowed
        self.cost_threshold = cost_threshold #Max cost value

        self.distance_threshold = distance_threshold

    #Update the state of the tracker
    def update(self, detections, image, dt):
        #Get new state predictions for each target
        for t in self.targs:
            self.filter.pred_next_state(t, dt)

        #Filter detections
        filt = detections[:,4] > 0.4#0.35
        o_filt = [not f for f in filt]
        sec_detections = detections[o_filt]
        detections = detections[filt] #TODO more work needed here + is it neded?

        #Obtain features
        detection_features = np.array([self.reid.get_features(image, bbox) for bbox in detections]) #TODO : make sure this works
        sec_detection_features = np.array([self.reid.get_features(image, bbox) for bbox in sec_detections])

        #Change detections format xyxy -> xywh
        img_h = image.shape[0]
        img_w = image.shape[1]

        detections[:, 2], detections[:, 3] = (detections[:, 2] - detections[:, 0]) / img_w, (detections[:, 3] - detections[:, 1]) / img_h
        detections[:, 0], detections[:, 1] = detections[:, 0] / img_w + detections[:, 2]/2, detections[:, 1] / img_h + detections[:, 3]/2

        sec_detections[:, 2], sec_detections[:, 3] = (sec_detections[:, 2] - sec_detections[:, 0]) / img_w, (sec_detections[:, 3] - sec_detections[:, 1]) / img_h
        sec_detections[:, 0], sec_detections[:, 1] = sec_detections[:, 0] / img_w + sec_detections[:, 2]/2, sec_detections[:, 1] / img_h + sec_detections[:, 3]/2
        
        #Cost matrix calculation - HIGH SCORE detections
        cost_matrix = None
        match, sec_match, unm_tr, unm_det = None, None, None, None
        if (len(self.targs) == 0 or len(detections) == 0):
            cost_matrix = np.array([[]])
            match, unm_tr, unm_det = [], list(range(len(self.targs))), range(len(detections))
        else:
            cost_matrix = self.get_cost_matrix(detections, self.targs, detection_features)
            #Get associations
            match, unm_tr, unm_det = self.associate(cost_matrix, self.cost_threshold)

        #Cost matrix calculation - LOW SCORE detections
        if (len(self.targs[unm_tr]) == 0 or len(sec_detections) == 0):
            sec_match = []
        else:
            cost_matrix = self.get_cost_matrix(sec_detections, self.targs[unm_tr], sec_detection_features)
            #Get associations
            sec_match, sec_unm_tr, sec_unm_det = self.associate(cost_matrix, self.cost_threshold / 3)
            sec_match = [(unm_tr[i], j) for i, j in sec_match]
            unm_tr = [unm_tr[i] for i in sec_unm_tr]

        #Indices of detections - this is for debug purposes
        list_ind_det = np.ones((len(detections),)) * -1

        #Process associations
        #   Targets which were matched - high scores
        new_targs = []
        for ind_track, ind_det in match:
            targ = self.targs[ind_track]
            self.filter.update_state(targ, detections[ind_det], dt)
            targ.update_feature(detection_features[ind_det], self.alpha)
            new_targs.append(targ)
            #Indices of detections
            list_ind_det[ind_det] = self.targs[ind_track].id

        #   Targets which were matched - low scores
        for ind_track, ind_det in sec_match:
            targ = self.targs[ind_track]
            self.filter.update_state(targ, sec_detections[ind_det], dt)
            targ.update_feature(sec_detection_features[ind_det], self.alpha)
            new_targs.append(targ)

        #   Targets which were not matched
        for ind_unm_tr in unm_tr:
            targ = self.targs[ind_unm_tr]
            self.filter.update_state_no_detection(targ, dt)

            #Eliminate old targets
            if targ.age <= self.age_max:
                new_targs.append(targ)
        
        #   New targets
        for ind_unm_det in unm_det:
            new_targs.append(target(detections[ind_unm_det], detection_features[ind_unm_det], self.next_id))
            self.next_id += 1
            #Indices of detections
            list_ind_det[ind_unm_det] = new_targs[-1].id
        
        self.targs = np.array(new_targs)
        
        #Indices of detections
        return list_ind_det
    
    def get_cost_matrix(self, detections, targs, detection_features):
        #   Calculate IOU cost matrix TODO (IOU CALCULATION TAKES xywh RIGHT NOW, NOT EFFICIENT!!!!!!!!)
        cost_matrix_IoU = np.array([[1 - IoU(t.pred_state.T[0], d) for d in detections] for t in targs])

        #Calculate distance matrix
        def get_length(v):
            return v[0]**2 + v[1]**2
        dist_matrix = np.array([[get_length(d[:2] - t.pred_state.T[0][:2]) for d in detections] for t in targs]) > self.distance_threshold

        #   Calculate features cost matrix
        cost_matrix_feature = np.maximum(0.0, cdist(np.array([t.features for t in targs]), detection_features, metric='cosine')) / 2.0 #TODO : better way of creating array? Maybe store separately?
        cost_matrix_feature[np.logical_and(cost_matrix_feature > self.cosine_threshold, cost_matrix_IoU > self.IoU_threshold)] = 1.0

        #   Final cost matrix calculation
        mat =  np.minimum(cost_matrix_IoU, cost_matrix_feature)
        mat[dist_matrix] = 1
        return mat

    def associate(self, cost_mat, cost_thres): # TODO : does this work?
        if cost_mat.size == 0:
            return np.empty((0, 2), dtype=int), tuple(range(cost_mat.shape[0])), tuple(range(cost_mat.shape[1]))
        matches, unmatch_track, unmatch_detection = [], [], []
        __, x, y = lap.lapjv(cost_mat, extend_cost=True, cost_limit=cost_thres)
        for ix, mx in enumerate(x):
            if mx >= 0:
                matches.append([ix, mx])
        unmatch_track = np.where(x < 0)[0]
        unmatch_detection = np.where(y < 0)[0]
        matches = np.asarray(matches)
        return matches, unmatch_track, unmatch_detection