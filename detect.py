import argparse

import cv2
import numpy as np
import time
import torch

from Detection import YOLOv7
from Tracker import Cyclop
from ReID import ResNeXt50
from Logger import Logger
from Utils import display_both, clamp01

@torch.no_grad()
def run(
    wait_screen = False,
    show_results = True,
    save_results = False,
    save_MOT = False,

    video = "testfish.avi",

    weights = "YoloV7x-m-c.pt",

    filter_weight_path = "filter_weights.npy", 
    age_max = 6, 
    alpha = 0.6, 
    IoU_threshold = 0.4, 
    cosine_threshold = 0.25, 
    cost_threshold = 0.4,
    distance_threshold = 0.2):

    det = YOLOv7(weights = weights)
    det.warm_up()
    tr = Cyclop(
        reid = ResNeXt50(det.device), 
        filter_weight_path = filter_weight_path, 
        age_max = age_max,
        alpha = alpha,
        IoU_threshold = IoU_threshold,
        cosine_threshold = cosine_threshold,
        cost_threshold = cost_threshold,
        distance_threshold = distance_threshold)
    
    #Screen to wait start of tracking
    if wait_screen:
        init_img = np.ones((640, 640, 3)) * 255
        cv2.putText(init_img, "PRESS SPACE TO START", (320 - 160, 320), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Yolov7 + FishSORT', init_img)
        while True:
            k = cv2.waitKey(100)
            if k == 32:
                break

    #Start video capture
    cam = cv2.VideoCapture(video)
    videoWidth = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
    videoHeight = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ret, frame = cam.read()

    #Init MOT file if necessary
    if save_MOT:
        mot_file = open(video.split('.')[0] + ".txt", 'w')

    #Init results file if necessary
    if save_results:
        writer = cv2.VideoWriter('output_' + video.split('/')[-1], cv2.VideoWriter_fourcc(*'DIVX'), 25, (640, 640))

    #Init variables
    frame_count = 1

    while ret:
        #Loading frame
        frame = cv2.resize(frame, (640, 640), interpolation=cv2.INTER_LINEAR)
        disp = frame.copy()
        frame = frame[:, :, ::-1].transpose(2, 0, 1)
        frame = np.ascontiguousarray(frame)

        #Logging frame number and strating timer
        Logger.frameNumber = frame_count
        t = time.perf_counter()

        #Running detection
        out = det.run_net(frame)
        out = np.clip(out, a_min = 0, a_max = 640)

        #Logging detection inference time
        Logger.totalDetections = len(out)
        Logger.detectionTime = (time.perf_counter() - t) * 1000

        #Running tracker
        ind_dets = tr.update(out.copy(), disp, 0.01)
        
        #Logging total inference time and printing
        Logger.totalTime = (time.perf_counter() - t) * 1000
        Logger.make_log()
        Logger.print_log()

        #Video feedback
        if show_results or save_results:
            img = display_both(tr.targs, out, ind_dets, disp)
            if show_results:
                cv2.imshow('Yolov5 + FishSORT', img)
            if save_results:
                writer.write(img)

        #MOT
        if save_MOT:
            for t in tr.targs:
                mot_file.write(
                str(frame_count) + "," + str(t.id + 1) + "," + 
                str(int(clamp01(t.state[0][0] - t.state[2][0]/2) * videoWidth)) + "," + str(int(clamp01(t.state[1][0] - t.state[3][0]/2) * videoHeight)) + "," + 
                str(int(clamp01(t.state[2][0]) * videoWidth)) + "," + str(int(clamp01(t.state[3][0]) * videoHeight)) + ",1,-1,-1,-1\n")

        #Escape sequence
        k = cv2.waitKey(1)
        if k == 27:
            break

        #Next frame
        frame_count += 1
        ret, frame = cam.read()

    #Logging last infos and printing
    Logger.maxId = tr.next_id
    Logger.make_final_log()
    Logger.print_log()

    #Releasing all
    cam.release()
    cv2.destroyAllWindows()
    if save_MOT:
        mot_file.close()
    if save_results:
        writer.release()

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait_screen", action='store_true', help = "add a wait screen before tracking starts")
    parser.add_argument("--show_results", action='store_true', help = "display results on screen")
    parser.add_argument("--save_results", action='store_true', help = "save display results")
    parser.add_argument("--save_MOT", action='store_true', help = "save targets to file in MOT format")
    parser.add_argument("--video", type = str, default = "testfish.avi", help = "path to video file")
    parser.add_argument("--weights", type = str, default = "YoloV7x-m-c.pt", help = "path to YoloV7 weights")
    parser.add_argument("--filter_weight_path", type = str, default = "filter_weights.npy", help = "path to weiths of NNFilter")
    parser.add_argument("--age_max", type = int, default = 6, help = "age max of tracklets")
    parser.add_argument("--alpha", type = float, default = 0.6, help = "alpha coeff for EMA")
    parser.add_argument("--IoU_threshold", type = float, default = 0.4, help = "IoU threshold for association")
    parser.add_argument("--cosine_threshold", type = float, default = 0.25, help = "Cosine threshold for association")
    parser.add_argument("--cost_threshold", type = float, default = 0.4, help = "Cosine threshold for association")
    parser.add_argument("--distance_threshold", type = float, default = 0.2, help = "Distance threshold for association")
    opt = parser.parse_args()
    return opt

def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)