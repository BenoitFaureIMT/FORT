# FORT
FORT - Fisheye Online Realtime Tracking

Paper -  
Fisheye detection dataset - https://github.com/BenoitFaureIMT/CERIS_FishEye

## Structure

Detection.py - Contains detection algorithm objects (YOLOv7)  
Filter.py - Contains the Kalman filter logic  
ReID.py - Contains ReID algorithm objects (ResNeXt50)  
Target.py - Contains the definition of a track object  
Utils.py - Utility functions  

detect.py - Used to run the code
  -> Creates the detector
  -> Creates the tracker
  -> Reads video
  -> Feeds images into detector to get detections
  -> Feeds detections into tracker

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install the required modules.

```bash
pip install -r requirements.py
```

## Usage

```bash
python detect.py --video <path to video> --show_results --save_results [other args]
```

All arguments:

| Option                  | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| --wait_screen           | add a wait screen before tracking starts                                   |
| --show_results          | display results on screen                                                  |
| --save_results          | save display results                                                       |
| --save_MOT              | save targets to file in MOT format                                         |
| --video                 | path to video file                                                         |
| --weights               | path to YoloV7 weights                                                     |
| --filter_weight_path    | path to weiths of NNFilter                                                 |
| --age_max               | age max of tracklets                                                       |
| --alpha                 | alpha coeff for EMA                                                        |
| --IoU_threshold         | IoU threshold for association                                              |
| --cosine_threshold      | Cosine threshold for association                                           |
| --cost_threshold        | Threshold between high and low score detections                            |
| --distance_threshold    | Distance threshold for association                                         |
