## Prerequisite
python >= Python 3.11.4

## Installation

```
pip3 install -r requirements.txt
```

## Capture frame from video

```
python3 capture_frame.py
```

## Select each parking slot in frame

```
python3 selectROI.py
```

## Run Inference

```
python3 parking_backup.py   -s    /path/video   // video
                            -r    /path/rois   // csv
```
