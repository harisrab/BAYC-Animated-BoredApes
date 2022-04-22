import cv2
import numpy as np
from pathlib import Path
from typing import Union

# This below mehtod will draw all those points which are from 0 to 67 on face one by one.

image = '/home/haris/BAYC-Animated-BoredApes/ape_src/ape5.jpg'
faceLandmarks = '/home/haris/BAYC-Animated-BoredApes/ape_src/ape5.pts'
startpoint = 1
endpoint = 68


def drawPoints(image, faceLandmarks, startpoint, endpoint, isClosed=False):
    points = []
    for i in range(startpoint, endpoint+1):
        point = [faceLandmarks.part(i).x, faceLandmarks.part(i).y]
        points.append(point)

    points = np.array(points, dtype=np.int32)
    cv2.polylines(image, [points], isClosed, (255, 200, 0),
                  thickness=2, lineType=cv2.LINE_8)

# Use this function for 70-points facial landmark detector model
# we are checking if points are exactly equal to 68, then we draw all those points on face one by one


def facePoints(image, faceLandmarks):
    assert(faceLandmarks.num_parts == 68)
    drawPoints(image, faceLandmarks, 0, 16)           # 0-16: Jaw line
    drawPoints(image, faceLandmarks, 17, 21)          # 17-21: Left eyebrow
    drawPoints(image, faceLandmarks, 22, 26)          # 22-26: Right eyebrow
    drawPoints(image, faceLandmarks, 27, 30)          # 27-30: Nose bridge
    drawPoints(image, faceLandmarks, 30, 35, True)    # 30-35: Lower nose
    drawPoints(image, faceLandmarks, 36, 41, True)    # 36-41: Left eye
    drawPoints(image, faceLandmarks, 42, 47, True)    # 42-47: Right Eye
    drawPoints(image, faceLandmarks, 48, 59, True)    # 48-59: Outer lip
    drawPoints(image, faceLandmarks, 60, 67, True)    # 60-67: Inner lip

# Use this function for any model other than
# 70 points facial_landmark detector model


def facePoints2(image, faceLandmarks, color=(0, 255, 0), radius=4):
    for p in faceLandmarks.parts():
        cv2.circle(image, (p.x, p.y), radius, color, -1)





def read_pts(filename: Union[str, bytes, Path]) -> np.ndarray:
    """Read a .PTS landmarks file into a numpy array"""
    with open(filename, 'rb') as f:
        # process the PTS header for n_rows and version information
        rows = version = None
        for line in f:
            if line.startswith(b"//"):  # comment line, skip
                continue
            header, _, value = line.strip().partition(b':')
            if not value:
                if header != b'{':
                    raise ValueError("Not a valid pts file")
                if version != 1:
                    raise ValueError(f"Not a supported PTS version: {version}")
                break
            try:
                if header == b"n_points":
                    rows = int(value)
                elif header == b"version":
                    version = float(value)  # version: 1 or version: 1.0
                elif not header.startswith(b"image_size_"):
                    # returning the image_size_* data is left as an excercise
                    # for the reader.
                    raise ValueError
            except ValueError:
                raise ValueError("Not a valid pts file")

        # if there was no n_points line, make sure the closing } line
        # is not going to trip up the numpy reader by marking it as a comment
        points = np.loadtxt(f, max_rows=rows, comments="}")
