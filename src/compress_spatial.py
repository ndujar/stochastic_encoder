#! /usr/bin/env python3

import os
import sys

import numpy as np
import cv2

def compress_image(image, quality):
    """
    Function to reduce image size
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', image, encode_param)
    if result:
        decimg=cv2.imdecode(encimg,1)
        return decimg
    else:
        return None
