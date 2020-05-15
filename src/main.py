"""
Module for video processing using opencv
"""
import time
import argparse
import sys
import os
import numpy as np
import pandas as pd
import cv2
import streamlit as st
import plotly.graph_objects as go
from sys import getsizeof
from compress_spatial import compress_image

PARSER = argparse.ArgumentParser()
PARSER.add_argument("source", help="The file with the source video")
ARGS = PARSER.parse_args()

def main():
    """
    Main loop 
    """
    cap = cv2.VideoCapture(ARGS.source)

    start_time = time.time()
    # Collect the pixels of the first frame
    _, prev_frame = cap.read()

    pixel_trajectories = prev_frame
    pixel_trajectories_blue = []
    pixel_trajectories_green = []
    pixel_trajectories_red = []
    test_pixel_coordinates = (435, 1000)
    pixel_trajectory = []
    # Read until video is completed
    frame_count = 1
    while cap.isOpened():
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret and frame_count < 15:
            frame_count += 1
            compressed_frame = compress_image(frame, 98)
            pixel_trajectories_blue.append(compressed_frame[:, :, 0])
            pixel_trajectories_green.append(compressed_frame[:, :, 1])
            pixel_trajectories_red.append(compressed_frame[:, :, 2])
            pixel_trajectory.append(compressed_frame[test_pixel_coordinates][0])
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()
    print(f'Elapsed time: {time.time() - start_time}, Frames: {frame_count}')

    # diff_blue = np.diff(np.asarray(pixel_trajectories_blue), axis=0)
    # diff_green = np.diff(np.asarray(pixel_trajectories_green), axis=0)
    # diff_red = np.diff(np.asarray(pixel_trajectories_red), axis=0)

    encoder = np.zeros((frame_count-1), dtype=np.float128) + 10
    exponents = np.arange(frame_count-1, dtype=np.uint16) * 3
    encoder = 1 / np.power(encoder, exponents)
    np.set_printoptions(precision=frame_count*2, suppress=True)
    print('Encoder:', encoder)
    shape = np.asarray(pixel_trajectories_blue).shape
    pixel_trajectories_blue = np.asarray(pixel_trajectories_blue).reshape((shape[2], shape[1], shape[0]))
    encoded_blue = np.dot(pixel_trajectories_blue, encoder)
    print('Pixel trajectories:', pixel_trajectories_blue[test_pixel_coordinates, :])
    print('Pixel trajectory:', pixel_trajectory)
    print('Encoded trajectory: {0:.100f}'.format(encoded_blue[test_pixel_coordinates[1], test_pixel_coordinates[0]]))
    np.savez_compressed('blue.npz', encoded_blue)
    print('Elapsed time:', time.time() - start_time)
 
    # data_plot = []
    # for column in np.arange(diff_blue.shape[1]):
    #     for row in np.arange(diff_blue.shape[2]):
    #         data_plot.append(go.Scatter(x=np.arange(diff_blue.shape[0]), y=diff_blue[:, column, row]))
    # # data_plot.append(go.Scatter(x=np.arange(diff_green.shape[0]), y=diff_green[:, 100, 400]))
    # # data_plot.append(go.Scatter(x=np.arange(diff_red.shape[0]), y=diff_red[:, 100, 400]))
    # fig = go.Figure(data=data_plot)
    # st.write(fig)
    
    # data_plot = []
    # data_plot.append(go.Scatter(x=np.arange(frame_count), y=np.asarray(pixel_trajectories_blue)[:, 100, 400]))
    # data_plot.append(go.Scatter(x=np.arange(frame_count), y=np.asarray(pixel_trajectories_green)[:, 100, 400]))
    # data_plot.append(go.Scatter(x=np.arange(frame_count), y=np.asarray(pixel_trajectories_red)[:, 100, 400]))

    # fig = go.Figure(data=data_plot)
    # st.write(fig)
    
    cv2.destroyAllWindows()
if __name__ == '__main__':

    main()
