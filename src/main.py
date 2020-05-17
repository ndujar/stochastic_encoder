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
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    max_frames = 4
    print(height, width, max_frames)
    scale_percent = 50 # percent of original size
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    dim = (width, height)
    # resize image
    pixel_trajectories_blue = np.zeros((height, width), dtype=np.uint8)
    pixel_trajectories_green = np.zeros((height, width), dtype=np.uint8)
    pixel_trajectories_red = np.zeros((height, width), dtype=np.uint8)
    test_pixel_coordinates = (int(height * np.random.random()), int(width * np.random.random()))
    print('Testing in', test_pixel_coordinates)
    pixel_trajectory = []
    # Read until video is completed
    frame_count = 0
    while True:
    # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            print(frame_count)
            if frame_count % 3 == 0:

                frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA) 

                compressed_frame = compress_image(frame, 95)
                pixel_trajectories_blue = np.dstack((pixel_trajectories_blue, compressed_frame[:, :, 0]))
                pixel_trajectories_green = np.dstack((pixel_trajectories_green, compressed_frame[:, :, 1]))
                pixel_trajectories_red = np.dstack((pixel_trajectories_red, compressed_frame[:, :, 2]))

                pixel_trajectory.append(compressed_frame[test_pixel_coordinates[0],
                                                        test_pixel_coordinates[1],
                                                        0])
                print('Pixel trajectory:', pixel_trajectory)
                print('Stored frames:', pixel_trajectories_blue.shape)
                if pixel_trajectories_blue.shape[2] == max_frames:
                    break
        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    print(f'Elapsed time: {time.time() - start_time}, Frames: {frame_count}')
    print('Pixel trajectory:', pixel_trajectory)

    pixel_trajectories_blue = np.asarray(pixel_trajectories_blue)
    diff_green = np.asarray(pixel_trajectories_green)
    diff_red = np.asarray(pixel_trajectories_red)

    np.set_printoptions(precision=max_frames*2,
                        suppress=True,
                        threshold=sys.maxsize)

    encoder = np.zeros((max_frames), dtype=np.float64) + 10
    exponents = np.asarray(range(max_frames)) * 3

    encoder = 1 * np.power(encoder, exponents)

    print('Exponents:', exponents)
    print('Encoder.', encoder)
    encoded_blue = np.matmul(pixel_trajectories_blue, encoder, dtype=np.float64) + 10 ** 12
    print('Encoded B channel shape:', encoded_blue.shape)
    print('Encoded B channel size in system:', sys.getsizeof(encoded_blue))
    print('Pixel trajectories:', pixel_trajectories_blue[test_pixel_coordinates[0], test_pixel_coordinates[1], :], pixel_trajectories_blue.shape)
    
    print('Encoded trajectory: {:f}'.format(encoded_blue[test_pixel_coordinates[0], test_pixel_coordinates[1]]))
    encoded_path = 'blue.npz'
    np.savez_compressed(encoded_path, encoded_blue)
    b = os.path.getsize(encoded_path)
    print('Compressed encoded B channel size in system:', b)
    print('Elapsed time:', time.time() - start_time)
    print('FPS:', frame_count / (time.time() - start_time))
    data_plot = []

    with np.load(encoded_path) as data:
        encoded_blue = data['arr_0'] - 10 ** 12
    decoder = -np.sort(-encoder)
    for step in np.nditer(decoder):
        print(decoder, step)
        print(encoded_blue[test_pixel_coordinates])
        frame = encoded_blue // step
        
        print(frame[test_pixel_coordinates])
        # cv2.imshow('decoded', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break
    
    data.close()
    # for column in np.arange(pixel_trajectories_blue.shape[1]):
    #     for row in np.arange(pixel_trajectories_blue.shape[2]):
    #         data_plot.append(go.Scatter(x=np.arange(pixel_trajectories_blue.shape[0]), y=pixel_trajectories_blue[:, column, row]))
    # # data_plot.append(go.Scatter(x=np.arange(diff_green.shape[0]), y=diff_green[:, 100, 400]))
    # # data_plot.append(go.Scatter(x=np.arange(diff_red.shape[0]), y=diff_red[:, 100, 400]))
    # fig = go.Figure(data=data_plot)
    # st.write(fig)
    
    data_plot = []
    data_plot.append(go.Scatter(x=np.arange(frame_count), y=pixel_trajectories_blue[test_pixel_coordinates[0], test_pixel_coordinates[1], :]))
    # data_plot.append(go.Scatter(x=np.arange(frame_count), y=np.asarray(pixel_trajectories_green)[:, 100, 400]))
    # data_plot.append(go.Scatter(x=np.arange(frame_count), y=np.asarray(pixel_trajectories_red)[:, 100, 400]))

    fig = go.Figure(data=data_plot)
    st.write(fig)
    
if __name__ == '__main__':

    main()
