import mediapipe as mp
import cv2
import pandas as pd
import csv
import numpy as np
import matplotlib.pyplot as plt

def detect_pose_write_to_video_and_csv(session_no, which_participant):

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    # participant = '2A'
    csv_out_path = session_no + which_participant + '_poses.csv'
    # Use OpenCV’s VideoCapture to load the input video.
    video_file = 'r' + session_no + '_' + which_participant + '.mp4'

    video_out_path = session_no + which_participant + '_pose_output.mp4'
    participant = session_no + which_participant

    v = cv2.VideoCapture(video_file)
    w = int(v.get(cv2.CAP_PROP_FRAME_WIDTH))    
    h = int(v.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # Load the frame rate of the video using OpenCV’s CV_CAP_PROP_FPS
    fps = v.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_pose = cv2.VideoWriter(video_out_path, fourcc, fps, (w, h))


    with open(csv_out_path, 'w') as csv_out_file:
        csv_out_writer = csv.writer(csv_out_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:  # The landmarker is initialized.
            frame_no = 0 
            # Loop through each frame in the video using VideoCapture#read()
            while v.isOpened():
                ret, image = v.read()
                if not ret:
                    break
                # timestamp for the current frame
                frame_timestamp_ms = v.get(cv2.CAP_PROP_POS_MSEC)

                # how many frames in total in the video
                length = int(v.get(cv2.CAP_PROP_FRAME_COUNT))

                # if participant is a (left) flip the image
                if which_participant == 'a':
                    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

                # if participant is b (right) do not flip the image, so they have same orientation
                if which_participant == 'b':
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.

                image.flags.writeable = False
                keypoints = pose.process(image)
                
                # an exception (no attribute 'landmark') occurs if no landmarks have been recognised in the image 
                
                if not keypoints.pose_landmarks:
                    l_wrist_x = None
                    l_wrist_y = None
                    l_wrist_z = None
                    r_wrist_x = None
                    r_wrist_y = None
                    r_wrist_z = None
                    nose_x = None
                    nose_y = None
                    nose_z = None
                    
                else: 
                    l_wrist_x = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
                    l_wrist_y = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
                    l_wrist_z = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].z
                
                    r_wrist_x = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
                    r_wrist_y = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
                    r_wrist_z = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].z
                
                    nose_x = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x
                    nose_y = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y
                    nose_z = keypoints.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].z
                

                # Draw the pose annotation on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Write pose sample to CSV.
                csv_out_writer.writerow([participant, frame_no, frame_timestamp_ms, l_wrist_x, l_wrist_y, l_wrist_z, r_wrist_x, r_wrist_y, r_wrist_z, nose_x, nose_y, nose_z])
                # Draw the pose annotation on the image.
                mp_drawing.draw_landmarks(image, keypoints.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                output_pose.write(image)
                if cv2.waitKey(5) & 0xFF == 27:
                      break
                frame_no += 1
            v.release()
            output_pose.release()





