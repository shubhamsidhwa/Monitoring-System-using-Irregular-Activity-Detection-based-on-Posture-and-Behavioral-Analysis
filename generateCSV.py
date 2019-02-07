import cv2
import csv

import numpy as np
import os


# This function takes the input file path and returns the averagr of the 7 Hu moments
def MHI(video_src):

    x8 = []
    h = []

    # The video is read frame by frame
    cam = cv2.VideoCapture(video_src)


    while True:
        ret, frame = cam.read()

        if not ret:
            break

        h, w = frame.shape[:2]
        prev_frame = frame.copy()

        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[:, :, 1] = 255

        #Motion History data is first initialized
        motion_history = np.zeros((h, w), np.float32)
        timestamp = 0

        while True:
            ret, frame = cam.read()

            if not ret:
                break

            frame_diff = cv2.absdiff(frame, prev_frame)

            #The difference between the consecutive frames is calculated and stored

            gray_diff = cv2.cvtColor(frame_diff, cv2.COLOR_BGR2GRAY)

            #A regular binary thresholding function is used to segment the difference frame
            ret, fgmask = cv2.threshold(gray_diff, DEFAULT_THRESHOLD, 1, cv2.THRESH_BINARY)

            #Gaussian filter is added in order tol remove background noises
            fgmask = cv2.GaussianBlur(fgmask, (7, 7), 0)
            fgmask = cv2.dilate(fgmask, None, iterations=2)
            timestamp += 1

            # update motion history
            mx = cv2.motempl.updateMotionHistory(fgmask, motion_history, timestamp, MHI_DURATION)
            mg_mask, mg_orient = cv2.motempl.calcMotionGradient(motion_history, MAX_TIME_DELTA, MIN_TIME_DELTA,
                                                                apertureSize=5)
            seg_mask, seg_bounds = cv2.motempl.segmentMotion(motion_history, timestamp, MAX_TIME_DELTA)

            # normalize motion history
            mh = np.uint8(np.clip((motion_history - (timestamp - MHI_DURATION)) / MHI_DURATION, 0, 1) * 255)


            MEI = (255 * mh > 0).astype(np.uint8)
            hsv[:, :, 0] = mg_orient / 2
            hsv[:, :, 2] = mg_mask * 255
            vis1 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

            #The hu moments are calculated and condensed to one dimension
            h = cv2.HuMoments(cv2.moments(mh)).flatten()

            #The mean values of the Hu moments are computed and stacked
            x8.append((h[0] + h[1] + h[2] + h[3] + h[4] + h[5] + h[6]) / 7.0)

            mh = cv2.cvtColor(mh, cv2.COLOR_GRAY2BGR)

            # print(MEI)
            prev_frame = frame.copy()
            key = cv2.waitKey(1) & 0xFF

            if key == 27:
                break

    cv2.destroyAllWindows()

    return x8

    pass


#The Hu moments are computed for the normal videos and stored in a CSV file

path_normal= os.path.join(os.getcwd(),"normal") #r'D:\DIP_PROJECT\data\data\normal'
list_normal=os.listdir(path_normal)

with open(os.path.join(os.getcwd(),"dataset_act.csv"), 'a', newline='') as csvFile:
    writer = csv.writer(csvFile)

    for file_normal in list_normal:
        video_src = path_normal+'/'+file_normal
        MHI_DURATION = 0.5
        DEFAULT_THRESHOLD = 32
        MAX_TIME_DELTA = 0.25
        MIN_TIME_DELTA = 0.05

        #The Hu moments are computed for each video and the average is stored in the CSV file
        x8 = []
        x8 = MHI(video_src)
        print(np.mean(x8))

        #1 indicates normal video

        row = [np.mean(x8), '1']
        writer.writerow(row)



#The Hu moments are computed for the abnormal videos and stored in a CSV file

path_abnormal=os.path.join(os.getcwd(),"abnormal") #r'D:\DIP_PROJECT\data\data\abnormal'
list_abnormal=os.listdir(path_abnormal)

with open(os.path.join(os.getcwd(),"dataset_act.csv"), 'a',newline='') as csvFile:
    writer = csv.writer(csvFile)

    for file_abnormal in list_abnormal:
        video_src = path_abnormal+'/'+file_abnormal
        MHI_DURATION = 0.5
        DEFAULT_THRESHOLD = 32
        MAX_TIME_DELTA = 0.25
        MIN_TIME_DELTA = 0.05

        # The Hu moments are computed for each video and the average is appended to the same CSV file
        x8 = []
        x8 = MHI(video_src)

        print(np.mean(x8))

        # 0 indicates abnormal video

        row = [np.mean(x8), '0']
        writer.writerow(row)

    csvFile.close()
