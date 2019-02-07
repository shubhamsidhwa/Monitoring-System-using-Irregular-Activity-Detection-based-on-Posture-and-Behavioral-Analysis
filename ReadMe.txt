MONITORING SYSTEM USING IRREGULAR ACTIVITY DETECTION
BASED ON POSTURE AND BEHAVIORAL ANALYSIS

Public places like restaurants and certain other viable
places like banks or ATM’s are in need of constant
supervision against amoral and undesirable activities.
These undesirable activites are often markedly preceded
by apparent behavioral quirks. However, constant
manual monitoring using security cameras may miss
abnormal cues preceeding unpleasant consequences due
to human error. Picking up these quirks can improve the
efficiency of monitoring any public place.This project
attempts to semi-automize the security system using
machine learning and image processing techniques to
increase the efficacy and accuracy in the prediction of
abnormal behaviour.

Getting Started
Downlod the zip file to your machine.
Unzip it.

Prerequisites

Python 3+ is required
Following libraries need to be installed :
scipy
cv2
csv
PIL
numpy as np
os
pandas
keras
sklearn
math


Installing

If any library is not installed, install them by pip install libraryname

Example, pip install keras will install the keras library

Running the codes :

1) Run the generateCSV.py file.(You don't need to do this step)
For this, you need access to the entire CASIA database.
The link for the same is : ftp://surveillance.idealtest.org/
Due to confidentiality purposes, we can't provide you the password for the same.
Hence, we have downloaded these videos and prepared the csv file and attaching it along with this folder.

The dataset_act.csv file contains the values of the OMV values for each video in the first column and value as 0 or 1 in the second column
which is used for binary classification.

2)Run the TrainAndTest.py file
Run this code.
It will classify the test videos as normal or abnormal activity.
All but one video is classified as per the ground truth expectation.

Videos are present in the test2 folder and divided as normal and subnormal.

This project was developed by : 
Shubham Sidhwa

Acknowledgments
I would like to express my gratitude to
Dr. Ghassan AlRegib for providing me an
opportunity to pursue this project and provide
continuous support. Further I also
thank Mr. Wenjie Y. for approving my application
and providing me access to the CASIA
human activity database.
