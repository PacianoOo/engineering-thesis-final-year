
import logging
import logging.handlers
import os
import time
import sys
import pandas as pd
import cv2
import numpy as np
import skvideo.io
import utils
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
import csv
import datetime
from math import sqrt
from sklearn.svm import SVR
import sklearn.svm as svm
from sklearn.linear_model import LinearRegression

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from datetime import timedelta
xrange=range
# without this some strange errors happen
cv2.ocl.setUseOpenCL(False)
# ============================================================================
IMAGE_DIR = "/home/pi/Desktop/object_detection_projects-master/opencv_traffic_capacity_counting/out"
#VIDEO_SOURCE = "/media/pi/PACIANO/1st traffic record 20 mins 430-5pm.mp4"
VIDEO_SOURCE = "/home/pi/Videos/CAM101/out000.mp4"
SHAPE = (720, 1280)
AREA_PTS = np.array([[0,720],[0, 600], [220, 230], [400, 230],[700,720],[830,10],[1150,10],[1280,300],[1280,720]])#1st traffic record 20 mins 430-5pm
# AREA_PTS = np.array([[200,0],[900,0],[1180,150],[1180,500],[900,720],[200,720]]) #gateway ketkai
#AREA_PTS = np.array([[0,0],[300,0],[1000,300],[1000,375],[170,720],[0,720]]) 2nd Ayala Footage1
#AREA_PTS = np.array([[0,0],[300,0],[750,275],[750,450],[650,720],[0,720]])
from pipeline import (
PipelineRunner,
CapacityCounter,
ContextCsvWriter
)

# ============================================================================
def main():
# os.system("ffmpeg -i rtsp://admin:qazmlpwsxnko123@192.168.1.101:554/dd-a -c copy -map 0 -fflags +genpts -f segment -segment_time 300 -segment_format mp4 -t 30 /home/pi/Videos/CAM101/out%03d.mp4")
log = logging.getLogger("main")
base = np.zeros(SHAPE + (3,), dtype='uint8')
area_mask = cv2.fillPoly(base, [AREA_PTS], (255, 255, 255))[:, :, 0]
pipeline = PipelineRunner(pipeline=[
CapacityCounter(area_mask=area_mask,save_image=True,image_dir=IMAGE_DIR,dataset = '/media/pi/PACIANO/WeatherPredictor-master/TrainingData.csv',column_names= ['Capacity','Time']), #save_image=True
ContextCsvWriter(path='/home/pi/Desktop/object_detection_projects-master/opencv_traffic_capacity_counting/report.csv',start_time=0,fps=1,faster=20,field_names=['capacity'])
# saving every 10 seconds
], log_level=logging.DEBUG)#logging.DEBUG
# Set up image source
cap= skvideo.io.vreader(VIDEO_SOURCE)
frame_number = -1
st = time.time()
try:
for frame in cap:
if not frame.any():
log.error("Frame capture failed, skipping...")
frame_number += 1
pipeline.set_context({
'frame': frame,

'frame_number': frame_number,
})
context = pipeline.run()
# skipping 10 seconds
for i in xrange(240):
next(cap)
except Exception as e:
log.exception(e)
# ==================================================================
if __name__ == "__main__":
log = utils.init_logging()
if not os.path.exists(IMAGE_DIR):
log.debug("Creating image directory `%s`...", IMAGE_DIR)
os.makedirs(IMAGE_DIR)
var=1
while var==1:
os.system("ffmpeg -i rtsp://admin:qazmlpwsxnko123@192.168.1.101:554/dd-a -c copy -map 0 -fflags +genpts -f segment -segment_time 300 -segment_format mp4 -t 30 /home/pi/Videos/CAM101/out%03d.mp4")
main()
os.system("ffmpeg -i rtsp://admin:qazmlpwsxnko123@192.168.1.102:554/dd-a -c copy -map 0 -fflags +genpts -f segment -segment_time 300 -segment_format mp4 -t 30 /home/pi/Videos/CAM101/out%03d.mp4")
AREA_PTS = np.array([[200,0],[900,0],[1180,150],[1180,500],[900,720],[200,720]])
main()
os.system("ffmpeg -i rtsp://admin:qazmlpwsxnko123@192.168.1.103:554/dd-a -c copy -map 0 -fflags +genpts -f segment -segment_time 300 -segment_format mp4 -t 30 /home/pi/Videos/CAM101/out%03d.mp4")

AREA_PTS = np.array([[0,0],[300,0],[1000,300],[1000,375],[170,720],[0,720]])
main()
os.system("ffmpeg -i rtsp://admin:qazmlpwsxnko123@192.168.1.104:554/dd-a -c copy -map 0 -fflags +genpts -f segment -segment_time 300 -segment_format mp4 -t 30 /home/pi/Videos/CAM101/out%03d.mp4")
AREA_PTS = np.array([[0,0],[300,0],[750,275],[750,450],[650,720],[0,720]])
main()
