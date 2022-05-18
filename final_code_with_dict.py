# Imorting the required libraries.
import cv2
from skimage.metrics import structural_similarity as compare_ssim
import math
import time
import numpy as np

# Read the video
cap = cv2.VideoCapture('03-29-2022-10_Trim.mp4')

# Frames Per Second
fps = int(cap.get(cv2.CAP_PROP_FPS))
print('FPS', fps)

# Image of normal condition.
frame1 = cv2.imread('empty.png')

# Select the Region appply the co-ordinates to the image.
r = cv2.selectROI('select ROI', frame1)
img1 = frame1[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]

# Variables for the processing.
run_once = 0
vehicle_count = 0
for_d = []
for_s = []
run2 = 0
start_time = 0
end_time = 0
elapsed = []
d = dict()


# To skip the Frames.
frame_start = 1
cap.set(1, frame_start)
frame_count = frame_start
skip_count = 0

# While loop to read the frames.
while True:
    frame_count += 1
    skip_count += 17
    cap.set(1, skip_count)

    # Read the Frames From the Video and break the loop if frames are None.
    ret, frame = cap.read()
    if ret is False and frame is None:
        break

    # ROI co-ordinates for the frame taken from the video.
    img2 = frame[r[1]:r[1] + r[3], r[0]:r[0] + r[2]]

    #Convert the read images to the grayscale.

    grayA = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # After Gray-scaling compare the both images.
    (score, diff) = compare_ssim(grayA, grayB, full=True)
    diff = (diff * 255).astype("uint8")
    # Applying Ceil function to Round the Score occured from comparision
    # s = math.ceil(score*100)/100
    s = score
    # print('ssim',s)

    # Condition For Vehical Count.
    if s <= 0.4 and run_once == 0:
        vehicle_count += 1
        for_d.append(vehicle_count)
        for_s.append(s)
        print('vehicle count',vehicle_count)
        print('SSIM Score : ',s)
        run_once = 1
    if s > 0.5:
        run_once = 0

    # Condition to measure the time difference of the Vehicle appeared.
    if s <= 0.4 and run2 == 0:
        start_time = time.time()
        run2 = 1
    if s > 0.5 and run2 == 1:
        end_time = time.time()
        timediff = (end_time - start_time)
        elapsed.append(timediff)
        run2 = 0

    # Show the frames.
    cv2.imshow('Normal-conditioned', img1)
    cv2.imshow('Frame', img2)
    cv2.imshow('Difference', diff)
    cv2.imshow('original ',frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break


# Insertion of Values to the Dictionary.
for key in for_d:
   for t in elapsed:
      d[key] = ['Vehicle_Count : ', key]
      d[key].extend(('Time : ', t))
      elapsed.remove(t)
      break

print(d)

cap.release()
cv2.destroyAllWindows()