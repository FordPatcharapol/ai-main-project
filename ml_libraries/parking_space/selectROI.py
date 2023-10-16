import cv2
import numpy as np
import csv


# Read image
image = cv2.imread("parking3_img.jpg")

# Select ROI/parking spots
r = cv2.selectROIs('Selector', image, showCrosshair=False, fromCenter=False)
rlist = r.tolist()

# write the list into a csv file
with open('rois_parking3.csv', 'w', newline='') as outf:
    csvw = csv.writer(outf)
    csvw.writerows(rlist)