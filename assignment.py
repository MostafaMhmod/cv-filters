#*********************************************************
#-----------------------IMPORTS---------------------------
#*********************************************************
import numpy as np
from cv2 import *
import cv2
import time as t
from matplotlib import pyplot as plt
import statistics

#*********************************************************
#*************** plotting histograms *********************
#*********************************************************
def getZaHistograms(img, out):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    hist = hist / np.max(hist)
    cumulative = np.cumsum(hist)
    cumulative = cumulative / np.max(cumulative)
    y_pos = np.arange(len(hist))
    fig, ax = plt.subplots(figsize=(10.24, 5.12))
    ax.set_facecolor("black")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.bar(range(0, 256),  hist, width=0.5, color="white", alpha=0.7)
    ax.bar(range(0, 256), cumulative, width=0.5, color="white", alpha=0.5)
    plt.savefig(out, pad_inches=-0.1, bbox_inches='tight')


def zaMedianFilter(source, EnterList):
    final = source[:]
    for y in range(len(source)):
        for x in range(y):
            final[y, x] = source[y, x]

    members = [source[0, 0]] * 9
    for y in range(1, len(source) - 1):
        for x in range(1, source.shape[1] - 1):
            if source[y][x] in EnterList:
                members[0] = source[y - 1, x - 1]
                members[1] = source[y, x - 1]
                members[2] = source[y + 1, x - 1]
                members[3] = source[y - 1, x]
                members[4] = source[y, x]
                members[5] = source[y + 1, x]
                members[6] = source[y - 1, x + 1]
                members[7] = source[y, x + 1]
                members[8] = source[y + 1, x + 1]
                members.sort()
                final[y, x] = members[4]
    return final


def zaMedianFilterStupid(source):
    final = source[:]
    for y in range(len(source)):
        for x in range(y):
            final[y, x] = source[y, x]

    members = [source[0, 0]] * 9
    for y in range(1, len(source) - 1):
        for x in range(1, source.shape[1] - 1):
            members[0] = source[y - 1, x - 1]
            members[1] = source[y, x - 1]
            members[2] = source[y + 1, x - 1]
            members[3] = source[y - 1, x]
            members[4] = source[y, x]
            members[5] = source[y + 1, x]
            members[6] = source[y - 1, x + 1]
            members[7] = source[y, x + 1]
            members[8] = source[y + 1, x + 1]
            members.sort()
            final[y, x] = members[4]
    return final

#---------------------cameraman.png ----------------------
img = cv2.imread('images/cameraman.png')
getZaHistograms(img, "cameraman_out")
#---------------------------------------------------------
#---------------------bat.png ----------------------------
#---------------------------------------------------------
img = cv2.imread('images/bat.png')
getZaHistograms(img, "bat_out")
#---------------------------------------------------------
#---------------------fog.png ----------------------------
#---------------------------------------------------------
img = cv2.imread('images/fog.png')
getZaHistograms(img, "out_fog.png")
#---------------------------------------------------------
#---------------------fognoise.png -----------------------
#---------------------------------------------------------
img = cv2.imread('images/fognoise.png')  # load rgb image
getZaHistograms(img, "out_fognoise_histo.png")
#*********************************************************
#*****************Mean ​​Verses ​Gaussian *******************
#*********************************************************
img = cv2.imread('images/cameraman.png')  # load rgb image
mean = cv2.blur(img, (5, 5))
gaussian = cv2.GaussianBlur(img, (5, 5), 0)
getZaHistograms(mean, "cameraman_mean_histo.png")
getZaHistograms(gaussian, "cameraman_gaussian_histo.png")


#*********************************************************
#*****************Selective ​ ​Median ​ ​Filter **************
#*********************************************************
img = cv2.imread('images/fognoise.png', 0)
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = np.hstack(hist)
histMedian = statistics.median(hist)
limit = histMedian - 300
EnterList = []
for i in range(len(hist)):
    if(hist[i] > 2170):
        EnterList.append(i)

img1 = img
t1 = t.time()
final1 = zaMedianFilter(img, EnterList)
t2 = t.time()
final2 = zaMedianFilterStupid(img1)
t3 = t.time()

print(" RunTime for first method : ", t2 - t1)
print(" RunTime for Second method : ", t3 - t2)

#*********************************************************
#***Contrast ​ ​Stretching ​ ​and ​ ​Histogram ​ ​Equalization ***
#*********************************************************
img = cv2.imread('images/frostfog.png')
hist = cv2.calcHist([img], [0], None, [256], [0, 256])
hist = np.hstack(hist)
a=0
b =255
d = np.amax(img)
c = np.amin(img)
img1 = img
for i in range(len(img)):
    for j in range(len(img[i])):
        temp1 =((img[i][j])+(-c))
        temp2 = (255/(d-c))
        temp3 = (temp1* temp2)
        img1[i][j] = temp3

img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
equ = cv2.equalizeHist(img)


cv2.imwrite("forest_equalized.png", equ)
cv2.imwrite("forest_Stretched.png", img1)

getZaHistograms(equ, "forest_equalized_Histo.png")
getZaHistograms(img1, "forest_Stretched_Histo.png")
