import numpy as np
import cv2 as cv
import argparse

videofile = "10X_LD_1024_R1.avi"

cap = cv.VideoCapture(videofile)

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

import matplotlib.pyplot as plt

#edges = cv.Canny(old_gray,100,200)
#plt.imshow(edges);plt.show()

#circles = cv.HoughCircles(old_gray,cv.HOUGH_GRADIENT,1,1,
# param1=50,param2=30,minRadius=0,maxRadius=5)

#cimg = cv.cvtColor(old_gray,cv.COLOR_GRAY2RGB)
#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
 # draw the outer circle
# cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
 # draw the center of the circle
# cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

#plt.imshow(cimg);plt.show()


_, old_bw = cv.threshold(old_gray,50,255,cv.THRESH_BINARY)

#vis = np.copy(old_frame)
#vis[:,:,2] = old_bw
#plt.imshow(vis);plt.show()

#plt.imshow(old_gray,cmap="gray");plt.show()
kernel = np.ones((3,3),np.uint8)
old_bw = cv.morphologyEx(old_bw, cv.MORPH_OPEN, kernel)
#plt.imshow(old_bw,cmap="gray");plt.show()

#vis = np.copy(old_frame)
#vis[:,:,2] = old_bw
#plt.imshow(vis);plt.show()

_, _, _, centroid = cv.connectedComponentsWithStats(old_bw, 4, cv.CV_32S) 
#(totalLabels, label_ids, values, centroid) = analysis 

#print(centroid.shape)
p0 = np.expand_dims(centroid,axis=1).astype(np.float32)
print(p0.shape)

#check = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

#print(p0)
#print(p0.dtype)
#print(check.shape)


result_vid = cv.VideoWriter("output.mp4",cv.VideoWriter_fourcc(*'MJPG'),10,(old_bw.shape[1],old_bw.shape[0]))

# Create some random colors
color = np.random.randint(0, 255, (p0.shape[0], 3))


# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]

    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
    img = cv.add(frame, mask)

    result_vid.write(img)

    cv.imshow('frame', img)
    k = cv.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

result_vid.release()

cv.destroyAllWindows()