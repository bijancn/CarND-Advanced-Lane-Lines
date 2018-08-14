# Starting point is the one mentioned in the lecture
# https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

DRAW = False

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
total = 0
success = 0
for idx, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)
    print('ret = ', ret)   #   Debugging

    # If found, add object points, image points
    total += 1
    if ret:
        success += 1
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        if DRAW:
            cv2.drawChessboardCorners(img, (9,6), corners, ret)
            write_name = 'camera_cal/corners_found_' + str(idx) + '.jpg'
            cv2.imwrite(write_name, img)
            cv2.imshow('img', img)
            cv2.waitKey(500)

cv2.destroyAllWindows()
print("Could use " + str(success) + "/" + str(total) + " pictures (" +
      str((100.0*success)/total) + "%)")

# Test undistortion on an image
img = cv2.imread('camera_cal/calibration3.jpg')
img_size = (img.shape[1], img.shape[0])

cv2.imshow("foo", img)
cv2.waitKey(500)

# Do camera calibration given object points and image points
objpoints = np.array(objpoints)
imgpoints = np.array(imgpoints)
print(objpoints.shape)
print(imgpoints.shape)
print(img_size)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

dst = cv2.undistort(img, mtx, dist, None, mtx)
cv2.imwrite('camera_cal/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/calibration_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
ax1.imshow(img)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(dst)
ax2.set_title('Undistorted Image', fontsize=30)
