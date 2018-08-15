# Starting point is the one mentioned in the lecture
# https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle


draw = False


# Step through the list and search for chessboard corners
def search_chessboard_corners_in(images):
  board_y = 6
  board_x = 9
  # Arrays to store object points and image points from all the images.
  objpoints = [] # 3d points in real world space
  imgpoints = [] # 2d points in image plane.
  total = 0
  success = 0
  objp = np.zeros((board_y*board_x,3), np.float32)
  objp[:,:2] = np.mgrid[0:board_x, 0:board_y].T.reshape(-1,2)
  for idx, fname in enumerate(images):
      img = cv2.imread(fname)
      gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
      # Find the chessboard corners
      ret, corners = cv2.findChessboardCorners(gray, (board_x,board_y), None)
      total += 1
      if ret:  # If found, add object points, image points
          success += 1
          objpoints.append(objp)
          imgpoints.append(corners)
          if draw:  # Draw and display the corners
              cv2.drawChessboardCorners(img, (board_x,board_y), corners, ret)
              write_name = 'camera_cal/corners_found_' + str(idx) + '.jpg'
              cv2.imwrite(write_name, img)
              cv2.imshow('img', img)
              cv2.waitKey(500)
  objpoints = np.array(objpoints)
  imgpoints = np.array(imgpoints)
  cv2.destroyAllWindows()
  print("Could use " + str(success) + "/" + str(total) + " pictures (" +
        str((100.0*success)/total) + "%)")
  return objpoints, imgpoints


def save_calibration_result(mtx, dist):
  dist_pickle = {}
  dist_pickle["mtx"] = mtx
  dist_pickle["dist"] = dist
  pickle.dump( dist_pickle, open( "camera_cal/calibration_pickle.p", "wb" ) )


def visualize_undistortion(img, mtx, dist):
  dst = cv2.undistort(img, mtx, dist, None, mtx)
  cv2.imwrite('camera_cal/test_undist.jpg',dst)
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
  ax1.imshow(img)
  ax1.set_title('Original Image', fontsize=30)
  ax2.imshow(dst)
  ax2.set_title('Undistorted Image', fontsize=30)
  plt.tight_layout()
  plt.savefig('camera_cal/test_undist_comparison.png')


def calibrate(objpoints, imgpoints):
  test_img = cv2.imread('camera_cal/calibration1.jpg')
  img_size = (test_img.shape[1], test_img.shape[0])
  ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
  save_calibration_result(mtx, dist)
  visualize_undistortion(test_img, mtx, dist)


def main():
  images = glob.glob('camera_cal/calibration*.jpg')
  objpoints, imgpoints = search_chessboard_corners_in(images)
  calibrate(objpoints, imgpoints)

main()
