# Starting point is the lecture
# Extra inspiration from https://towardsdatascience.com/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3

import numpy as np
import cv2
import glob
from drawing import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle


pickle_file_name = "camera_cal/calibration_pickle.p"
draw_perspective_transform = True


def undistort(image):
  calibration = pickle.load( open( pickle_file_name, "rb" ) )
  mtx = calibration["mtx"]
  dist = calibration["dist"]
  return cv2.undistort(image, mtx, dist, None, mtx)


def perspective_transform(image, fname):
  src = np.float32([
    [595,450],
    [689,450],
    [970,630],
    [344,630]
  ])
  dst = np.float32([
    [470,450],
    [830,450],
    [830,680],
    [470,680]
  ])
  M = cv2.getPerspectiveTransform(src, dst)
  img_size = (image.shape[1], image.shape[0])
  top_down = cv2.warpPerspective(image, M, img_size)
  if (draw_perspective_transform):
    draw_perspective_trafo(image, top_down, src, dst, fname)
  return top_down


def color_mask(image):
  # thresholds for white and yellow too strict?
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  yellow_hsv_low  = np.array([0, 80, 200])
  yellow_hsv_high = np.array([40, 255, 255])
  yellow_mask = cv2.inRange(hsv, yellow_hsv_low, yellow_hsv_high)
  white_hsv_low  = np.array([  20,   0,   200])
  white_hsv_high = np.array([ 255,  80, 255])
  white_mask = cv2.inRange(hsv, white_hsv_low, white_hsv_high)
  mask = cv2.bitwise_or(yellow_mask, white_mask)
  res = cv2.bitwise_and(image, image, mask=mask)
  return res, mask


def sobel(image):
  s_thresh = (170, 255)
  sx_thresh = (20, 100)
  hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  l_channel = hls[:,:,1]
  s_channel = hls[:,:,2]
  # Sobel x
  sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
  abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
  scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
  # Threshold x gradient
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
  # Threshold color channel
  #  s_binary = np.zeros_like(s_channel)
  #  s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

  foo, mask = color_mask(image)
  combined_mask = cv2.bitwise_or(sxbinary, mask)

  # Stack each channel
  #  color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, mask)) * 255
  #  combined_binary = np.zeros_like(sxbinary)
  #  combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
  #  combined_binary = combined_binary * 255
  res = cv2.bitwise_and(image, image, mask=combined_mask)
  return res


def null_out_edges(image):
  print(image.shape)
  cut_at_top = 20
  cut_at_bottom = 40
  top = 0 + cut_at_top
  bot = 720 - cut_at_bottom
  relevant = image[top:bot, 400:1000]
  new = np.zeros_like(image)
  new[top:bot,400:1000] = relevant
  assert(new.shape == image.shape)
  return new


def pipeline(image, fname):
  images = [image]
  image = undistort(image)
  images.append(image)
  image = perspective_transform(image, fname)
  images.append(image)
  image = sobel(image)
  #  image = color_mask(image)
  image = null_out_edges(image)
  images.append(image)

  images = map(lambda x: cv2.cvtColor(x, cv2.COLOR_BGR2RGB), images)
  f, axes = plt.subplots(2, 2, figsize=(24, 9))
  f.tight_layout()
  [a.imshow(i) for a, i in zip(axes.flatten(), images)]
  plt.savefig('processing/' + fname + '.png')

  #  lane_detect()
  #  inverse_perspective_transform_and_plot()
  return image


def main():
  images = glob.glob('test_images/*.jpg')
  for fname in images:
    image = cv2.imread(fname)
    cv2.imshow('img', image)
    image = pipeline(image, fname)
    cv2.imshow('img', image)
    cv2.waitKey(2000)

main()
