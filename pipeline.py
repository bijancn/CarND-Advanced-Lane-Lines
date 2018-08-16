# Starting point is the lecture
# Extra inspiration from https://towardsdatascience.com/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import perspective
from sliding_window import *

################################################################################
#                              STEERING VARIABLES                              #
################################################################################
pickle_file_name = "camera_cal/calibration_pickle.p"
draw_perspective_transform = True
draw_overview = True


################################################################################
#                              TUNABLE PARAMETERS                              #
################################################################################
# white-yellow selection
# TODO: too strict?
yellow_hsv_low  = np.array([0, 80, 200])
yellow_hsv_high = np.array([40, 255, 255])
white_hsv_low  = np.array([  20,   0,   200])
white_hsv_high = np.array([ 255,  80, 255])

# Sobel
sobel_threshold = (20, 100)

# cutting
cut_at_top = 0
cut_at_bottom = 40
cut_left = 350
cut_right = 250

################################################################################

def undistort(image):
  calibration = pickle.load( open( pickle_file_name, "rb" ) )
  mtx = calibration["mtx"]
  dist = calibration["dist"]
  return cv2.undistort(image, mtx, dist, None, mtx)


def perspective_transform(image, fname):
  img_size = (image.shape[1], image.shape[0])
  top_down = cv2.warpPerspective(image, perspective.getM(), img_size)
  if (draw_perspective_transform):
    perspective.draw(image, top_down, fname)
  return top_down


def color_mask(image):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  yellow_mask = cv2.inRange(hsv, yellow_hsv_low, yellow_hsv_high)
  white_mask = cv2.inRange(hsv, white_hsv_low, white_hsv_high)
  mask = cv2.bitwise_or(yellow_mask, white_mask)
  return mask


def sobel(image):
  hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
  l_channel = hls[:,:,1]
  sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
  abs_sobelx = np.absolute(sobelx)
  scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
  sxbinary = np.zeros_like(scaled_sobel)
  sxbinary[(scaled_sobel >= sobel_threshold[0]) & (scaled_sobel <= sobel_threshold[1])] = 1
  return sxbinary


def threshold(image):
  combined_mask = cv2.bitwise_or(sobel(image), color_mask(image))
  #  res = cv2.bitwise_and(image, image, mask=combined_mask)
  combined_mask = combined_mask / 255.0
  combined_mask[np.nonzero(combined_mask)] = 1
  return combined_mask


def null_out_edges(image):
  top = 0 + cut_at_top
  bot = 720 - cut_at_bottom
  left = 0 + cut_left
  right = 1280 - cut_right
  relevant = image[top:bot, left:right]
  new = np.zeros_like(image)
  new[top:bot,left:right] = relevant
  assert(new.shape == image.shape)
  return new


def pipeline(image, fname):
  images = [image]
  image = undistort(image)
  images.append(image)
  image = perspective_transform(image, fname)
  images.append(image)
  image = threshold(image)
  image = null_out_edges(image)
  images.append(image)
  image = fit_polynomial(image, images[0], fname)
  #  inverse_perspective_transform_and_plot()

  if (draw_overview):
    images = map(convert_if_possible, images)
    f, axes = plt.subplots(2, 2, figsize=(24, 9))
    f.tight_layout()
    [a.imshow(i) for a, i in zip(axes.flatten(), images)]
    plt.savefig('processing/' + fname + '.png')

  return image


def convert_if_possible(img):
  try:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  except:
    return img

def main():
  images = glob.glob('test_images/*.jpg')
  for fname in images:
    image = cv2.imread(fname)
    cv2.imshow('img', image)
    image = pipeline(image, fname)
    cv2.imshow('img', image)
    cv2.waitKey(2000)

main()
