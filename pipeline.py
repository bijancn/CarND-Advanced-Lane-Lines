# Starting point is the lecture
# Extra inspiration from https://towardsdatascience.com/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3

import numpy as np
import cv2
import glob
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import perspective
from line import Line
from moviepy.editor import VideoFileClip
from sliding_window import *

################################################################################
#                              STEERING VARIABLES                              #
################################################################################
pickle_file_name = "camera_cal/calibration_pickle.p"
draw_perspective_transform = False
draw_overview = False
test_on_test_images = False

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
  return new


def pipeline(image, fname='foo', lines=[]):
  images = [image]
  image = undistort(image)
  images.append(image)
  image = perspective_transform(image, fname)
  images.append(image)
  image = threshold(image)
  image = null_out_edges(image)
  images.append(image)
  image, lines = fit_and_draw_on_undistorted(image, images[1], fname, lines)

  if (draw_overview):
    images = map(convert_if_possible, images)
    f, axes = plt.subplots(2, 2, figsize=(24, 9))
    f.tight_layout()
    [a.imshow(i) for a, i in zip(axes.flatten(), images)]
    plt.savefig('processing/' + fname + '.png')

  cv2.imshow('img', image)
  cv2.waitKey(1)
  return image, lines


def convert_if_possible(img):
  try:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  except:
    return img


lines = [Line(), Line()]
nr = 0
def pipeline_with_line(image):
  global lines
  global nr
  image, lines = pipeline(image, fname='video_' + str(nr), lines=lines)
  nr += 1
  return image


def main():
  if (test_on_test_images):
    images = glob.glob('test_images/*.jpg')
    for fname in images:
      image = cv2.imread(fname)
      image, _ = pipeline(image, fname=fname)
  else:
    clip1 = VideoFileClip("project_video.mp4") # .subclip(21,23)
    clip = clip1.fl_image(pipeline_with_line)
    clip.write_videofile("project_video_out.mp4", audio=False)

main()
