# Starting point is the lecture
# Some additional inspiration is from https://towardsdatascience.com/robust-lane-finding-using-advanced-computer-vision-techniques-mid-project-update-540387e95ed3

import cv2
import glob
import time
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from moviepy.editor import VideoFileClip

import perspective
import sliding_window
from line import Line

################################################################################
#                              STEERING VARIABLES                              #
################################################################################
test_on_test_images = True

pickle_file_name = "camera_cal/calibration_pickle.p"

draw_distortion = False
draw_threshold = False
draw_perspective_transform = True
draw_overview = True

################################################################################
#                              TUNABLE PARAMETERS                              #
################################################################################
# white-yellow selection
yellow_hsv_low  = np.array([0, 80, 200])
yellow_hsv_high = np.array([40, 255, 255])
sensitivity = 30
white_hsv_low  = np.array([  0,   0, 255 - sensitivity])
white_hsv_high = np.array([ 255,  sensitivity, 255])

# Sobel
sobel_threshold = (20, 100)

# cutting
cut_at_top = 0
cut_at_bottom = 40
cut_left = 300
cut_right = 300

################################################################################

calibration = pickle.load( open( pickle_file_name, "rb" ) )
def undistort(image):
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
  image = threshold(image)
  images.append(image)
  image = perspective_transform(image, fname)
  image = null_out_edges(image)
  images.append(image)
  image, lines = sliding_window.fit_and_draw_on_undistorted(image,
                                images[1], fname, lines)

  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  #  images = map(convert_if_possible, images)
  if (draw_distortion):
    draw(images[0:2], 'distortion/' + fname)
  if (draw_threshold):
    draw(images[1:3], 'threshold/' + fname)
  if (draw_overview):
    draw(images, 'processing/' + fname)

  cv2.imshow('img', image)
  cv2.waitKey(1)
  return image, lines


def draw(images, suffix):
  if len(images) == 2:
    f_axes = plt.subplots(2, 1, figsize=(16, 18))
  else:
    f_axes = plt.subplots(2, 2, figsize=(16, 9))
  f_axes[0].tight_layout()
  [a.imshow(i) for a, i in zip(f_axes[1].flatten(), images)]
  plt.savefig(suffix + '.png')


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
      image, _ = pipeline(image, fname=fname, lines=[Line(), Line()])
  else:
    clip1 = VideoFileClip("project_video.mp4")
    clip = clip1.fl_image(pipeline_with_line)
    clip.write_videofile("project_video_out.mp4", audio=False)

main()
