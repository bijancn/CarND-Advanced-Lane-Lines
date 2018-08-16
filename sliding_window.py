import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import perspective
from matplotlib.patches import Rectangle
import cv2

draw_histo = False
draw_sliding = True
draw_result = False
################################################################################
#                              TUNABLE PARAMETERS                              #
################################################################################
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 50
# Set minimum number of pixels found to recenter window
minpix = 50

# meters per pixel in x/y dimension
ym_per_pix = 30/720
xm_per_pix = 3.7/700

left_color = 'red'
right_color = 'green'


def draw_histogram(image, histogram, fname):
    f, axes = plt.subplots(2, 1, figsize=(24, 9))
    axes[0].imshow(image)
    axes[1].plot(histogram)
    plt.savefig('histo/' + fname + '.png')


def find_lane_pixels(image, fname):
    histogram = np.sum(image[image.shape[0]//2:,:], axis=0)
    midpoint = np.int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    if (draw_histo):
      draw_histogram(image, histogram, fname)

    window_height = np.int(image.shape[0]//nwindows)
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated later for each window in nwindows
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    if (draw_sliding):
      f, ax = plt.subplots(1, 1, figsize=(24, 9))
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        if (draw_sliding):
          rect = Rectangle((win_xleft_low,win_y_low),
                                   2 * margin,window_height,
                                   linewidth=1, edgecolor=left_color, facecolor='none')
          ax.add_patch(rect)
          rect = Rectangle((win_xright_low,win_y_low),
                                   2 * margin,window_height,
                                   linewidth=1, edgecolor=right_color, facecolor='none')
          ax.add_patch(rect)

        # Identify the nonzero pixels in x and y within the window #
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    return leftx, lefty, rightx, righty


def fit_and_draw_on_undistorted(image, undist, fname, lines):
  leftx, lefty, rightx, righty = find_lane_pixels(image, fname)

  left_fit, left_residuals, _, _, _ = np.polyfit(lefty, leftx, 2, full=True)
  right_fit, right_residuals, _, _, _ = np.polyfit(righty, rightx, 2, full=True)

  ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
  try:
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
  except TypeError:
    print('The function failed to fit a line!')
    left_fitx = 1*ploty**2 + 1*ploty
    right_fitx = 1*ploty**2 + 1*ploty

  if (draw_sliding):
    plt.imshow(image)
    plt.plot(left_fitx, ploty, color=left_color)
    plt.plot(right_fitx, ploty, color=right_color)
    plt.savefig('sliding/' + fname + '.png')

  left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
  left_radius, right_radius = measure_curvature(ploty, left_fit_cr, right_fit_cr)

  result = draw_on_undistored(left_fitx, right_fitx, ploty, image, undist, fname)

  return result, lines

def draw_on_undistored(left_fitx, right_fitx, ploty, image, undist, fname):
  warp_zero = np.zeros_like(image).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
  f, ax = plt.subplots(1, 1, figsize=(24, 9))
  newwarp = cv2.warpPerspective(color_warp, perspective.getMinv(),
                                (image.shape[1], image.shape[0]))
  undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
  result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
  if (draw_result):
    plt.imshow(result)
    plt.savefig('result/' + fname + '.png')
  return result


def measure_curvature(ploty, left_fit, right_fit):
  y_eval = np.max(ploty) * ym_per_pix
  left_curverad = ((1 + (2*left_fit[0]*y_eval + left_fit[1])**2)**1.5) / \
    np.absolute(2*left_fit[0])
  right_curverad = ((1 + (2*right_fit[0]*y_eval + right_fit[1])**2)**1.5) / \
    np.absolute(2*right_fit[0])
  return left_curverad, right_curverad
