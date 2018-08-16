import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import perspective
from matplotlib.patches import Rectangle
import cv2

draw_histo = False
draw_sliding = False
draw_result = False
################################################################################
#                              TUNABLE PARAMETERS                              #
################################################################################
# Choose the number of sliding windows
nwindows = 9
# Set the width of the windows +/- margin
margin = 30
# Set minimum number of pixels found to recenter window
minpix = 50

# meters per pixel in x/y dimension
ym_per_pix = 30/720
xm_per_pix = 3.7/700

car_center = 1280/2

left_color = 'red'
right_color = 'green'


def draw_histogram(image, histogram, fname):
    f, axes = plt.subplots(2, 1, figsize=(24, 9))
    axes[0].imshow(image)
    axes[1].plot(histogram)
    plt.savefig('histo/' + fname + '.png')


def draw_sliding_rectangles(win_xleft_low, win_y_low, window_height, ax):
  rect = Rectangle((win_xleft_low, win_y_low),
                           2 * margin, window_height,
                           linewidth=1, edgecolor=left_color, facecolor='none')
  ax[0].add_patch(rect)
  rect = Rectangle((win_xright_low,win_y_low),
                           2 * margin, window_height,
                           linewidth=1, edgecolor=right_color, facecolor='none')
  ax[0].add_patch(rect)


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
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []
    if (draw_sliding):
      f, ax = plt.subplots(2, 1, figsize=(9, 9))
    else:
      ax = None
    for window in range(nwindows):
        win_y_low = image.shape[0] - (window+1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        if (draw_sliding):
          draw_sliding_rectangles(win_xleft_low, win_y_low, window_height, ax)
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)
                          ).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)
                           ).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    return leftx, lefty, rightx, righty, ax


def draw_on_undistored(left_fitx, right_fitx, ploty, image, undist, fname,
                       distance_off_center, left_rad, right_rad):
  newwarp = invert_perspective_trafo(left_fitx, right_fitx, ploty, image)
  undist = cv2.cvtColor(undist, cv2.COLOR_BGR2RGB)
  result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
  draw_text_info(result, left_rad, right_rad, distance_off_center)
  return result


def invert_perspective_trafo(left_fitx, right_fitx, ploty, image):
  warp_zero = np.zeros_like(image).astype(np.uint8)
  color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
  pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
  pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
  pts = np.hstack((pts_left, pts_right))
  cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
  return cv2.warpPerspective(color_warp, perspective.getMinv(),
                                (image.shape[1], image.shape[0]))


def draw_text_info(result, left_rad, right_rad, distance_off_center):
  radius = (left_rad+right_rad) / 2
  if radius < 3000:
    radius_text = "Radius: {:6.2f} m".format(radius)
  else:
    radius_text = "Radius: straight"
  center_text = "Distance off-center: {:6.2f} m".format(distance_off_center)
  cv2.putText(result, center_text, (100, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (256, 256, 256))
  cv2.putText(result, radius_text, (100, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (256, 256, 256))


def measure_curvature(y_eval, left_fit_coeff, right_fit_coeff):
  left_curverad = ((1 + (2*left_fit_coeff[0]*y_eval + left_fit_coeff[1])**2)**1.5) / \
    np.absolute(2*left_fit_coeff[0])
  right_curverad = ((1 + (2*right_fit_coeff[0]*y_eval + right_fit_coeff[1])**2)**1.5) / \
    np.absolute(2*right_fit_coeff[0])
  return left_curverad, right_curverad


def fit_and_draw_on_undistorted(image, undist, fname, lines):
  leftx, lefty, rightx, righty, ax = find_lane_pixels(image, fname)

  left_fit_coeff, left_residuals, _, _, _ = np.polyfit(lefty, leftx, 2, full=True)
  right_fit_coeff, right_residuals, _, _, _ = np.polyfit(righty, rightx, 2, full=True)

  ploty = np.linspace(0, image.shape[0]-1, image.shape[0])
  try:
    left_fitx = left_fit_coeff[0]*ploty**2 + left_fit_coeff[1]*ploty + left_fit_coeff[2]
    right_fitx = right_fit_coeff[0]*ploty**2 + right_fit_coeff[1]*ploty + right_fit_coeff[2]
  except TypeError:
    print('The function failed to fit a line!')
    left_fitx = 1*ploty**2 + 1*ploty
    right_fitx = 1*ploty**2 + 1*ploty

  left_fit_coeff_m = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
  right_coeff_m = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
  y_eval = np.max(ploty) * ym_per_pix
  left_rad, right_rad = measure_curvature(y_eval, left_fit_coeff_m, right_coeff_m)
  line_center = (left_fitx[np.max(ploty)] + right_fitx[np.max(ploty)]) / 2
  distance_off_center = (car_center - line_center) * xm_per_pix

  result = draw_on_undistored(left_fitx, right_fitx, ploty, image, undist, fname,
                       distance_off_center, left_rad, right_rad)

  if (draw_sliding):
    ax[0].imshow(image)
    ax[0].plot(left_fitx, ploty, color=left_color)
    ax[0].plot(right_fitx, ploty, color=right_color)
    result_for_plt = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    ax[1].imshow(result_for_plt)
    plt.savefig('sliding/' + fname + '.png')
  if (draw_result):
    f, ax = plt.subplots(1, 1, figsize=(16, 9))
    plt.imshow(result)
    plt.savefig('result/' + fname + '.png')
  return result, lines


