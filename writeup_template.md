# Advanced lane finding project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients
  given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary
  image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation
  of lane curvature and vehicle position.

[//]: # (Image References)

[image0]: ./camera_cal/corners_found_1.jpg.png "Chessboard"
[image1]: ./camera_cal/test_undist_comparison.png "Undistorted"
[image2]: ./distortion/test_images/test4.jpg.png "Road Transformed"
[image3]: ./threshold/test_images/test1.jpg.png "Binary Example"
[image4a]: ./warped/test_images/straight_lines1.jpg.png "Warp Example"
[image4b]: ./warped/test_images/test3.jpg.png "Warp Example"
[image5]: ./histo/test_images/test4.jpg.png "Histogram Example"
[image6]: ./sliding/test_images/test1.jpg.png "Windows Example"
[image7]: ./result/test_images/test2.jpg.png "Result Example"
[image8]: ./result/test_images/test3.jpg.png "Result Example"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Writeup / README

This write up contains the main discussion. Instructions on how to run
the code are in the [README](README.md) and some random thoughts during
developments in the [lab journal](lab_jounral.md).

### Camera calibration: camera matrix and distortion coefficients

The calibration can be found in `calibration.py`.

Just as in the lecture, corners are found with the help of
`cv2.findChessboardCorners` in the `search_chessboard_corners_in`
function. This worked in 17 of the 20 pictures and can be visualized
like this

![alt text][image0]

The other three pictures were not used.  The found object points (in 3D
world while assuming z=0 approximately) and image points (the corners in
2D found by OpenCV) are passed on `cv2.calibrateCamera`. The obtained
calibration is saved to disk. The distortion correction is tested on
one of the chessboard images to obtain this result:

![alt text][image1]

The calibration clearly removes significant amounts of distortion of the
image, especially towards the edges of image.

### Pipeline (single images)

The pipeline can be found as `pipeline` function from image to image in
`pipeline.py`.

#### Distortion correction

The first step of the pipeline is to apply the distortion correction of
the last section. The results are loaded from disk and used in
`cv2.undistort` to obtain this result

![alt text][image2]

where the original image is on top and the undistorted on bottom as can
be seen easily by looking at the white car.

#### Color and gradient thresholds

I used a combination of color and gradient thresholds to generate a
binary image in the `threshold` function in `pipeline.py`. Specifically,
I combined them with a bitwise or. The `color_mask` itself looks for
white and yellow in the HSV color space and combines both filters with a
bitwise or as well. The gradient threshold considers the HLS color space
and looks for changes in `x` direction (the lines are mostly along the
`y` direction) in the L channel. This gives the following picture

![alt text][image3]

In my rough tuning, I looked to detect white and yellow reliably in any
lighting condition while not picking up too much noise.

#### Perspective transform

The `perspective_transform` function of the `pipeline` applies
`cv2.warpPerspective` given the linear transformation matrix in
`perspective.py`. To obtain it, I selected the following points along an
trapezoid in one of the straight images with an image viewer:

```
perspective_source_points = np.float32([
  [595,450],
  [689,450],
  [970,630],
  [344,630]
])
```

To transform this perspective into an rectangle that nicely fills the
image, I chose the following desired destination points
```
perspective_destination_points = np.float32([
  [470,50],
  [830,50],
  [830,680],
  [470,680]
])
```
These eight points are enough to define the transformation matrix,
which is obtained with `cv2.getPerspectiveTransform`. I tested that the
transformation lets the lines in all test images appear as parallel:

![alt text][image4a]
![alt text][image4b]

#### Edge removal

To reduce the amount of unneeded information, I nulled out the edges of
the image with the `null_out_edges` function. Hereby, I took away 300
pixels from the sides (to avoid detecting parallel lanes or road
boundaries) and 40 pixels from the bottom (where the hood can lead to
irritating results).

#### Polynomial fit

All steps toward the polynomial fit are in `sliding_window.py` where the
entry point is the `fit_and_draw_on_undistorted` function. The first
step is to find the lane pixels. For this, at first a histogram is
created where the bottom half of the y dimension is integrated out, to
obtain the most likely starting position of the lane. This works for all
test images nicely even in the presence of other potential lines:

![alt text][image5]

Next, we step through the image and collect the pixels that belong to
the lane, i.e. the sliding window with a width of 100 pixels here,
recentering the window when at least 50 new pixels have been found.

![alt text][image6]

All of the pixels that are contained in one of the nine windows are
considered for the polynomial fit and saved to the `allx` and `ally`
properties of the left and right `lines`, which is the output of
`find_lane_pixels`. The fit is performed in `construct_fit` with
`np.polyfit` to a 2nd degree polynomial.

#### Radius of curvature of the lane the position of the vehicle

The radius and lane position is computed in `measure_radius_and_center`.
This is simply done by using the radius formulae provided in the lecture
with the following factors to roughly convert pixels into meters
```
ym_per_pix = 30/720
xm_per_pix = 3.7/700
```
For the distance off-center, I simply compared the middle of the screen
(1280/2) with the center of the left and right line fit position at the
bottom of the screen. While the precision of the conversion factors is
questionable, it yields the right order of magnitude for both radius and
distance for all test images:

![alt text][image7]

#### Transform back to undistorted image space

To visualize the obtained lanes, I transform in the function
`draw_on_undistorted` (and the `invert_perspective_trafo` therein) the
obtained fit back to the original undistorted image space by applying
the inverse perspective transform. This can be then drawn with
`cv2.fillPoly` on top of the undistorted image.

![alt text][image8]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  


I did not thoroughly fine tune the color and gradient threshold
parameters, which can likekly be improved.
