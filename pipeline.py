def main():
  undistort()
  perspective_transform()
  null_out_edges() # (select trapezoidal region)
  thresholding()
  lane_detect()
  inverse_perspective_transform_and_plot()
