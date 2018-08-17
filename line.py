import numpy as np

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None
        # was the line detected in the last iteration?
        self.detected = False
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #distance in meters of vehicle center from the line
        self.line_base_pos = None

        # x values of the last n fits of the line
        self.recent_xfitted = []
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None

        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
