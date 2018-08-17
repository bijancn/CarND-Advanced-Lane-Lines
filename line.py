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
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #number of unsane fits in a row
        self.nr_of_unsane = 0
