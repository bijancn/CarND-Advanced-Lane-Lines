import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection

# perspective transform
perspective_source_points = np.float32([
  [595,450],
  [689,450],
  [970,630],
  [344,630]
])
perspective_destination_points = np.float32([
  [470,50],
  [830,50],
  [830,680],
  [470,680]
])
M = cv2.getPerspectiveTransform(perspective_source_points,
                                perspective_destination_points)
Minv = cv2.getPerspectiveTransform(perspective_destination_points,
                                   perspective_source_points)


def getM():
  return M


def getMinv():
  return Minv


def draw(image, top_down, fname):
  def set_ax(ax, image, points):
    ax.imshow(image)
    for point in points:
      ax.plot(point[0], point[1], '.', color="yellow", markersize=20)
    ax.add_collection(PatchCollection([Polygon(points)], color="yellow", alpha=0.3))
  f, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 18))
  f.tight_layout()
  set_ax(ax1, image, perspective_source_points)
  set_ax(ax2, top_down, perspective_destination_points)
  plt.savefig('warped/' + fname + '.png')
