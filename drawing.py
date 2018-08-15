import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection


def draw_perspective_trafo(image, top_down, src, dst, fname):
  def set_ax(ax, image, title, points):
    ax.imshow(image)
    ax.set_title(title, fontsize=30)
    for point in points:
      ax.plot(point[0], point[1], '.', color="red", markersize=20)
    ax.add_collection(PatchCollection([Polygon(points)], color="red", alpha=0.3))
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
  top_down = cv2.cvtColor(top_down, cv2.COLOR_BGR2RGB)
  f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
  f.tight_layout()
  set_ax(ax1, image, 'Original Image', src)
  set_ax(ax2, top_down, 'Undistorted and Warped Image', dst)
  plt.savefig('warped/' + fname + '.png')
