import numpy as np
from utils import FloatArray


def load_center_line(filename: str) -> tuple[FloatArray, FloatArray]:
  """
  Loads the center line stored in the CSV file at filename.
  Expected format: X,Y,right_width,left_width
  Returns center line as (N, 2) array and track widths as (N, 2) array.
  """
  arr = np.genfromtxt(filename, delimiter=",", dtype=float, skip_header=1)
  return arr[:, :2], arr[:, 2:]


def load_cones(filename: str) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
  """
  Loads cones from the CSV file at filename.
  Expected format: cone_type,X,Y,Z,std_X,std_Y,std_Z,right,left
  Returns (blue, yellow, big_orange, small_orange, right, left) cone arrays.
  """
  arr = np.genfromtxt(filename, delimiter=",", dtype=str, skip_header=1)
  blue_cones = arr[arr[:, 0] == "blue"][:, 1:3].astype(float)
  yellow_cones = arr[arr[:, 0] == "yellow"][:, 1:3].astype(float)
  big_orange_cones = arr[arr[:, 0] == "big_orange"][:, 1:3].astype(float)
  small_orange_cones = arr[arr[:, 0] == "small_orange"][:, 1:3].astype(float)
  right_cones = arr[arr[:, 7] == "1"][:, 1:3].astype(float)
  left_cones = arr[arr[:, 8] == "1"][:, 1:3].astype(float)
  return blue_cones, yellow_cones, big_orange_cones, small_orange_cones, right_cones, left_cones
