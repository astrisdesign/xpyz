# XYZframe
XYZframe is a Python class that provides a convenient way to work with 3D point cloud data. It includes methods for manipulating and transforming point clouds, as well as calculating derived data such as nearest neighbor distances and cubic splines.

# Installation
To install XYZframe, simply clone or download this repository and save the xyzframe.py file to your site-packages or project directory.

# Usage
Creating an XYZframe object
from xyz import XYZframe

### Initialize an empty XYZframe object
xf = XYZframe()

### Load point cloud data from a CSV file
xf = XYZframe.from_csv('my_data.csv')

### Create an XYZframe object from a list of points
xf = XYZframe.from_point_list([(0,0,0), (1,0,0), (0,1,0)])

### Translate point cloud by a given vector
xf.translate((1,1,1))

### Rotate point cloud around a given axis by a given angle in degrees
xf.rotate(45, (1,0,0))

### Scale point cloud by a given factor
xf.scale(2)

### Calculate nearest neighbor distances for each point in the cloud
xf.nn_distances()

### Calculate root sum of squares for each point in the cloud
xf.rss()

### Calculate cubic splines for x, y, and z values of the point cloud
xf.cubicspline()

### Calculate normals for each point in the cloud based on a given source point
xf.normals_from_source((0,0,0))

### Plot the point cloud in 3D space
xf.plot()

### Apply a color map to the point cloud based on a given data vector
xf.plot(vector='nn_dist', cmap_name='viridis')

### Save a series of images to create an animation of the point cloud
xf.animate(rotation_degrees=2, interval=50, save_path='my_animation.gif')

# Dependencies
NumPy
Pandas
SciPy
Matplotlib
tqdm
imageio

# License
This project is licensed under the Unlicense. See the LICENSE file for details.
