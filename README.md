# XYZframe
XYZframe is a Python class that extends a Pandas DataFrame to work with 3D point cloud data. It includes methods for manipulating and transforming point clouds, as well as calculating derived data such as nearest neighbor distances and cubic splines.

XYZframes have two "memory" attributes: a Pandas Dataframe (containing point data and any point-wise data columns), and a coordinate system object. The coordinate system "remembers" affine transormations on the point data.

Dataframe special methods (such as .loc or indexing) pass directly to the XYZframe's DataFrame attribute and return a new dataframe. The new dataframe can be assigned to the .df attribute of a new XYZframe to create a filtered XYZframe, or etc. Try it out!

# Installation
To install XYZframe, simply clone or download this repository and save the xyzframe.py file to your site-packages or project directory.

# Usage
import xyz

### Initialize an empty XYZframe object
xf = xyz.xfgen()

### Load point cloud data from a CSV file
xf = xyz.xfgen('my_data.csv')

### Create an XYZframe object from a list of points
xf = xyz.xfgen([(0,0,0), (1,0,0), (0,1,0)])

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
