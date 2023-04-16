'''
Version 3, updated 4/5/23
Reference: https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib
https://shapely.readthedocs.io/en/stable/manual.html
'''
import numpy as np, matplotlib as mpl
from matplotlib import cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon
import shapely.geometry, shapely.affinity
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
plt.grid(True, which='major', color='silver', linewidth=0.1, linestyle='-')

class colorbar_rgba:

    def __init__(self, cmap_name, min_val, max_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)
        self.vmin = self.norm.vmin
        self.vmax = self.norm.vmax

    def __call__(self, val):
        return self.scalarMap.to_rgba(val)
    
    def apply_to_axis(self, ax):
        self.bar = ax.get_figure().colorbar(self.scalarMap)

    def apply_to_axis_discrete(self, ax, n_levels):
        ax.get_figure().subplots_adjust(right=0.8)
        cbar_ax = ax.get_figure().add_axes([0.85, 0.1, 0.03, 0.775])

        bounds = np.linspace(self.norm.vmin, self.norm.vmax, n_levels)
        self.bar = mpl.colorbar.ColorbarBase(cbar_ax, cmap=self.cmap, norm=self.norm,
                                                         spacing='proportional', ticks=bounds, boundaries=bounds)

def plot_polygon(ax, poly, **kwargs):
    path = Path.make_compound_path(
        Path(np.asarray(poly.exterior.coords)[:, :2]),
        *[Path(np.asarray(ring.coords)[:, :2]) for ring in poly.interiors])

    patch = PathPatch(path, **kwargs)
    collection = PatchCollection([patch], **kwargs)

    ax.add_collection(collection, autolim=True)
    ax.autoscale_view()
    ax.set_axisbelow(True)
    return collection

def get_poly_verts(poly):
    return list(poly.boundary.coords)

def get_poly_centroid(poly):
    return poly.centroid.coords[0]

def circ_pattern_poly(poly, rotation_center_coord, total_count, angular_spacing):
    polies = [poly]
    for i in range(total_count-1):
        polies.append(shapely.affinity.rotate(poly, (1+i)*angular_spacing, rotation_center_coord))
    return polies

def scale_poly(poly, xfactor, yfactor, origin):
    '''Origin can be "centroid" (for center of vertices) or a 2-tuple.
    Negative scale factors mirror coordinates.'''
    return shapely.affinity.scale(poly, xfactor, yfactor, origin=origin)

def point_in_poly(poly, point:tuple):
    return poly.contains(shapely.geometry.Point(*point))

def label_poly(ax, poly, text, **kwargs):
    ax.annotate(text, xy=get_poly_centroid(poly), **kwargs)

if __name__ == '__main__':
    # Generate geometry
    xpoints = np.sin(0.2*np.pi*np.arange(10))
    ypoints = np.cos(0.2*np.pi*np.arange(10))+1.5
    p1 = shapely.geometry.Polygon(shell=list(zip(xpoints,ypoints)))
    polygons = circ_pattern_poly(p1, (0,0), 4, 90)

    # Color bar stuff
    polygon_values = [i+0.5 for i,p in enumerate(polygons)]
    cbar = colorbar_rgba('RdYlBu_r', 0, len(polygons))

    # plot everything
    fig,ax = plt.subplots()
    cbar.apply_to_axis_discrete(ax, 16)
    for j,p in enumerate(polygons):
        plot_polygon(ax,p, color=cbar(j))
        p_inner = scale_poly(p, 0.5, 0.5,'centroid')
        plot_polygon(ax, p_inner, color=cbar(j+.25))
        label_poly(ax, p, j, color='white', fontsize=15)
    plt.grid(True, which='major', color='silver', linewidth=0.1, linestyle='--')
    fig.show()

pass