# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 20:31:52 2021
@author: clay
Two years later, I added this code to Github!
"""
import copy
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import os
import imageio
plt.style.use('bmh')
plt.grid(True, which='major', color='silver', linewidth=0.1, linestyle='-')

#-----------------------------------------------------------------------------#
#                       Colorbar RGBA                                         #
#-----------------------------------------------------------------------------#
class colorbar_rgba:

    def __init__(self, cmap_name, min_val, max_val):
        '''jet is the suggested color map.'''
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

#-----------------------------------------------------------------------------#
#                              Plot2d                                         #
#-----------------------------------------------------------------------------#
class Plot2d():

    def __init__(self, title=None, xlabel=None, ylabel=None,
                    cmap_name='coolwarm', style='bmh', minorgrid=True,
                    dpi=150, figsize=(6.4,4.8), logx=False, logy=False,
                    fig=None, ax=None):
        self.title, self.xlabel, self.ylabel =  title, xlabel, ylabel
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.style = style
        self.minorgrid = minorgrid
        self.logx, self.logy = logx,logy
        self.dpi, self.figsize = dpi, figsize
        self.fig, self.ax = fig, ax
        self.cbar = None
        self._dim = 2
        self._points = []

    def _apply_ax_formats(self, ax):
        ax.set_title(self.title)
        ax.set_xlabel(self.xlabel)
        ax.set_ylabel(self.ylabel)
        if self.minorgrid:
            ax.grid(which='minor', linestyle='--', linewidth=0.1)
            ax.tick_params(which='minor', length=1, width=0.5)
            ax.minorticks_on()
        if self.logx:
            ax.set_xscale('log')
        if self.logy:
            ax.set_yscale('log')
        return ax

    def _blankplot(self):
        with plt.style.context(self.style):
            fig,ax = plt.subplots(dpi=self.dpi, figsize=self.figsize)

        ax = self._apply_ax_formats(ax)
        return fig,ax

    def _fig_gen(self):
        if self.fig == None or self.ax == None:
            self.fig, self.ax = self._blankplot()

    def _reset_cbar(self):
        try:
            self.cbar.remove()
            self.cbar = None
        except AttributeError:
            pass

    def _manual_point_colors(self, points_xyc:tuple):
        coords = list(zip(*points_xyc))
        c_to_rgba = colorbar_rgba(self.cmap_name, min(coords[-1]), max(coords[-1]))
        rgbas = [c_to_rgba(v) for v in coords[-1]]
        xyr = list(zip(*coords[:-1],rgbas))
        return xyr,c_to_rgba

    def _plot_multi_lines(self, start_points, end_points, **kwargs):
        self._fig_gen()

        if len(start_points) == len(end_points):
            if len(start_points[0]) == len(end_points[0]):
                if len(start_points[0]) == self._dim+1:
                    xyc_all = [e for sub in zip(start_points,end_points) for e in sub]
                    xyr_all, c_to_rgba = self._manual_point_colors(xyc_all)
                    xyr_0 = xyr_all[::2]
                    xyr_1 = xyr_all[1::2]
                    for i,vals in enumerate(zip(xyr_0,xyr_1)):
                        coords0,coords1 = vals[0][:-1], vals[1][:-1]
                        blend = c_to_rgba.blend(vals[0][-1], vals[1][-1])
                        self.ax.plot(*list(zip(coords0,coords1)), color=blend, **kwargs)
                elif len(start_points[0]) == self._dim:
                    for i,vals in enumerate(zip(start_points, end_points)):
                        self.ax.plot(*list(zip(vals[0],vals[1])), **kwargs)
                else:
                    raise ValueError('Points must have {} or {} values.'.format(self._dim, self._dim+1))
            else:
                raise ValueError('Points in start_points and end_points must have the same dimension.')
        else:
            raise ValueError('start_points and end_points must have the same number of points.')

    def scatter(self, points_xyc:tuple, **kwargs):
        self._fig_gen()

        if len(points_xyc[0]) == 3:
            self._reset_cbar()
            x,y,c = list(zip(*points_xyc))
            chart = self.ax.scatter(x,y,c=c, cmap=self.cmap, **kwargs)
            self.cbar = plt.colorbar(chart)
        elif len(points_xyc[0]) == 2:
            x,y = list(zip(*points_xyc))
            self.ax.scatter(x,y, **kwargs)
        else:
            raise ValueError('Each point must have 2 or 3 values.')

    def line(self, points_xyc:tuple, **kwargs):
        self._fig_gen()

        if len(points_xyc[0]) == self._dim+1:
            start_points = points_xyc[:-2]
            end_points = [points_xyc[1:][i] for i,v in enumerate(start_points)]
            self._plot_multi_lines(start_points, end_points, **kwargs)
            self.scatter(points_xyc, s=0)
        elif len(points_xyc[0]) == self._dim:
            coords = list(zip(*points_xyc))
            self.ax.plot(*coords, **kwargs)
            self._points.extend(points_xyc)
        else:
            raise ValueError('Each point must have {} or {} values.'.format(self._dim, self._dim+1))

    def lines(self, start_points, end_points, **kwargs):
        self._fig_gen()

        self._plot_multi_lines(start_points, end_points, **kwargs)
        if len(start_points[0]) == self._dim+1:
            self.scatter(start_points+end_points, s=0)
        elif len(start_points[0]) == self._dim:
            pass
        else:
            raise ValueError('Each point must have {} or {} values.'.format(self._dim, self._dim+1))

    def show(self, *args, **kwargs):
        plt.figure(self.fig)
        plt.show(*args, **kwargs)

    def fig_ax(self):
        return self.fig, self.ax

    def save(self, path, **kwargs):
        plt.figure(self.fig)
        plt.savefig(path, dpi=self.dpi, **kwargs)

    def plain(self):
        plain = copy.copy(self)
        plain.ax.grid(False)
        plain.ax.set_axis_off()
        plain._reset_cbar()
        plain.ax.set_title(None)
        plain.ax.set_xlabel(None)
        plain.ax.set_ylabel(None)
        return plain

#-----------------------------------------------------------------------------#
#                              Plot3d                                         #
#-----------------------------------------------------------------------------#
class Plot3d(Plot2d):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.xlabel == None:
            self.xlabel = 'X-Coordinate'
        if self.ylabel == None:
            self.ylabel = 'Y-Coordinate'
        self.minorgrid = False #---- Not ideal, prevents minorgrid=True
        #----------- Non-2D attributes get default values -------------#
        self.minorgrid = False
        self.logz = False
        self.azim, self.elev = -60, 30
        self.zlabel = 'Z-Coordinate'
        self._dim = 3

    def _apply_ax_formats(self, ax):
        ax = super()._apply_ax_formats(ax)
        ax.set_zlabel(self.zlabel)
        ax.azim, ax.elev = self.azim, self.elev
        if self.logz:
            ax.set_zscale('log')
        return ax

    def _blankplot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax = self._apply_ax_formats(ax)

        return fig,ax

    def scatter(self, points_xyzc:tuple, **kwargs):
        self._fig_gen()
        self._points.extend(points_xyzc)

        if len(points_xyzc[0]) == 4:
            self._reset_cbar()
            x,y,z,c = list(zip(*points_xyzc))
            chart = self.ax.scatter(x,y,z,c=c, cmap=self.cmap, **kwargs)
            self.cbar = plt.colorbar(chart)
        elif len(points_xyzc[0]) == 3:
            x,y,z = list(zip(*points_xyzc))
            self.ax.scatter(x,y,z, **kwargs)
        else:
            raise ValueError('Each point must have {} or {} values.'.format(self._dim, self._dim+1))

    def equal_aspect_ratio(self, oversize_factor=1):
        equal_aspect_ratio(self.ax, oversize_factor)

def figax3d():
    p3d = Plot3d()
    return p3d._blankplot()

def equal_aspect_ratio(ax, oversize_factor=1):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1]-x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1]-y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1]-z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = oversize_factor*0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle-plot_radius, x_middle+plot_radius])
    ax.set_ylim3d([y_middle-plot_radius, y_middle+plot_radius])
    ax.set_zlim3d([z_middle-plot_radius, z_middle+plot_radius])

#-----------------------------------------------------------------------------#
#                              animate                                        #
#-----------------------------------------------------------------------------#
def animate(plotfunc, path, frame_plot_args:dict, fps=24, loop=0):
    '''- plotfunc: function that returns mpl fig, ax
    - path: full filepath ending in '.gif' for final animation
    - frame_plot_args: dict of lists {frame_number:[plot_func_args]}
    Still frames are saved into and deleted from the lowest directory in path.'''
    name_base = 'frame_{}.png'
    savedir = os.path.dirname(path)

    framenames = []
    for k,v in frame_plot_args.items():
        number = str(k).zfill(8)
        framename = name_base.format(number)
        framenames.append(framename)
        fig,ax = plotfunc(*v)
        plt.figure(fig)
        plt.savefig(os.path.join(savedir, framename))
        plt.close()

    framepaths = [os.path.join(savedir,f) for f in framenames]
    imgs = [imageio.imread(f) for f in framepaths]
    imageio.mimwrite(path, imgs, fps=fps, loop=loop)

    for f in set(framepaths):
        os.remove(f)