# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 18:56:43 2020
@author: cbutton
"""
#--------------------------------- Imports -----------------------------------#
import pandas as pd
import numpy as np
import scipy.spatial
import scipy.interpolate
import copy
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
from xpyz import plottools

#----------------------- Classes: TransformMatrix3d --------------------------#
class TransM3d(object):
    #--------------------- Non-User Funcs ------------------------------------#
    def __init__(self):
        self.a = np.identity(4,dtype=np.float64)

    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError("'{}' object has no attribute {}".format(type(self), name))
        return getattr(self.a, name)

    def __iter__(self):
        self.a.__iter__()

    def __setitem__(self, key, value):
        self.a[key] = value

    def __getitem__(self, key):
        return self.a.__getitem__(key)

    def __repr__(self):
        return '{}\n{}'.format(type(self),str(self.a))

    def __str__(self):
        return self.__repr__()

    def copy(self):
        return copy.deepcopy(self)

    def _argfix(self, arr3d:tuple, *args):
        try:
            iter(arr3d)
        except TypeError:
            arr4d = arr([arr3d]+list(args)+[1])
        else:
            arr4d = arr(list(arr3d)+[1])
        return arr4d

    def __call__(self, arr3d_or_TransM3d, *args):
        if type(arr3d_or_TransM3d) == TransM3d:
            m3d = arr3d_or_TransM3d.copy()
            m3d.a = np.matmul(self.a,m3d.a)
            return m3d
        else:
            arr4d = self._argfix(arr3d_or_TransM3d, *args)
            return np.matmul(self.a,arr4d)[:3]

    #---------------------- Matrix and Vector Generation ---------------------#
    def co(self):
        origin = [self.a[i,3] for i in range(3)]
        return arr(origin)

    def inv(self, arr3d:tuple=None, *args):
        m_inv = np.linalg.inv(self.a)
        invm3d = TransM3d()
        invm3d.a = m_inv
        if type(arr3d) == type(None):
            return invm3d
        elif type(arr3d) == TransM3d:
            return invm3d(arr3d, *args)
        else:
            return invm3d(arr3d, *args)[:3]

    def tmat(self, arr3d:tuple, invertedcsys=True):
        tm = np.identity(4)
        if invertedcsys:
            arr3d = self.inv(arr3d)
        for i,v in enumerate(arr3d):
            tm[i,3] += v
        newtm3d = TransM3d()
        newtm3d.a = tm
        return newtm3d

    def rmat(self, rotation_degrees:float, rotation_axis_vector:tuple, invertedcsys=True):
        if rss(rotation_axis_vector) == 0:
            raise ValueError('rotation_axis_vector must have length > 0.')
        radians = np.radians(rotation_degrees)
        axis = arr(rotation_axis_vector)
        if invertedcsys:
            axis = self.inv(axis)
        r = scipy.spatial.transform.Rotation.from_rotvec(axis/np.linalg.norm(axis)*radians)
        rm = np.vstack((r.as_matrix(),arr(0,0,0)))
        newcol = arr(((0,0,0,1),(0,0,0,0))).transpose()[:,:1]
        rm = np.hstack((rm,newcol))
        newtm3d = TransM3d()
        newtm3d.a = rm
        return newtm3d

    def smat(self, xyzvec:tuple, invertedcsys=True):
        if invertedcsys:
            xyzvec = self.inv(xyzvec)
        
        sm = np.identity(4)
        for i,d in enumerate(xyzvec):
            sm[i,i] = d
        newtm3d = TransM3d()
        newtm3d.a = sm
        return newtm3d

    def alignment_rotation_matrix(self, tm2):
        tm10 = self.copy()
        tm20 = tm2.copy()
        tm10.translate_o()
        tm20.translate_o()
        sv = arr(1,0,0)
        ev = normalize(tm10.inv(tm20(1,0,0)))
        rotvec = normalize(np.cross(sv,ev))
        rads = np.arccos(sv.dot(ev))

        rm = TransM3d()
        if rss(rotvec) != 0:
            rm.rotate(np.degrees(rads), rotvec)
        tm10.a = np.matmul(tm10.a,rm.a)

        sv = arr(0,1,0)
        ev = normalize(tm10.inv(tm20(0,1,0)))
        rotvec = np.cross(sv,ev)
        rads = np.arccos(sv.dot(ev))

        rm2 = TransM3d()
        if rss(rotvec) != 0:
            rm2.rotate(np.degrees(rads), rotvec)

        rmf = TransM3d()
        rmf.a = np.matmul(rm.a,rm2.a)
        
        return rmf

    def alignment_rotvec(self, tm2):
        rm = self.alignment_rotation_matrix(tm2)
        return rm.euler_angle_axis()

    def alignment_rotvec_g(self, tm2):
        a,vl = self.alignment_rotvec(tm2)
        tm = self.copy()
        tm.translate_o()
        return a, tm(vl)

    #------------------------ Transformations --------------------------------#
    def translate(self, arr3d:tuple, *args):
        arr3d = self._argfix(arr3d, *args)[:3]
        tm = self.tmat(arr3d, False)
        self.a = np.matmul(self.a, tm.a)

    def rotate(self, rotation_degrees:float, rotation_axis_vector:tuple):
        rm = self.rmat(rotation_degrees, rotation_axis_vector, False)
        self.a = np.matmul(self.a, rm.a)

    def scale(self, xyz_scale_vec:tuple, *args):
        xyzvec = self._argfix(xyz_scale_vec, *args)[:3]
        sm = self.smat(xyzvec, False)
        self.a = np.matmul(self.a, sm.a)

    def translate_g(self, arr3d:tuple, *args):
        arr3d = self._argfix(arr3d, *args)[:3]
        for i,v in enumerate(arr3d):
            self.a[i,3] += v

    def rotate_g(self, rotation_degrees:float, rotation_axis_vector:tuple):
        rm = TransM3d().rmat(rotation_degrees, rotation_axis_vector, False)
        offset = self.co()
        self.translate_g(-offset)
        self.a = np.matmul(rm.a, self.a)
        self.translate_g(offset)

    def rotate_point(self, point, rotation_degrees:float, rotation_axis_vector:tuple):
        rm = TransM3d().rmat(rotation_degrees, rotation_axis_vector, False)
        delta = self.co()-point
        delta2 = rm(delta)
        self.translate_o()
        self.a = np.matmul(rm.a,self.a)
        self.translate_g(point)
        self.translate_g(delta2)

    def rotate_origin(self, rotation_degrees:float, rotation_axis_vector:tuple):
        self.rotate_point((0,0,0),rotation_degrees,rotation_axis_vector)

    def align(self, local_start_vec, local_end_vec):
        sv = normalize(local_start_vec)
        ev = normalize(local_end_vec)
        rotvec = np.cross(sv,ev)
        rads = np.arccos(sv.dot(ev))
        self.rotate(np.degrees(rads), rotvec)

    def align_gg(self, global_start_vec, global_end_vec):
        tm = self.copy()
        tm.translate_o()
        self.align(tm.inv(global_start_vec), tm.inv(global_end_vec))

    def align_lg(self, local_start_vec, global_end_vec):
        tm = self.copy()
        tm.translate_o()
        self.align(local_start_vec, tm.inv(global_end_vec))

    def ptp_l(self, origin_point:tuple, destination_point:tuple):
        vec = arr(destination_point)-arr(origin_point)
        self.translate(vec)

    def ptp_g(self, origin_point:tuple, destination_point:tuple):
        vec = arr(destination_point)-arr(origin_point)
        self.translate_g(vec)

    def translate_o(self):
        self.ptp_g(self.co(),(0,0,0))

    #--------------------------- Other ---------------------------------------#
    def reset(self):
        self.a = np.identity(4, dtype=np.float64)

    def euler_angles(self):
        '''returns xyz (roll,pitch,yaw) rotation angles'''
        R = self.a[:3,:3]
        if R[2,0] == 1:
            theta_z = 0
            theta_y = np.pi/2
            theta_x = theta_z + np.arctan2(R[0,1],R[0,2])
        elif R[2,0] == -1:
            theta_z = 0
            theta_y = -np.pi/2
            theta_x = -theta_z + np.arctan2(-R[0,1],-R[0,2])
        else:
            theta_x = np.arctan2(R[2,1],R[2,2])
            theta_y = np.arctan2(-R[2,0], (R[2,1]**2+R[2,2]**2)**0.5)
            theta_z = np.arctan2(R[1,0],R[0,0])
        return tuple(np.degrees((theta_x,theta_y,theta_z)))

    def euler_mats(self):
        ax,ay,az = self.euler_angles()
        mx,my,mz = [TransM3d() for i in range(3)]
        for m,a,v in zip((mx,my,mz),
                         (ax,ay,az),
                         [(1,0,0),(0,1,0),(0,0,1)]):
            m.rotate(a,v)
        return mx,my,mz

    def euler_angle_axis(self):
        R = self.a[:3,:3]
        theta = np.arccos((np.trace(R)-1)/2)
        if np.isnan(theta):
            theta = np.arccos((np.round(np.trace(R),10)-1)/2)
            if np.isnan(theta):
                raise BaseException(R, np.trace(R))
        vec = (R[2,1]-R[1,2],
                R[0,2]-R[2,0],
                R[1,0]-R[0,1])
        return np.degrees(theta),normalize(vec)

    def plot(self, fig=None, ax=None, return_fig_ax=False):
        if fig==None or ax==None:
            fig,ax = plottools.figax3d()
        xo = [(0,0,0), (1,0,0)]
        yo = [(0,0,0), (0,1,0)]
        zo = [(0,0,0), (0,0,1)]
        xwm = [self(0,0,0), self(xo[1])]
        ywm = [self(0,0,0), self(yo[1])]
        zwm = [self(0,0,0), self(zo[1])]

        ax_data = {'origin_x':['red','dotted',5],
                   'origin_y':['green','dotted',5],
                   'origin_z':['blue','dotted',5],
                   'world_matrix_x':['red','solid',1],
                   'world_matrix_y':['green','solid',1],
                   'world_matrix_z':['blue','solid',1]}

        for p,v in zip((xo,yo,zo,xwm,ywm,zwm),ax_data.values()):
            v.extend(list(zip(p[0],p[1])))
        ax_data['csys_link'] = ['black', 'dotted', 1]+list(zip((0,0,0),self.co()))
 
        for k,v in ax_data.items():
            ax.plot(v[3],v[4],v[5],color=v[0],linestyle=v[1],linewidth=v[2])

        plottools.equal_aspect_ratio(ax)
        if return_fig_ax:
            return fig,ax
        else:
            fig.show()

#-------------------------- Classes: Coordgen --------------------------------#
class CoordinateGenerator(object):
    '''All bound methods return coordinates as 3-tuples in xyz row format.'''
    def __init__(self):
        super().__init__()

    def cube_grid(self, xyz_dims:tuple, xyz_counts:tuple):
        xyz_counts = [n-1 for n in xyz_counts]
        if any([n==0 for n in xyz_counts]):
            raise ValueError('cube_grid xyz_counts must be at least 2.')
        corner = -arr(xyz_dims)/2
        increment = arr(xyz_dims)/arr(xyz_counts)
        points = []
        for i in range(xyz_counts[0]+1):
            for j in range(xyz_counts[1]+1):
                for k in range(xyz_counts[2]+1):
                    delta = arr(i*increment[0],j*increment[1],k*increment[2])
                    p = corner+delta
                    points.append(tuple(p))
        return points

    def cube_random(self, x_width:float, y_width:float, z_width:float, count:int):
        ps = np.random.rand(count,3) - arr(.5,.5,.5)
        p2s = []
        for p in ps:
            scaled = []
            for el,w in zip(p,(x_width,y_width,z_width)):
                    scaled.append(el*w)
            p2s.append(tuple(scaled))
        return p2s

    def helix(self, steps:int, helix_radius:float, step_degrees:float, step_height:float):
        '''Helix axis is +Z.'''
        theta = np.radians(step_degrees)
        rotations = [i*theta for i in range(steps)]
        x1s = [helix_radius*np.cos(t) for t in rotations]
        y1s = [helix_radius*np.sin(t) for t in rotations]
        zs = [i*step_height for i in range(steps)]
        xyz_coords = []
        for x1,y1,z in zip(x1s,y1s,zs):
            xyz_coords.append((x1,y1,z))
        return xyz_coords

    def helix_double(self, steps:int, helix_radius:float, step_degrees:float, step_height:float):
        '''Helix axis is +Z. Output coordinates are in pairs moving in +Z.'''
        coords1 = self.helix(steps,helix_radius,step_degrees,step_height)
        xyz_coords = []
        for x1,y1,z in coords1:
            xyz_coords.append((x1,y1,z))
            xyz_coords.append((-x1,-y1,z))
        return xyz_coords

    def sphere_grid(self, radius, radial_count, azimuth_count, inclination_count, center=True, add_poles=True):
        raddist, azrad, incrad = radius/radial_count, 2*np.pi/azimuth_count, np.pi/inclination_count
        points = []
        if center:
            points.append((0,0,0))

        raddist2, azrad2, incrad2 = raddist,0,incrad
        while raddist2 <= radius:
            while azrad2 < 2*np.pi:
                while incrad2 < np.pi-incrad/2:
                    points.append(spherical_to_cartesian(incrad2, azrad2, raddist2))
                    incrad2 += incrad
                incrad2 = incrad
                azrad2 += azrad
            azrad2 = azrad
            raddist2 += raddist
            
        if add_poles:
            points.extend([(0,0,radius),(0,0,-radius)])
        
        return points

    def circle_radial(self, radius, angular_count:float, radial_count:float):
        r0 = radius/radial_count
        a0 = (2*np.pi)/angular_count
        
        r = r0
        points = []
        for i in range(radial_count-1):
            for j in range(angular_count):
                p = list(spherical_to_cartesian(np.pi/2,a0*j,r))
                p[2] = 0
                points.append(tuple(p))
            r += r0
        return points

#-------------------------- Classes: XYZframe --------------------------------#
class XYZframe(object):

    def __init__(self):
        super().__init__()
        self.df = None
        self.clickdata = []
        self.wm = TransM3d()

    #------------ Allowing df attributes to be accessed directly -------------#
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError("'{}' object has no attribute {}".format(type(self), name))
        return getattr(self.df, name)

    def __iter__(self):
        self.df.__iter__()

    def __setitem__(self, key, value):
        self.df[key] = value

    def __getitem__(self, key):
        return self.df.__getitem__(key)

    def __repr__(self):
        return '{}\n{}'.format(type(self),self.df.head())

    def __str__(self):
        return self.__repr__()

    #------------- Easy coordinate access ------------#    
    def __call__(self, index:int, xyz_only=True):
        series = self.df.loc[index,:].copy()
        if xyz_only:
            return series[['X','Y','Z']].values
        else:
            return series

    def points(self, vector=None):
        if vector != None:
            return self[['X','Y','Z',vector]].values.tolist()
        else:
            return self[['X','Y','Z']].values.tolist()

    #----------------------------- Copy --------------------------------------#
    def copy(self, df_or_pointlist=None):
        dfps = df_or_pointlist
        newxf = copy.deepcopy(self)
        if type(df_or_pointlist) == type(None):
            return newxf
        elif type(dfps) == pd.core.frame.DataFrame:
            newxf.df = dfps
        else:
            newxf.df = newxf.makedf(dfps)
        return newxf

    #-------------------- df generation and editing --------------------------#
    def makedf(self, xyz_list:np.array):
        cols = ['X','Y','Z']
        if len(np.shape(xyz_list[0]))!=0:
            df = pd.DataFrame(columns=cols,data=xyz_list)
        else:
            df = pd.DataFrame(columns=cols,data=[xyz_list])
        return df

    def append(self, xyzframe_or_df):
        xdf = xyzframe_or_df
        if type(xdf) == pd.core.frame.DataFrame:
            self.df = self.df.append(xdf)
        else:
            self.df = self.df.append(xdf.df)
        self.df.reset_index(inplace=True,drop=True)

    def pointadd(self, point_or_point_list):
        p = point_or_point_list
        df = self.makedf(p)
        self.append(df)

    #----------------------- Geometric Transformations -----------------------#
    def transform(self, xf=None):
        if xf==None:
            xf = self
        for i in xf.index:
            co = xf(i)
            xf.loc[i,['X','Y','Z']] = self.wm(co)

    def revert(self, xf=None, reset_wm=True):
        m = self.wm
        if xf==None:
            xf = self
        if reset_wm:
            xf.wm.reset()
        for i in xf.index:
            co = xf(i)
            xf.loc[i,['X','Y','Z']] = m.inv(co)

    def translate(self, xyzvec:tuple, *args):
        self.revert(None,False)
        self.wm.translate(xyzvec, *args)
        self.transform()

    def rotate(self, rotation_degrees:float, rotation_axis_vector:tuple):
        self.revert(None,False)
        self.wm.rotate(rotation_degrees,rotation_axis_vector)
        self.transform()

    def scale(self, xyzvec:tuple, *args):
        self.revert(None,False)
        self.wm.scale(xyzvec, *args)
        self.transform()

    def translate_g(self, xyzvec:tuple, *args):
        self.revert(None,False)
        self.wm.translate_g(xyzvec, *args)
        self.transform()

    def rotate_g(self, rotation_degrees:float, rotation_axis_vector:tuple):
        self.revert(None,False)
        self.wm.rotate_g(rotation_degrees,rotation_axis_vector)
        self.transform()

    def rotate_origin(self, rotation_degrees:float, rotation_axis_vector:tuple):
        self.revert(None,False)
        self.wm.rotate_origin(rotation_degrees, rotation_axis_vector)
        self.transform()

    def ptp(self, startpoint:tuple, endpoint:tuple):
        delta = arr(endpoint)-arr(startpoint)
        self.translate(delta)

    def align_to_vector(self, start_vector:tuple, destination_vector:tuple):
        dv = normalize(destination_vector)
        sv = normalize(start_vector)
        rotvec = np.cross(sv,dv)
        rads = np.arccos(sv.dot(dv))
        self.rotate_rad(rotvec, rads)

    def offset_rotate(self, rotation_axis_vector:np.array, rotation_degrees:float, rotation_center:np.array):
        self.translate(-arr(rotation_center))
        self.rotate(arr(rotation_axis_vector), float(rotation_degrees))
        self.translate(arr(rotation_center))

    #--------------------- Nonlinear Transformations -------------------------#
    def scramble(self, magnitude:float):
        data = self.df[['X','Y','Z']].values
        rands = np.random.rand(len(data),3)
        rands *= magnitude
        data += rands
        self.df[['X','Y','Z']] = data

    def twist(self, rotation_degrees:float):
        '''Twist is along Z-axis. Combine with rotations to get other axes of twist.'''
        zmin = self.df['Z'].min()
        zrange = self.df['Z'].max()-zmin
        data = self.df[['X','Y','Z']].values
        data2 = []
        for p in data:
            ps = list(cartesian_to_spherical(*p))
            deg = rotation_degrees*(p[2]-zmin)/zrange
            ps[1] += np.radians(deg)
            data2.append(spherical_to_cartesian(*ps))
        self.df[['X','Y','Z']] = data2

    #--------------------------- Calculations --------------------------------#
    def tree_gen(self, update=False):
        if update == False:
            try:
                return self.tree
            except AttributeError:
                update = True              
        if update == True:
            self.tree = scipy.spatial.cKDTree(self.df[['X','Y','Z']].values)
            return self.tree

    def knns(self, point:tuple, count:int, max_dist:float=None):
        point = arr(point)
        tree = self.tree_gen()
        ds,inds = tree.query(point,count)

        ds2,inds2= [],[]
        if max_dist != None:
            ds2,inds2= [],[]
            for d,i in zip(ds,inds):
                if d<max_dist:
                    ds2 += [d]
                    inds2 += [i]
            ds,inds = ds2,inds2

        if count > 1:
            df_inds = [self.df.index.tolist()[i] for i in inds]
        else:
            df_inds = [self.df.index.tolist()[inds]]
        df2 = self.df[self.df.index.isin(df_inds)].copy()

        return df2

    def cubicspline(self, close=False, returnCubicSplines=False, on='index', **kwargs):
        '''Returns cubic splines for x,y, and z on index. Kwargs from 
        scipy.interpolate.CubicSpline documentation.'''
        if close:
            self.df = self.df.append(self.df.iloc[0,:], ignore_index=True)
            kwargs['bc_type'] = 'periodic'
        
        if on == 'index':
            indepvar = self.index
        else:
            indepvar = self[on]
        csx = scipy.interpolate.CubicSpline(indepvar, self['X'], **kwargs)
        csy = scipy.interpolate.CubicSpline(indepvar, self['Y'], **kwargs)
        csz = scipy.interpolate.CubicSpline(indepvar, self['Z'], **kwargs)

        self.cs = lambda val: (csx(val), csy(val), csz(val))
        if returnCubicSplines:
            return csx,csy,csz

    #---------------------- append derived data ------------------------------#
    def normals_from_sources(self, source_point_list:list):
        norms = [normalize(calc_ray(s,d)) for s,d
                 in zip(source_point_list,self.df[['X','Y','Z']].values)]
        self.df['nX'],self.df['nY'],self.df['nZ'] = zip(*norms)

    def normals_from_source(self, source:tuple):
        norms = [normalize(calc_ray(source,r)) for r in self.df[['X','Y','Z']].values]
        self.df['nX'],self.df['nY'],self.df['nZ'] = zip(*norms)

    def normals_from_centroid(self):
        ctr = self.df[['X','Y','Z']].mean()
        self.normals_from_source(ctr)

    def nn_distances(self):
        tree = self.tree_gen()
        dists = []
        pbar = tqdm(total=self.df.shape[0], ascii=True)
        for i,row in self.df.iterrows():
            p = row[['X','Y','Z']].values
            ds,inds = tree.query(p,2)
            dists.append(ds[1])
            pbar.update(1)
        pbar.close()
        self.df['nn_dist'] = dists

    def rss(self):
        self.df['rss'] = [rss((x,y,z)) for x,y,z in self[['X','Y','Z']].values]

    def rss_xy(self):
        self.df['rss_xy'] = [(x**2+y**2)**0.5 for x,y in self[['X','Y']].values]

    def rss_xz(self):
        self.df['rss_xz'] = [(x**2+z**2)**0.5 for x,z in self[['X','Z']].values]

    def rss_yz(self):
        self.df['rss_yz'] = [(y**2+z**2)**0.5 for y,z in self[['Y','Z']].values]

    #--------------------------- 2D conversions ------------------------------#
    def flatten(self, method='spherical_equirec'):
        methods = {'spherical_equirec':self._spherical_equirec}
        methods[method]()

    def _spherical_equirec(self):
        '''Vertical seam introduced at Y+ half-plane.'''
        spheredata = []
        for i,row in self.df.iterrows():
            rsph = cartesian_to_spherical(*row[['X','Y','Z']].values)
            spheredata.append(rsph)
        self.df['X'],self.df['Y'] = list(zip(*spheredata))[:2]
        self.df['Z'] = self.df['Z'].apply(lambda x: 0)
        if hasattr(self,'tree'):
            self.tree_gen(True)
        if 'nn_dist' in self.df:
            self.nn_distances()

    #------------------------------ plotting ---------------------------------#
    def plot(self,vector:str=None, return_fig_ax=False, **kwargs):
        '''
        Colormaps are specified by "cmap_name":
            - coolwarm (default) is blue low, red high.
            - coolwarm_r is red low, blue high.
            - gist_rainbow when intermediate values need to be distinct.
            - Set1 and Set1_r are good for binary values.
        '''
        p3d = plottools.Plot3d()

        if vector == None:
            points = self.df[['X','Y','Z']].values.tolist()
        else:
            points = self.df[['X','Y','Z',vector]].values.tolist()

        p3d.scatter(points, **kwargs)

        if return_fig_ax:
            return p3d.fig, p3d.ax
        else:
            p3d.show()

    def plot_with_wm(self, return_fig_ax=False, **xf_plot_kwargs):
        if return_fig_ax == True:
            xf_plot_kwargs['return_fig_ax'] = True
            fig,ax = self.plot(**xf_plot_kwargs)
            self.wm.plot(fig,ax)
            return fig,ax
        else:
            fig,ax = self.plot(**xf_plot_kwargs, return_fig_ax=True)
            self.wm.plot(fig,ax)
            fig.show()

class xf_generator(object):
    def __init__(self):
        super().__init__()
 
    def __call__(self, df_or_pointlist_or_csvfilepath):
        return makeframe(df_or_pointlist_or_csvfilepath)

    def cube_grid(self, xyz_dims:tuple, xyz_counts:tuple):
        return self(coordgen.cube_grid(xyz_dims, xyz_counts))

    def cube_random(self, x_width:float, y_width:float, z_width:float, count:int):
        return self(coordgen.cube_random(x_width, y_width, z_width, count))
    
    def helix(self, steps:int, helix_radius:float, step_degrees:float, step_height:float):
        return self(coordgen.helix(steps,helix_radius,step_degrees,step_height))
   
    def helix_double(self, steps:int, helix_radius:float, step_degrees:float, step_height:float):
        return self(coordgen.helix_double(steps,helix_radius,step_degrees,step_height))

    def helix_n(self, helix_number:int, steps:int, helix_radius:float, step_degrees:float, step_height:float):
        xf0 = self.helix(steps, helix_radius, step_degrees, step_height)
        rotation_angle = 360/helix_number
        for i in range(helix_number):
            xf = xf0.copy()
            xf.rotate(rotation_angle, (0,0,1))
            xf0.append(xf)
        return xf0

    def sphere_grid(self, radius, radial_count, azimuth_count, inclination_count, center=True, add_poles=True):
        return self(coordgen.sphere_grid(radius,radial_count,azimuth_count,inclination_count,center,add_poles))
    
    def circle_radial(self, radius, angular_count:float, radial_count:float, radial_normals=True):
        xf = self(coordgen.circle_radial(radius,angular_count,radial_count))
        xf.normals_from_source((0,0,0))
        return xf

#----------------------------- Module Objects --------------------------------#
coordgen = CoordinateGenerator()
xfgen = xf_generator()
frame = XYZframe()

#----------------------- Module Functions: Math ------------------------------#
def arr(iterable_1d, *args):
    '''Numpy type-forcing to suppress dtype errors.'''
    try:
        test = len(iterable_1d)
        del test
        return np.array(iterable_1d, dtype=np.float64)
    except TypeError:
        input_data = [iterable_1d]
        input_data.extend(args)
        return np.array(input_data, dtype=np.float64)

def rss(vec, *args):
    norm = np.linalg.norm(arr(vec, *args))
    return norm

def normalize(vec):
    norm = rss(vec)
    if norm != 0:
        return vec / norm
    else:
        return vec

def midpoint(vec1, vec2):
    return (arr(vec1)+arr(vec2))/2

def spherical_to_cartesian(inclination, azimuth, radius):
    '''Angles in radians.'''
    i,a,r = inclination, azimuth, radius
    x = r*np.sin(i)*np.cos(a)
    y = r*np.sin(i)*np.sin(a)
    z = r*np.cos(i)
    return (x,y,z)

def cartesian_to_spherical(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    i = np.arccos((z/r))
    a = np.arctan2(x,y)
    return (i,a,r)

def calc_ray(source:tuple, dest:tuple):
    ray = arr(dest)-arr(source)
    return ray

def vector_angle(vector1, vector2):
    v1_u = normalize(vector1)
    v2_u = normalize(vector2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def global_euler_deltas(tm1,tm2):
    ea1 = tm1.euler_angles()
    ea2 = tm2.euler_angles()
    deltas = arr(ea2)-arr(ea1)
    return deltas

#--------------------- Module Functions: xyzframe ----------------------------#
def makeframe(df_or_pointlist_or_csvfilepath):
    val = df_or_pointlist_or_csvfilepath
    if type(val)==str:
        df = pd.read_csv(val)
        return frame.copy(df)
    elif type(val)==dict:
        df = pd.DataFrame(val)
        return frame.copy(df)
    else:
        return frame.copy(val)
