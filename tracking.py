# tracking.py creates a dataset which contains the radius (r) as a function of time and height,
# and the thermal midpoint (x_c, y_c) as functions of time


import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/no_g/data/slice*.nc',concat_dim='t')
contour = np.loadtxt('/work/bnm/buoyant_entrainment/data/no_g/1e4_sim5_no_g/contour_flux.dat')
midpoint = np.loadtxt('/work/bnm/buoyant_entrainment/data/no_g/1e4_sim5_no_g/thermal_midpoint_1e4_g0.dat')
rout = '/work/bnm/buoyant_entrainment/data/no_g/1e4_sim5_no_g/thermal_boundary.nc'
mout = '/work/bnm/buoyant_entrainment/data/no_g/1e4_sim5_no_g/mask.nc'

tracking = xr.Dataset({'r': (['t', 'z'],  contour),
                 'x_c': (['t'], midpoint[1]),
                 'y_c': (['t'], midpoint[2])},
             coords={'t': (['t'], ds.t.values[:-1]), #remove last element from array
                     'z': (['z'], ds.z.values)})

tracking.to_netcdf(rout,engine='scipy')

tracking = xr.open_dataset(rout)

# convert thermal boundary dataset to have same dimensions as rho, u, v, w, ...
r, foo, bar = xr.broadcast(tracking.r,ds.x,ds.y)
delta_x = ds.x - tracking.x_c
delta_y = ds.y - tracking.y_c

mask = xr.ufuncs.sqrt(delta_x ** 2 + delta_y ** 2) < r
mask.name = 'mask'
mask.to_netcdf(mout,engine='scipy')
