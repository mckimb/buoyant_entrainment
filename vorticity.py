print('IMPORTING LIBRARIES')
import xarray as xr
import numpy as np
from mpi4py import MPI
import argparse
import shlex, subprocess

'''Note: I have commented out the calculation of omega_phi and the z-component of vorticity sine neither are necessary'''

print('PARSING ARGUMENTS')
parser = argparse.ArgumentParser(description='calculate vorticity data')
parser.add_argument('--time', metavar='time', help='time slice selected')
args = parser.parse_args()

time  = float(args.time)
print('time='+str(time))
print('type(time)='+str(type(time)))

data_input = '/work/bnm/buoyant_entrainment/no_g/data/slice*.nc'
track_input = '/work/bnm/buoyant_entrainment/no_g/tracking.nc'
output = '/work/bnm/buoyant_entrainment/no_g/data/vorticity_xyz/'
output2 = '/work/bnm/buoyant_entrainment/no_g/data/vorticity_xyz_cart/'
output3 = '/work/bnm/buoyant_entrainment/no_g/data/vorticity_xyz_lap/'


num = '{:03d}'.format(int(time*10))
print('num='+num)
print('type(num)='+str(type(num)))

fout = output+'slice'+num+'.nc'
print('fout='+fout)

fout2 = output2+'slice'+num+'.nc'
print('fout2='+fout2)

fout3 = output3+'slice'+num+'.nc'
print('fout3='+fout3)

print('Loading Tracking')
tracking = xr.open_dataset(track_input).sel(t=time,method='nearest')

print('LOADING input data')
ds = xr.open_mfdataset(data_input, concat_dim='t').sel(t=time,method='nearest')

dx = ds.x[1]-ds.x[0]
dy = ds.y[1]-ds.y[0]
dz = ds.z[1]-ds.z[0]

# x component of vorticity
dw_dy = (ds.w.shift(y=-1) - ds.w)/dy
dv_dz = (ds.v.shift(z=-1) - ds.v)/dz

# y component of vorticity
dw_dx = (ds.w.shift(x=-1) - ds.w)/dx
du_dz = (ds.u.shift(z=-1) - ds.u)/dz

'''
# z component of vorticity
dv_dx = (ds.v.shift(x=-1) - ds.v)/dx
du_dy = (ds.u.shift(y=-1) - ds.u)/dy
'''

omega_x = dw_dy - dv_dz
omega_y = du_dz - dw_dx
'''
omega_z = dv_dx - du_dy
'''

omega_x = omega_x.rename('omega_x')
omega_y = omega_y.rename('omega_y')
'''
omega_z = omega_z.rename('omega_z')
'''
'''
omega = xr.merge([omega_x,omega_y,omega_z])
'''
omega = xr.merge([omega_x,omega_y]).compute()

omega.to_netcdf(fout2)

print('CALCULATING 2ND DERIVATIVES')
lap_om_x = ((omega_x.shift(x=-1) - 2*omega_x + omega_x.shift(x=1)) / dx**2
         +(omega_x.shift(y=-1) - 2*omega_x + omega_x.shift(y=1)) / dy**2
         +(omega_x.shift(z=-1) - 2*omega_x + omega_x.shift(z=1)) / dz**2).compute()
lap_om_y = ((omega_y.shift(x=-1) - 2*omega_y + omega_y.shift(x=1)) / dx**2
         +(omega_y.shift(y=-1) - 2*omega_y + omega_y.shift(y=1)) / dy**2
         +(omega_y.shift(z=-1) - 2*omega_y + omega_y.shift(z=1)) / dz**2).compute()

print('CALCULATING radius COORDINATE')
r = np.sqrt((omega.x-tracking.x_c)**2 + (omega.y-tracking.y_c)**2)


'''
# now get azimuthal vorticity
a1 = omega.omega_x * (omega.y-tracking.y_c)
a2 = a1 * -1
b1 = omega.omega_y * (omega.x-tracking.x_c)
omega_phi = (a2 + b1) / r

print('SAVING DATA TO '+fout)
omega_phi.to_netcdf(fout)
'''

# now get azimuthal vorticity laplacian
a1 = lap_om_x * (omega.y-tracking.y_c)
a2 = a1 * -1
b1 = lap_om_y * (omega.x-tracking.x_c)
lap_omega_phi = (a2 + b1) / r

print('SAVING DATA TO '+fout3)
lap_omega_phi.to_netcdf(fout3)