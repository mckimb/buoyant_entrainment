# h5_to_netcdf.py takes the output of dedalus simulations which are in .h5 and converts them to netcdf format


import xarray as xr
import numpy as np
import os
import glob
import h5py
from tqdm import tqdm
import argparse

def h5_to_ds(data):

    rho = data['tasks']['rho']
    rho = rho.value[:,:,:,:]

    u = data['tasks']['u']
    u = u.value[:,:,:,:]

    v = data['tasks']['v']
    v = v.value[:,:,:,:]

    w = data['tasks']['w']
    w = w.value[:,:,:,:]


    t = data['scales']['sim_time'].value
    x = data['scales']['x']['1.0'].value
    y = data['scales']['y']['1.0'].value
    z = data['scales']['z']['1.0'].value

    ds = xr.Dataset({'rho': (['t', 'x', 'y', 'z'],  rho),
                     'u': (['t', 'x', 'y', 'z'],  u),
                     'v': (['t', 'x', 'y', 'z'],  v),
                     'w': (['t', 'x', 'y', 'z'],  w)},
                 coords={'x': (['x'], x),
                         'y': (['y'], y),
                         'z': (['z'], z),
                         't': (['t'], t)})
    return ds

parser = argparse.ArgumentParser(description='chop up h5 data into individual netcdf files')

parser.add_argument('--fpath', metavar='fpath', help='path to the h5 file')

args = parser.parse_args()

fpath  = args.fpath

print("Input Path: ", fpath)

data = h5py.File(fpath, 'r')

print("Reading in data...")
ds = h5_to_ds(data)

print("Dataset complete")

batch = fpath[37:][:-3]

print("Writing out files")
for t_ind in tqdm( range(len(ds.t)) ):
    base = '/work/bnm/buoyant_entrainment/no_g/data/slice_'
    fout = base + batch.zfill(3) +'_' + str(t_ind).zfill(3)+'.nc'

    ds_slice = ds.isel(t=t_ind)
    ds_slice.to_netcdf(fout,engine='scipy')
    print("Wrote", fout)

print("Finished")
