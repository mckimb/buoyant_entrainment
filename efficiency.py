# efficiency.py plots the entrainment efficiency e from the turbulent and laminar simulations with g == 0 and g != 0


import xarray as xr
import numpy as np
import matplotlib
from faceted import faceted
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lighten_color import lighten_color

import os.path

class SimulationData:

  def __init__(self,dir):

    self.t,self.z_ct,self.z_new,self.w,self.r,self.vol = load_data_sqrt(dir)

    if os.path.isfile("%s/mean_rho.dat" %dir):
      self.mean_rho = np.loadtxt('%s/mean_rho.dat' %dir)

    if os.path.isfile("%s/mass_below.dat" %dir):
      data = np.loadtxt("%s/mass_below.dat" %dir)
      self.mass_below = data[0,:]
      self.mass_way_below = data[1,:]
    else:
      self.mass_below = self.t*0
      self.mass_way_below = self.t*0

    self.total_mass = -4*np.pi/3*(0.05)**3

    self.calculate_dvoldt()
    self.calculate_mass()
    self.calculate_dMdt()
    self.calculate_efficiency()
    self.calculate_entrainment()

    if os.path.isfile('%s/mean_c.dat' %dir):
      data = np.loadtxt("%s/mean_c.dat" %dir)
      self.c_in = data[0,:]
      self.c_out = data[1,:]
      self.c_below = data[2,:]
      self.c_z0 = data[3,:]
      self.calculate_detrainment()

  def calculate_detrainment(self):
    c_len = len(self.c_z0)
    self.t_d = self.t[5:c_len:10]
    self.z_d = self.z_ct[5:c_len:10]
    if c_len == 172: c_len=170
    self.detrainment = (self.c_z0[9:c_len:10] - self.c_z0[:c_len:10])/(self.z_ct[9:c_len:10] - self.z_ct[:c_len:10])/(self.c_in[5:c_len:10]+self.c_out[5:c_len:10])

  def calculate_dvoldt(self):
    self.dvoldt = np.gradient(self.vol,self.t)

  def calculate_mass(self):
    self.mass = self.vol*self.mean_rho

  def calculate_dMdt(self):
    self.dMdt = np.zeros(len(self.t))
    i_ave = 20
    self.dMdt[:i_ave] = self.mass_below[:i_ave]/self.t[:i_ave]
    self.dMdt[i_ave:] = (self.mass_below[:-i_ave]-self.mass_below[i_ave:])/(self.t[:-i_ave]-self.t[i_ave:])

  def calculate_efficiency(self):
    self.efficiency = self.dvoldt/self.vol/self.w*self.r

  def calculate_entrainment(self):
    self.entrainment = self.dvoldt/self.vol/self.w

class DataCollection:

  def __init__(self):
    self.data_list = []

  def add(self,dir):
    self.data_list.append(SimulationData(dir))

  def average(self):
    self.t = self.data_list[0].t

    def do_average(quantity_list):
      ave = 0*getattr(self.data_list[0],quantity_list[0])
      for data in self.data_list:
        single = 0*ave+1
        for quantity in quantity_list:
          single *= getattr(data,quantity)
        ave += single
      return ave/len(self.data_list)

    self.ave_efficiency = do_average(['efficiency'])
    self.ave_entrainment = do_average(['entrainment'])
    self.ave_r = do_average(['r'])
    self.ave_w = do_average(['w'])
    self.ave_mean_rho = do_average(['mean_rho'])
    self.ave_mass = do_average(['mean_rho','vol'])
    self.ave_vol = do_average(['vol'])
    self.ave_mass_below = do_average(['mass_below'])
    self.ave_mass_way_below = do_average(['mass_way_below'])

def z_sqrt_func(t, a, t0, z0):
  return a*np.abs(t)**(1/2) + z0

def w_sqrt_func(t, a, t0, z0):
  return a/2/np.abs(t)**(1/2)

from scipy.optimize import curve_fit

def load_data_sqrt(dir):
  r_array = np.loadtxt('%s/contour_flux.dat' %dir)

  z_data = np.loadtxt('%s/cloud_top.txt' %dir)
  t = z_data[0]
  z_ct = z_data[1]

  mask = (z_ct < 1.8) & (z_ct>0.5)
  popt, pcov = curve_fit(z_sqrt_func, t[mask], z_ct[mask])
  z_new = z_sqrt_func(t,*popt)

  w = w_sqrt_func(t,*popt)

  dz = 2/r_array.shape[1]
  vol = np.sum(dz*np.pi*r_array**2,axis=1)
  r = np.max(r_array,axis=1)

  return (t,z_ct,z_new,w,r,vol)

t_turb_g, z_ct_turb_g, z_new_turb_g, w_turb_g, r_turb_g, vol_turb_g = load_data_sqrt('/work/bnm/buoyant_entrainment/data/no_g/1e4_sim5/')
# t_lam_g, z_ct_lam_g, z_new_lam_g, w_lam_g, r_lam_g, vol_lam_g = load_data_sqrt('/work/bnm/buoyant_entrainment/data/no_g/1e4_sim5_no_g/')
t_turb_nog, z_ct_turb_nog, z_new_turb_nog, w_turb_nog, r_turb_nog, vol_turb_nog = load_data_sqrt('/work/bnm/buoyant_entrainment/data/no_g/1e4_sim5_no_g/')
t_lam_nog, z_ct_lam_nog, z_new_lam_nog, w_lam_nog, r_lam_nog, vol_lam_nog = load_data_sqrt('/work/bnm/buoyant_entrainment/data/no_g/1e3_sim5_no_g/')

entrain_turb_g = np.gradient(vol_turb_g,t_turb_g)/(w_turb_g*vol_turb_g)
# entrain_lam_g = np.gradient(vol_lam_g,t_lam_g)/(w_lam_g*vol_lam_g)
entrain_turb_nog = np.gradient(vol_turb_nog,t_turb_nog)/(w_turb_nog*vol_turb_nog)
entrain_lam_nog = np.gradient(vol_lam_nog,t_lam_nog[:95])/(w_lam_nog[:95]*vol_lam_nog[:95])

# If I later have doubts about this data, go to
# /work/bnm/buoyant_entrainment/exploration/tracking_and_mask.ipynb for mask
# /work/bnm/bouyant_entrainment/exploration/entrainment for volume averages and height
lam_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/lam/rho_u_v_w/slice*.nc',concat_dim='t')
lam_omega = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/vort_phi/azi_lam_vort.nc')
lam_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/laminar_mask.nc',engine='scipy')
lam_track = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/thermal_boundary.nc')
lam_w_avg = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/w_avg.nc').w
lam_rho_avg = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/rho_avg.nc')
lam_r = lam_track.r.max('z')
lam_z = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/z_top.nc').z

t0 = 4

lrho0 = lam_rho_avg.sel(t=t0,method='nearest').values
lw0 = lam_w_avg.sel(t=t0,method='nearest').values
lr0 = lam_r.sel(t=t0,method='nearest').values
lz0 = lam_z.sel(t=t0,method='nearest').values

lam_dz = lam_track.z.values[2] - lam_track.z.values[1]
lam_dt = lam_track.t.values[2] - lam_track.t.values[1]
lam_vol = lam_dz*np.pi*(lam_track.r**2).sum('z')
lam_dvol_dt = (lam_vol.shift(t=-1) - lam_vol.shift(t=+1))/(2*lam_dt)
lam_entrain = 1/(lam_w_avg*lam_vol)*lam_dvol_dt

# efficiency as a function of time

matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally

tick_locator = ticker.MaxNLocator(nbins=2)
fig, axes = faceted(1, 1, width=5, aspect=1/1.618, bottom_pad=0.75, internal_pad=(0.,0.),sharex=False,sharey=False)

lam_eff_no_g = entrain_lam_nog[:95]*r_lam_nog[:95]
turb_eff_no_g = entrain_turb_nog*r_turb_nog

# remove the clearly erroneous value (caused by r close to 0)
lam_eff_no_g[1] = np.nan
turb_eff_no_g[1] = np.nan

# axes[0].plot(turb_time/t0,(0.489179)*np.ones_like(turb_time),linewidth=1,color='rebeccapurple')
# axes[0].plot(lam_time/t0,(0.407886)*np.ones_like(lam_time),linewidth=1,color='darkgoldenrod')

# axes[0].plot(turb_time/t0,(0.489179-.35)*np.ones_like(turb_time),linewidth=1,color='rebeccapurple')
# axes[0].plot(lam_time/t0,(0.407886-.35)*np.ones_like(lam_time),linewidth=1,color='darkgoldenrod')

axes[0].plot(lam_entrain.t/t0,lam_entrain*lam_r,color=lighten_color('darkgoldenrod'),linewidth=2.5,linestyle='dashed',label=r'$\mathrm{Eq. 14}$')
axes[0].plot(t_turb_g/t0,entrain_turb_g*r_turb_g,color=lighten_color('rebeccapurple'),linestyle='dashed',linewidth=2.5,label=r'$\mathrm{Re} = 630$ $\mathrm{simulation}$')

axes[0].plot(lam_time/t0,lam_eff,color='darkgoldenrod',linewidth=2.5)
axes[0].plot(turb_time/t0,turb_eff,color='rebeccapurple',linewidth=2.5)



# axes[0].plot(t_lam_nog[:95]/t0,lam_eff_no_g,color='darkgoldenrod',linewidth=2.5,label=r'$\mathrm{Eq. 14}$')
# axes[0].plot(t_turb_nog/t0,turb_eff_no_g,color='rebeccapurple',linewidth=2.5,label=r'$\mathrm{Re} = 6300$ $\mathrm{simulation}$')

# axes[0].plot(r,entrain,'r')
# axes[0].plot(r,3*r0/(2*w0*t0*r),'b')


axes[0].set_xlim([0,5])
axes[0].set_ylim([0,1])

axes[0].set_yticks([0,0.2,0.4,0.6,0.8,1])
axes[0].set_yticklabels(['0','0.2','0.4','0.6','0.8','1'],fontsize=15)

axes[0].set_xticks([0,1,2,3,4,5])
axes[0].set_xticklabels(['0','1','2','3','4','5'],fontsize=15)

axes[0].xaxis.set_tick_params(direction='in')
axes[0].yaxis.set_tick_params(direction='in')

axes[0].set_xlabel(r'$\tau$',fontsize=18)
axes[0].set_ylabel(r'$e$',fontsize=22)

# plt.legend(loc='upper right',fontsize=12)

lines = [Line2D([0],[0],color=lighten_color('rebeccapurple'),linestyle='dashed',linewidth=2.5),
         Line2D([0],[0],color='rebeccapurple',linewidth=2.5),
        Line2D([0],[0],color=lighten_color('darkgoldenrod'),linestyle='dashed',linewidth=2.5),
        Line2D([0],[0],color='darkgoldenrod',linewidth=2.5)]

labels = [r'$\mathrm{Re} = 6300$',r'$\mathrm{Re} = 6300, \ \mathrm{g}=0$',r'$\mathrm{Re} = 630$',r'$\mathrm{Re} = 630, \ \mathrm{g}=0$']

plt.legend(lines, labels,loc='upper right', ncol=2,fontsize=12.5,frameon=False)

plt.show()
