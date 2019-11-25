# circulation.py takes an area integral of azimuthally averaged azimuthal vorticity over the thermal to calclate the circulation


import xarray as xr
import numpy as np
import matplotlib
from faceted import faceted
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from lighten_color import lighten_color


lam_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/lam/azi_vort_phi/lam*.nc',concat_dim='t').omega_phi
lam_track = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/thermal_boundary.nc')
lam_dr = lam_ds.r.values[1] - lam_ds.r.values[0]
lam_dz = lam_ds.z.values[1] - lam_ds.z.values[0]

turb_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/azi_vort_phi/turb*.nc',concat_dim='t').omega_phi
turb_track = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/thermal_boundary.nc')
turb_dr = turb_ds.r.values[1] - turb_ds.r.values[0]
turb_dz = turb_ds.z.values[1] - turb_ds.z.values[0]

lam_masked = lam_ds.where(lam_ds.r<lam_track.r)
turb_masked = turb_ds.where(turb_ds.r<turb_track.r)

lam_circ = (lam_masked.sum(('r','z')) * lam_dr * lam_dz).compute()
turb_circ = (turb_masked.sum(('r','z')) * turb_dr * turb_dz).compute()

t0=4

'''RESCALE FACTOR IS 10*sqrt(10)'''

# circulation as a function of time

matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally

tick_locator = ticker.MaxNLocator(nbins=2)
fig, axes = faceted(1, 1, width=5, aspect=1/1.618, bottom_pad=0.75, internal_pad=(0.,0.),sharex=False,sharey=False)

axes[0].plot(lam_circ.t/t0,lam_circ*10*np.sqrt(10),color='darkgoldenrod',linewidth=2.5,label=r'$\mathrm{Re}=630$')
axes[0].plot(turb_circ.t/t0,turb_circ*10*np.sqrt(10),color='rebeccapurple',linewidth=2.5,label=r'$\mathrm{Re} = 6300$')

axes[0].set_xlim([0,5])
axes[0].set_ylim([0,2.3])

axes[0].set_yticks([0,.5,1,1.5,2])
axes[0].set_yticklabels(['0','0.5','1','1.5','2'],fontsize=15)

axes[0].set_xticks([0,1,2,3,4,5])
axes[0].set_xticklabels(['0','1','2','3','4','5'],fontsize=15)

axes[0].xaxis.set_tick_params(direction='in')
axes[0].yaxis.set_tick_params(direction='in')

axes[0].set_xlabel(r'$\tau$',fontsize=18)
axes[0].set_ylabel(r'$\Gamma$',fontsize=22)

plt.legend(loc='lower right',fontsize=12,frameon=False)

plt.show()
