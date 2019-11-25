# entrainment.py computes the entrainment by using Eq. 15 from the paper.


import xarray as xr
import numpy as np
import matplotlib
from faceted import faceted
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from lighten_color import lighten_color


lam_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/lam/rho_u_v_w/slice*.nc',concat_dim='t')
lam_omega = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/vort_phi/azi_lam_vort.nc')
lam_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/laminar_mask.nc',engine='scipy')
lam_track = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/thermal_boundary.nc')
lam_w_avg = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/w_avg.nc').w
lam_rho_avg = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/rho_avg.nc')
lam_r = lam_track.r.max('z')
lam_z = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/z_top.nc').z

turb_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/rho_u_v_w/slice*.nc', concat_dim='t')
turb_omega = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/vort_phi/turb*.nc',concat_dim='t').omega_phi
turb_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/mask_correct.nc',engine='scipy')
turb_track = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/thermal_boundary.nc')
turb_w_avg = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/w_avg.nc').w
turb_rho_avg = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/rho_avg.nc')
turb_r = turb_track.r.max('z')
turb_z = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/z_top.nc').z

# initial values
t0 = 4

lrho0 = lam_rho_avg.sel(t=t0,method='nearest').values
lw0 = lam_w_avg.sel(t=t0,method='nearest').values
lr0 = lam_r.sel(t=t0,method='nearest').values
lz0 = lam_z.sel(t=t0,method='nearest').values

trho0 = turb_rho_avg.sel(t=t0,method='nearest').values
tw0 = turb_w_avg.sel(t=t0,method='nearest').values
tr0 = turb_r.sel(t=t0,method='nearest').values
tz0 = turb_z.sel(t=t0,method='nearest').values

lam_dz = lam_track.z.values[2] - lam_track.z.values[1]
turb_dz = turb_track.z.values[2] - turb_track.z.values[1]

lam_dt = lam_track.t.values[2] - lam_track.t.values[1]
turb_dt = turb_track.t.values[2] - turb_track.t.values[1]

lam_vol = lam_dz*np.pi*(lam_track.r**2).sum('z')
turb_vol = turb_dz*np.pi*(turb_track.r**2).sum('z')

lam_dvol_dt = (lam_vol.shift(t=-1) - lam_vol.shift(t=+1))/(2*lam_dt)
turb_dvol_dt = (turb_vol.shift(t=-1) - turb_vol.shift(t=+1))/(2*turb_dt)

# entrainment as a function of time
lam_entrain = 1/(lam_w_avg*lam_vol)*lam_dvol_dt
turb_entrain = 1/(turb_w_avg*turb_vol)*turb_dvol_dt

# entrainment as a function of time
matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally
matplotlib.rcParams['axes.titlepad'] = 31
tick_locator = ticker.MaxNLocator(nbins=2)

fig, axes = faceted(2, 1, width=5, aspect=1/1.618, bottom_pad=0.75, internal_pad=(0.,0.),sharex=False,sharey=False)

xticklabels=[[],['0.5','1','1.5','2','2.5','3']]
xticks = [[0.05,0.1,0.15,0.2,0.25,0.3],[0.05,0.1,0.15,0.2,0.25,0.3]]

for i, ax in enumerate(axes):
    axes[0].plot(lam_r,lam_entrain,color=lighten_color('darkgoldenrod'),linewidth=2.5,label=r'$\mathrm{Re} = 630$ $\mathrm{simulation}$')
    axes[1].plot(turb_r,turb_entrain,color=lighten_color('rebeccapurple'),linewidth=2.5,label=r'$\mathrm{Re} = 6300$ $\mathrm{simulation}$')
#     axes[0].plot(lam_r,lam_entrain,color=lighten_color('darkgoldenrod'),marker='.',linewidth=1.5,label=r'$\mathrm{Re} = 630$ $\mathrm{simulation}$')
#     axes[1].plot(turb_r,turb_entrain,color=lighten_color('rebeccapurple'),marker='.',linewidth=1.5,label=r'$\mathrm{Re} = 6300$ $\mathrm{simulation}$')
    axes[0].plot(lam_r,3*lr0/(2*lw0*t0*lam_r),'darkgoldenrod',linewidth=2.5,linestyle='dashed',label=r'$\mathrm{Eq. 14}$')
    axes[1].plot(turb_r,3*tr0/(2*tw0*t0*turb_r),'rebeccapurple',linewidth=2.5,linestyle='dashed',label=r'$\mathrm{Eq. 14}$')

    axes[i].set_xlim([0.05,.3])
    axes[i].set_ylim([0,6])

    axes[i].set_yticks([0,1,2,3,4,5])
    axes[i].set_yticklabels(['0','0.1','0.2','0.3','0.4','0.5'],fontsize=15)

    axes[i].set_xticks(xticks[i])
    axes[i].set_xticklabels(xticklabels[i],fontsize=15)

    axes[i].xaxis.set_tick_params(direction='in')
    axes[i].yaxis.set_tick_params(direction='in')

    axes[i].set_xlabel(r'$a$',fontsize=18)
    axes[i].set_ylabel(r'$\epsilon$',fontsize=22)

# plt.legend(loc='upper right',fontsize=12)

lines2 = [Line2D([0],[0],color='rebeccapurple',linewidth=2.5,linestyle='dashed'),
        Line2D([0],[0],color=lighten_color('rebeccapurple'),linewidth=2.5)]

lines1 = [Line2D([0],[0],color='darkgoldenrod',linewidth=2.5,linestyle='dashed'),
        Line2D([0],[0],color=lighten_color('darkgoldenrod'),linewidth=2.5)]

labels2 = [r'$\mathrm{Eq. 14}$',r'$\mathrm{Re} = 6300$ $\mathrm{sim}$']
labels1 = [r'$\mathrm{Eq. 14}$',r'$\mathrm{Re} = 630$ $\mathrm{sim}$']

leg = plt.legend(lines2, labels2,ncol=1,loc='upper right',bbox_to_anchor=(1, 1),fontsize=12.5,frameon=False)
ax.add_artist(leg)
plt.legend(lines1, labels1,ncol=1,loc='upper right',bbox_to_anchor=(1, 2),fontsize=12.5,frameon=False)

plt.show()
