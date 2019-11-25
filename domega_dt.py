# domega_dt.py takes the azimuthally vorticity computed previously and azimuthally averages it and then takes the time derivative


import xarray as xr
import numpy as np
import matplotlib
from faceted import faceted
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
matplotlib.rcParams['figure.dpi']= 600

lam_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/lam/rho_u_v_w/slice*.nc',concat_dim='t')
lam_tracking = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/thermal_boundary.nc').sel(t=10,method='nearest')
lam_rho_azi_out ='/work/bnm/buoyant_entrainment/data/lam/mask/rho_azi.nc'

turb_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/rho_u_v_w/slice*.nc',concat_dim='t')
turb_tracking = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/thermal_boundary.nc').sel(t=10,method='nearest')
turb_rho_azi_out ='/work/bnm/buoyant_entrainment/data/turb/mask/rho_azi.nc'

def azimuthal(r0, radius, dr, data):
    azi_mask = radius.where(r0-dr < radius).where(radius < r0+dr)
    data_axim = data.where(np.isfinite(azi_mask)).mean(('x','y'))
    data_axim['r'] = r0
    return data_axim

lam_rho = lam_ds.rho.sel(t=10,method='nearest')
turb_rho = turb_ds.rho.sel(t=10,method='nearest')

lam_radius = np.sqrt((lam_rho.x-lam_tracking.x_c)**2 + (lam_rho.y-lam_tracking.y_c)**2)
turb_radius = np.sqrt((turb_rho.x-turb_tracking.x_c)**2 + (turb_rho.y-turb_tracking.y_c)**2)

lam_dr = lam_rho.x.diff('x').values[0]
turb_dr = turb_rho.x.diff('x').values[0]

lam_rho_azi = xr.concat([azimuthal(r0, lam_radius, lam_dr/2, lam_rho) for r0 in np.arange(0,.5+lam_dr, lam_dr)], dim='r').compute()
turb_rho_azi = xr.concat([azimuthal(r0, turb_radius, turb_dr/2, turb_rho) for r0 in np.arange(0,.5+turb_dr, turb_dr)], dim='r').compute()

lam_Domega_Dt = (lam_rho_azi.shift(r=-1) - lam_rho_azi.shift(r=+1))/(lam_rho_azi.r.shift(r=-1) - lam_rho_azi.r.shift(r=+1))
turb_Domega_Dt = (turb_rho_azi.shift(r=-1) - turb_rho_azi.shift(r=+1))/(turb_rho_azi.r.shift(r=-1) - turb_rho_azi.r.shift(r=+1))

lam_om_merge = np.concatenate((np.flip(lam_Domega_Dt.values,0),lam_Domega_Dt.values),axis=0)
lam_r_merge = np.concatenate((np.flip(lam_Domega_Dt.r.values,0)*-1,lam_Domega_Dt.r.values),axis=0)

turb_om_merge = np.concatenate((np.flip(turb_Domega_Dt.values,0),turb_Domega_Dt.values),axis=0)
turb_r_merge = np.concatenate((np.flip(turb_Domega_Dt.r.values,0)*-1,turb_Domega_Dt.r.values),axis=0)

# eliminate r = 0 overlap
lam_om_merge_0 = lam_om_merge[np.where(lam_r_merge!=0)[0]]
lam_r_merge_0 = lam_r_merge[np.where(lam_r_merge!=0)[0]]

turb_om_merge_0 = turb_om_merge[np.where(turb_r_merge!=0)[0]]
turb_r_merge_0 = turb_r_merge[np.where(turb_r_merge!=0)[0]]

lam_Domega_new = xr.DataArray(lam_om_merge_0,coords=[('r',lam_r_merge_0),('z',lam_Domega_Dt.z.values)]).T
turb_Domega_new = xr.DataArray(turb_om_merge_0,coords=[('r',turb_r_merge_0),('z',turb_Domega_Dt.z.values)]).T

# now lets add the thermal boundary
lam_mask = xr.ufuncs.fabs(lam_Domega_new.r) < lam_tracking.r
turb_mask = xr.ufuncs.fabs(turb_Domega_new.r) < turb_tracking.r

matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally
tick_locator = ticker.MaxNLocator(nbins=2)

xlabels = [r'$x$',r'$x$']
ylabels = [r'$z$','']

datum =  [lam_Domega_new/10, turb_Domega_new/10]
labelsa = [r'$\mathrm{Re} = 630$',r'$\mathrm{Re} = 6300$']
labelsb = ['', r'$\langle \frac{D \omega_\phi}{Dt} \rangle$']

boundaries = [lam_mask.T, turb_mask.T]

max_vals = [0.08,0.08]

# clr = 'PuOr_r'
# clr = 'BrBG_r'
clr = 'RdBu_r'
# clr = 'seismic'

fig, axes, caxes = faceted(1,2,width=5,aspect=2.0,
                   bottom_pad=0.75,cbar_mode='single',cbar_pad=0.,
                    internal_pad=(0.,0.5),cbar_location='top',cbar_short_side_pad=0.)

plt.suptitle(r'$\langle \frac{D \omega_\phi}{Dt} \rangle$',fontsize=20,y=1.10)


for i, ax in enumerate(axes):
    c = datum[i].plot(ax=ax,add_colorbar=False,label='a',vmax = max_vals[i],cmap=clr)
    boundaries[i].plot.contour(ax=ax,colors='k',linewidths=1.5)
    ax.set_xlim([-0.5,0.5])
    ax.set_title('')
    ax.set_xlabel(xlabels[i],fontsize=22)
    ax.set_ylabel(ylabels[i],fontsize=22)
    ax.text(0.25, .92, labelsa[i],transform=ax.transAxes,fontsize=18)
#     ax.text(0.61, .90, labelsb[i],transform=ax.transAxes,fontsize=24)
    cb = plt.colorbar(c, cax=caxes,orientation='horizontal')
    cb.locator = tick_locator
    ax.yaxis.set_ticks([0,0.5,1,1.5,2])
    ax.yaxis.set_ticklabels(['0','5','10','15','20'],fontsize=15)
    ax.xaxis.set_ticks([-.25,0,.25])
    ax.xaxis.set_ticklabels(['-2.5','0','2.5'],fontsize=15)
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    cb.update_ticks()
    caxes.xaxis.tick_top()
    caxes.xaxis.set_tick_params(direction='in')
    caxes.xaxis.set_ticklabels([-0.08,0,0.08],fontsize=12)

plt.show()
