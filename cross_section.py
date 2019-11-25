# cross_section.py makes figure 1 (the cross section of rho (density anomaly) and omega (vorticity) for
# laminar and turbululent thermals)


import xarray as xr
import numpy as np
import matplotlib
import colorcet
from faceted import faceted
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi']= 600

lam_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/lam/rho_u_v_w/slice*.nc',concat_dim='t')
lam_omega = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/vort_phi/azi_lam_vort.nc')
lam_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/laminar_mask.nc',engine='scipy')
lam_tracking = xr.open_dataset('/work/bnm/buoyant_entrainment/data/lam/mask/thermal_boundary.nc')
turb_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/rho_u_v_w/slice*.nc', concat_dim='t')
turb_omega = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/vort_phi/turb*.nc',concat_dim='t').omega_phi
turb_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/mask_correct.nc',engine='scipy')
turb_tracking = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/thermal_boundary.nc')
# turb_mask = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/mask/turb_mask*.nc', concat_dim='t').mask

t0 = 10
lam_y = lam_tracking.y_c.sel(t=10,method='nearest').values
lam_rho = lam_ds.rho.T.sel(t=t0,x=lam_y,method='nearest')
lam_vort = lam_omega.T.sel(t=t0,x=lam_y,method='nearest')
lam_boundary = lam_mask.T.sel(t=t0,x=lam_y,method='nearest')

turb_y = turb_tracking.y_c.sel(t=10,method='nearest').values
turb_rho = turb_ds.rho.sel(t=t0,y=turb_y,method='nearest')
# turb_vort = turb_omega.sel(t=t0,x=0,method='nearest').rolling(y=15, center=True).mean().rolling(z=15,center=True).mean()
turb_vort = turb_omega.sel(t=t0,y=turb_y,method='nearest')
turb_boundary = turb_mask.sel(t=t0,y=turb_y,method='nearest')


'''NOTE: RATHER THAN RESCALING THE DATA, I AM RESCALING THE AXES
         ACTUALLY I HAD TO RESCALE
x_new = 10 * x_old
t_new = sqrt(10) * t_old
rho_new = rho_old
omega_new = 1/sqrt(10) * omega_old
gamma_new = 10 * sqrt(10) * gamma_old
P_new = sqrt(10) * P_old
epsilon_new = 1/10 * epsilon_old
'''

# plot contours of dvort_dt
lam_Domega_new = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/domega_dt.nc')
turb_Domega_new = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/domega_dt.nc')

# but only do it within the thermal

# now lets add the thermal boundary
# lam_mask = xr.ufuncs.fabs(lam_Domega_new.r - lam_tracking.x_c.sel(t=t0,method='nearest')) < lam_tracking.r.sel(t=t0,method='nearest')
# turb_mask = xr.ufuncs.fabs(turb_Domega_new.r - turb_tracking.x_c.sel(t=t0,method='nearest')) < turb_tracking.r.sel(t=t0,method='nearest')

# lam_Domega = lam_Domega_new.where(lam_mask)
# turb_Domega = turb_Domega_new.where(turb_mask)


# now remove turbulent contours
turb_mask = xr.ufuncs.fabs(turb_Domega_new.r) < 0
lam_Domega = lam_Domega_new
turb_Domega = turb_Domega_new.where(turb_mask)

matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally
tick_locator = ticker.MaxNLocator(nbins=2)

xlabels1, xlabels2 = ['',''], [r'$x$',r'$x$']
ylabels1, ylabels2 = [r'$z$',''], [r'$z$','']

datum1, datum2 = [lam_rho,turb_rho], [lam_vort/np.sqrt(10), turb_vort/np.sqrt(10)]
labelsa1 = [r'$\mathrm{Re} = 630$',r'$\mathrm{Re} = 6300$']
labelsb1 = ['', r'$\rho^{\prime}$']
labelsa2 = [r'$\mathrm{Re} = 630$',r'$\mathrm{Re} = 6300$']
labelsb2 = ['', r'$\omega_{\phi}$']

boundaries1, boundaries2 = [lam_boundary, turb_boundary.T], [lam_boundary, turb_boundary.T]

contours =  [lam_Domega/10, turb_Domega/10]


max_vals1 = [.05,.05]
max_vals2 = [4, 4]

clr = 'RdBu_r'

fig, axes, caxes = faceted(1,2,width=5,aspect=2.0,
                   bottom_pad=0.75,cbar_mode='single',cbar_pad=0.,
                    internal_pad=(0.,0.5),cbar_location='top',cbar_short_side_pad=0.)

plt.suptitle(r'$\rho^\prime$',fontsize=24,y=1.06)


for i, ax in enumerate(axes):
    c = datum1[i].plot(ax=ax,add_colorbar=False,label='a',vmax = max_vals1[i],cmap=clr)
    boundaries1[i].plot.contour(ax=ax,colors='k',linewidths=1.5)
    ax.set_title('')
    ax.set_xlabel(xlabels1[i],fontsize=22)
    ax.set_ylabel(ylabels1[i],fontsize=22)
    ax.text(0.05, .92, labelsa1[i],transform=ax.transAxes,fontsize=18)
#     ax.text(0.80, .92, labelsb1[i],transform=ax.transAxes,fontsize=24)
    cb = plt.colorbar(c, cax=caxes,orientation='horizontal')
    cb.locator = tick_locator
    ax.yaxis.set_ticks([0,0.5,1,1.5,2])
    ax.yaxis.set_ticklabels(['0','5','10','15','20'],fontsize=15) #these have been rescaled
    ax.xaxis.set_ticks([-.25,0,.25])
    ax.xaxis.set_ticklabels(['','',''])
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    cb.update_ticks()
    caxes.xaxis.tick_top()
    caxes.xaxis.set_tick_params(direction='in')
    caxes.xaxis.set_ticklabels([-.05,0,.05],fontsize=12)


fig, axes, caxes = faceted(1,2,width=5,aspect=2.0,
                   bottom_pad=0.75,cbar_mode='single',cbar_pad=0.,
                    internal_pad=(0.,0.5),cbar_location='top',cbar_short_side_pad=0.)

plt.suptitle(r'$\omega_\phi$',fontsize=24,y=1.06)

for i, ax in enumerate(axes):
    c = datum2[i].plot(ax=ax,add_colorbar=False,label='a',vmax = max_vals2[i],cmap=clr)
    boundaries2[i].plot.contour(ax=ax,colors='k',linewidths=1.5)
#     contours[i].plot.contour(ax=ax,colors='k',linewidths=0.75,levels=[-.04,.04])
    ax.set_title('')
    ax.set_xlabel(xlabels2[i],fontsize=22)
    ax.set_ylabel(ylabels2[i],fontsize=22)
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    ax.text(0.05, .92, labelsa2[i],transform=ax.transAxes,fontsize=18)
#     ax.text(0.80, .92, labelsb2[i],transform=ax.transAxes,fontsize=24)
    cb = plt.colorbar(c, cax=caxes,orientation='horizontal')
    caxes.xaxis.set_tick_params(direction='in')
    cb.locator = tick_locator
    cb.update_ticks()
    ax.yaxis.set_ticks([0,0.5,1,1.5,2])
    ax.yaxis.set_ticklabels(['0','5','10','15','20'],fontsize=15)
    ax.xaxis.set_ticks([-.25,0,.25])
    ax.xaxis.set_ticklabels(['-2.5','0','2.5'],fontsize=15)
    caxes.xaxis.tick_top()
#     caxes.xaxis.set_ticks([-4,0,4],fontsize=12)
    caxes.xaxis.set_ticklabels([-4,0,4],fontsize=12)

#     caxes.xaxis.set_ticklabels(['-15','0','15'])

plt.show()
