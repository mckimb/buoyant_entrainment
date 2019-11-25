# momentum.py computes the impulse of the thermal in 3 different ways.
# 1) look at the impulse imparted by buoyancy,
# 2) use the simplifed formula I = pi*R**2 * gamma,
# 3) use the full formula: I = \int r \cross \omega dV


import xarray as xr
import numpy as np
import matplotlib
from faceted import faceted
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# matplotlib.rc('text', usetex=True)
from lighten_color import lighten_color

lam_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/lam/rho_u_v_w/slice*.nc', concat_dim='t')
lam_omega = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/vort_phi/azi_lam_vort.nc')
lam_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/laminar_mask.nc',engine='scipy')
lam_circ = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/lam/mask/circ.nc')
lam_azi_omega = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/lam/azi_vort_phi/lam*.nc',concat_dim='t').omega_phi

turb_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/rho_u_v_w/slice*.nc', concat_dim='t')
turb_omega = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/vort_phi/turb*.nc',concat_dim='t').omega_phi
turb_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/mask_correct.nc',engine='scipy')
turb_circ = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/circ.nc')
turb_azi_omega = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/azi_vort_phi/turb*.nc',concat_dim='t').omega_phi

def get_ring_radius(data, output):
    """
    Get's laminar and turbulent ring radius from azimuthally averaged datasets
    ie. lam_azi_omega and turb_azi_omega
    """
    tau = np.linspace(0,20,200)
    r = np.zeros_like(tau)

    for i in range(200):
        omega = data.sel(t=tau[i], method='nearest')
        omega1 = omega.where(omega.r>.04,drop=True)
        ring = omega1.where(omega1==omega1.max(),drop=True)
        r[i] = ring.r.values[0]

    radius = xr.Dataset({'r':(['t'],r)},coords={'t':(['t'],data.t)})
    radius.to_netcdf(output)
    return radius

# lam_ring_r = get_ring_radius(lam_azi_omega,'/work/bnm/buoyant_entrainment/data/lam/mask/ring_radius.nc').r
# turb_ring_r =  get_ring_radius(turb_azi_omega,'/work/bnm/buoyant_entrainment/data/turb/mask/ring_radius.nc').r

lam_impulse = np.pi * lam_circ.sel(t=4,method='nearest') * lam_ring_r ** 2
turb_impulse = np.pi * turb_circ.sel(t=4,method='nearest') * turb_ring_r ** 2

'''This is how the below data was created, but it's very memory intensive'''

# lam_dx = lam_omega.x.diff('x').values[0]
# lam_dy = lam_omega.y.diff('y').values[0]
# lam_dz = lam_omega.z.diff('z').values[0]
# lam_r_coor = xr.ufuncs.sqrt(lam_omega.x ** 2 + lam_omega.y ** 2)
# lam_ellipse = lam_omega.where(lam_mask)
# lam_int = 0.5 * (lam_ellipse * lam_r_coor).sum(['x', 'y', 'z']) * lam_dx * lam_dy * lam_dz

# turb_dx = turb_omega.x.diff('x').values[0]
# turb_dy = turb_omega.y.diff('y').values[0]
# turb_dz = turb_omega.z.diff('z').values[0]
# turb_r_coor = xr.ufuncs.sqrt(turb_omega.x ** 2 + turb_omega.y ** 2)
# turb_ellipse = turb_omega.where(turb_mask)
# turb_int = 0.5 * (turb_ellipse * turb_r_coor).sum(['x', 'y', 'z']) * turb_dx * turb_dy * turb_dz

'''This is how the below data was created'''

# lam_B_impulse = lam_ds.rho.where(lam_mask).sum(('x','y','z')) * lam_dx * lam_dy * lam_dz * -1 * lam_ds.t
# turb_B_impulse = turb_ds.rho.where(turb_mask).sum(('x','y','z')) * turb_dx * turb_dy * turb_dz * -1 * turb_ds.t

'''RESCALE FACTOR IS sqrt(10)'''

# momentum vs time
t0 = 4
matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally

tick_locator = ticker.MaxNLocator(nbins=2)
fig, axes = faceted(1, 1, width=5, aspect=1/1.618, bottom_pad=0.75, internal_pad=(0.,0.),sharex=False,sharey=False)

axes[0].plot(lam_int.t/t0,lam_int*np.sqrt(10),color=lighten_color('darkgoldenrod'),linewidth=2.5)
axes[0].plot(lam_B_impulse.t/t0,lam_B_impulse*np.sqrt(10),color=lighten_color('peru'),linestyle='dashed',linewidth=2.5)
axes[0].plot(lam_impulse.t/t0,lam_impulse*np.sqrt(10),color='darkgoldenrod',linewidth=2.5)


axes[0].plot(turb_int.t/t0,turb_int*np.sqrt(10),color=lighten_color('rebeccapurple'),linewidth=2.5)
axes[0].plot(turb_B_impulse.t/t0,turb_B_impulse*np.sqrt(10),color='orchid',linestyle='dashed',linewidth=2.5)
axes[0].plot(turb_impulse.t/t0,turb_impulse*np.sqrt(10),color='rebeccapurple',linewidth=2.5)

# axes[0].plot(lam_circ.t/t0,lam_circ,color=lighten_color('darkgoldenrod'),marker='.',linewidth=1.5,label=r'$\mathrm{Re}=630$')
# axes[0].plot(turb_circ.t/t0,turb_circ,color=lighten_color('rebeccapurple'),marker='.',linewidth=1.5,label=r'$\mathrm{Re} = 6300$')

axes[0].set_xlim([0,5])
axes[0].set_ylim([0,0.05])

axes[0].set_yticks([0,0.01,0.02,0.03,0.04,.05])
axes[0].set_yticklabels(['0','10','20','30','40','50'],fontsize=15)

axes[0].set_xticks([0,1,2,3,4,5])
axes[0].set_xticklabels(['0','1','2','3','4','5'],fontsize=15)

axes[0].xaxis.set_tick_params(direction='in')
axes[0].yaxis.set_tick_params(direction='in')

axes[0].set_xlabel(r'$\tau$',fontsize=18)
axes[0].set_ylabel(r'$I_z$',fontsize=22)

lines = [Line2D([0],[0],color=lighten_color('darkgoldenrod'),linewidth=2.5),
        Line2D([0],[0],color='darkgoldenrod',linewidth=2.5),
        Line2D([0],[0],color=lighten_color('peru'),linestyle='dashed',linewidth=2.5),
        Line2D([0],[0],color=lighten_color('rebeccapurple'),linewidth=2.5),
        Line2D([0],[0],color='rebeccapurple',linewidth=2.5),
        Line2D([0],[0],color='orchid',linestyle='dashed',linewidth=2.5)]

# labels = [r'$\mathrm{Re} = 630 \ \  \mathrm{Eq }\ 3$',r'$\mathrm{Re} = 630 \ \  \mathrm{Eq }\ 2$',r'$\mathrm{Re} = 630 \ \ B\tau$',r'$\mathrm{Re} = 6300 \ \ \mathrm{Eq }\ 3$',r'$\mathrm{Re} = 6300  \ \ \mathrm{Eq }\ 2$',r'$\mathrm{Re} = 6300 \  \ B\tau$']
labels = [r'$\mathrm{Eq }\ 2$',r'$\mathrm{Eq }\ 3$',r'$F \ \tau$',r'$\mathrm{Eq }\ 2$',r'$\mathrm{Eq }\ 3$',r'$F \ \tau$']

axes[0].text(0.043, 0.9, r'$\mathrm{Re}= 630$', transform=axes[0].transAxes, fontsize = 14)
axes[0].text(0.31, 0.9, r'$\mathrm{Re}= 6300$', transform=axes[0].transAxes, fontsize = 14)

plt.legend(lines,labels,loc=(0.03,0.58), ncol=2, fontsize=12.5,frameon=False)

plt.show()
