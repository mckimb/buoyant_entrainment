# scalings.py shows how the prediction for the thermals height, density, and radius align with the simulation results


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
lam_z2 = np.loadtxt('/work/bnm/buoyant_entrainment/data/lam/mask/cloud_top_1e3_sim_5.txt')

turb_ds = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/rho_u_v_w/slice*.nc', concat_dim='t')
turb_omega = xr.open_mfdataset('/work/bnm/buoyant_entrainment/data/turb/vort_phi/turb*.nc',concat_dim='t').omega_phi
turb_mask = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/mask_correct.nc',engine='scipy')
turb_track = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/thermal_boundary.nc')
turb_w_avg = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/w_avg.nc').w
turb_rho_avg = xr.open_dataarray('/work/bnm/buoyant_entrainment/data/turb/mask/rho_avg.nc')
turb_r = turb_track.r.max('z')
turb_z = xr.open_dataset('/work/bnm/buoyant_entrainment/data/turb/mask/z_top.nc').z
turb_z2 = np.loadtxt('/work/bnm/buoyant_entrainment/data/turb/mask/cloud_top_1e4_sim_5.txt')

# initial values
t0 = 4

lrho0 = lam_rho_avg.sel(t=t0,method='nearest').values
lw0 = lam_w_avg.sel(t=t0,method='nearest').values
lr0 = lam_r.sel(t=t0,method='nearest').values
# lz0 = lam_z.sel(t=t0,method='nearest').values
lz0 = lam_z2[1][lam_z2[0] >= t0][0]

trho0 = turb_rho_avg.sel(t=t0,method='nearest').values
tw0 = turb_w_avg.sel(t=t0,method='nearest').values
tr0 = turb_r.sel(t=t0,method='nearest').values
# tz0 = turb_z.sel(t=t0,method='nearest').values
tz0 = turb_z2[1][lam_z2[0] >= t0][0]

# abbreviate time for readability
lam_t = lam_ds.t.values
# lam_zt = lam_z.t.values
lam_zt = lam_z2[0]

turb_t = turb_ds.t.values
# turb_zt = turb_z.t.values
turb_zt = turb_z2[0]

matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally
matplotlib.rcParams['axes.titlepad'] = 31
tick_locator = ticker.MaxNLocator(nbins=2)

fig, axes = faceted(3, 2, width=5, aspect=1.0, bottom_pad=0.75, internal_pad=(0.,0.),sharex=False,sharey=False)

xlabels = ['','','','',r'$\tau$',r'$\tau$']
ylabels = [r'$\frac{a}{a_o}$','',r'$\frac{z}{z_o}$','',r'$\frac{\langle \rho^{\prime} \rangle}{ \langle \rho_o^{\prime} \rangle}$','']

# datum = [lam_r/lr0, turb_r/tr0, lam_z/lz0, turb_z/tz0, lam_rho_avg/lrho0, turb_rho_avg/trho0]
datum = [lam_r/lr0, turb_r/tr0, lam_z2[1]/lz0, turb_z2[1]/tz0, lam_rho_avg/lrho0, turb_rho_avg/trho0]
times = [lam_t/t0,  turb_t/t0,  lam_zt/t0, turb_zt/t0, lam_t/t0,          turb_t/t0]


theory = [(lam_t/t0)**(1/2), (turb_t/t0)**(1/2), 1+2*lw0*t0/lz0*((lam_zt/t0)**(1/2)-1),
          1+2*tw0*t0/tz0*((turb_zt/t0)**(1/2)-1), (lam_t/t0)**(-3/2), (turb_t/t0)**(-3/2)]

labels = [r'$(a)$', r'$(b)$', r'$(c)$',r'$(d)$', r'$(e)$', r'$(f)$']

# xlims = [[1,3.45],[1,4.9],[1,3.45],[1,4.9],[1,3.45],[1,4.9]]
xlims = [[1,4.9],[1,4.9],[1,4.9],[1,4.9],[1,4.9],[1,4.9]]


# xticka = [1,1.5,2,2.5,3]
xticka = [1,2,3,4]
xtickb = [1,2,3,4,4.9]
xticks = [xticka, xtickb, xticka, xtickb, xticka, xtickb]

# xticklabela1 = ['1','','2','','3']
# xticklabela2 = [' ','',' ','',' ']
xticklabela1 = ['1','2','3','4']
xticklabela2 = [' ',' ','',' ']
xticklabelb1 = ['1','2','3','4','5']
xticklabelb2 = [' ',' ',' ',' ','']
xticklabels = [xticklabela2, xticklabelb2,xticklabela2, xticklabelb2, xticklabela1, xticklabelb1]


ylims = [[1,2.3],[1,2.3],[1,2.6],[1,2.6],[0,1],[0,1]]

yticka = [1.0,1.4,1.8,2.2]
ytickb = [0,0.5,1.0]
yticks = [yticka,yticka,yticka,yticka,ytickb,ytickb]

yticklabela1 = ['1','1.4','1.8','2.2']
yticklabela2 = ['', '','   ','','   ']
yticklabelb1 = ['0','0.5','1']
yticklabelb2 = [' ','   ','   ',' ']
yticklabels = [yticklabela1, yticklabela2, yticklabela1, yticklabela2, yticklabelb1, yticklabelb2]

titles = [r'$\mathrm{Re} = 630$',r'$\mathrm{Re} = 6300$', '','','','']
label1 = [r'$simulation$','','','','','']
label2 = [r'$theory$','','','','','',]

colors = ['darkgoldenrod','rebeccapurple','darkgoldenrod','rebeccapurple','darkgoldenrod','rebeccapurple']

for i, ax in enumerate(axes):
#     ax.plot(times[i],datum[i],color=lighten_color(colors[i]),marker='.',markersize=4,linewidth=1.5,label='simulation')
    ax.plot(times[i],datum[i],color=lighten_color(colors[i]),linewidth=2.5,label='simulation')
    ax.plot(times[i],theory[i],color=colors[i],linewidth=2.5,linestyle='dashed',label='theory')
    ax.set_xlim(xlims[i])
    ax.set_ylim(ylims[i])
    ax.set_xlabel(xlabels[i],fontsize=18)
    ax.set_ylabel(ylabels[i],fontsize=22)
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    ax.set_xticks(xticks[i])
    ax.set_xticklabels(xticklabels[i],fontsize=13)
    ax.set_yticks(yticks[i])
    ax.set_yticklabels(yticklabels[i],fontsize=13)
    ax.text(0.05, .9, labels[i],transform=ax.transAxes,fontsize=15)
    ax.set_title(titles[i],fontsize=16)
    ax.update_datalim
#     ax.legend(loc='upper center')

lines2= [Line2D([0],[0],color='darkgoldenrod',linewidth=2.5,linestyle='dashed'),
        Line2D([0],[0],color=lighten_color('darkgoldenrod'),linewidth=2.5)]
labels2 = [r'$\mathrm{theory}$',r'$\mathrm{sim}$']

lines1 = [Line2D([0],[0],color='rebeccapurple',linewidth=2.5,linestyle='dashed'),
        Line2D([0],[0],color=lighten_color('rebeccapurple'),linewidth=2.5)]
labels1 = [r'$\mathrm{theory}$',r'$\mathrm{sim}$']

leg = plt.legend(lines1, labels1,ncol=2,loc='upper left',bbox_to_anchor=(0, 3.20),fontsize=12.5,frameon=False)
ax.add_artist(leg)
plt.legend(lines2, labels2,ncol=2,loc='upper right',bbox_to_anchor=(0, 3.20),fontsize=12.5,frameon=False)
plt.show()
