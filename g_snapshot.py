# g_snapshot.py shows the vertical evolution of thermals when gravity is kept on, or turned off


import xarray as xr
import numpy as np
import matplotlib
import colorcet
from faceted import faceted
from matplotlib import ticker
matplotlib.rcParams['mathtext.fontset'] = 'cm'
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.dpi']= 600



matplotlib.rcParams['axes.linewidth'] = 2 #set the value globally
tick_locator = ticker.MaxNLocator(nbins=2)

xlabels1, xlabels2 = [r'$x$',r'$x$'], [r'$x$',r'$x$']
ylabels1, ylabels2 = [r'$z$',''], [r'$z$','']

datum1, datum2 = [g+g0+g1,g+nog0+nog1], [g1, nog1]
labelsa1 = [r'$g \ne 0$',r'$g=0 \ when \ \tau > 1.5$']
labelsb1 = ['', r'$\rho^{\prime}$']
labelsa2 = [r'$g \ne 0$',r'$g=0$']
labelsb2 = ['', r'$\rho^{\prime}$']

times =  ['',r'$\tau = 0$']
times0 = ['',r'$\tau = 1.5$']
times1 = ['',r'$\tau = 4$']

boundaries1, boundaries2 = [g_bound0+g_bound1, g_bound0+nog_bound1], [g_bound1, nog_bound1]

# contours =  [lam_Domega/10, turb_Domega/10]


max_vals1 = [.05,.05]
max_vals2 = [.05, .05]

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
#     ax.text(0.05, .92, labelsa1[i],transform=ax.transAxes,fontsize=18)
#     ax.text(0.05, .06, times[i],transform=ax.transAxes,fontsize=15)
#     ax.text(0.05, .42, times0[i],transform=ax.transAxes,fontsize=15)
#     ax.text(0.05, .72, times1[i],transform=ax.transAxes,fontsize=15)
#     ax.text(0.80, .92, labelsb1[i],transform=ax.transAxes,fontsize=24)
    cb = plt.colorbar(c, cax=caxes,orientation='horizontal')
    cb.locator = tick_locator
    ax.yaxis.set_ticks([0,0.5,1,1.5,2])
    ax.yaxis.set_ticklabels(['0','5','10','15','20'],fontsize=15) #these have been rescaled
    ax.xaxis.set_ticks([-.25,0,.25])
#     ax.xaxis.set_ticklabels(['','',''])
    ax.xaxis.set_ticklabels(['-2.5','0','2.5'],fontsize=15)
    ax.xaxis.set_tick_params(direction='in')
    ax.yaxis.set_tick_params(direction='in')
    cb.update_ticks()
    caxes.xaxis.tick_top()
    caxes.xaxis.set_tick_params(direction='in')
    caxes.xaxis.set_ticklabels([-.05,0,.05],fontsize=12)


fig, axes, caxes = faceted(1,2,width=5,aspect=2.0,
                   bottom_pad=0.75,cbar_mode='single',cbar_pad=0.,
                    internal_pad=(0.,0.5),cbar_location='top',cbar_short_side_pad=0.)

plt.suptitle(r'$\rho^\prime$',fontsize=24,y=1.06)

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
    caxes.xaxis.set_ticklabels([-.05,0,.05],fontsize=12)

#     caxes.xaxis.set_ticklabels(['-15','0','15'])

plt.show()
