import numpy as np
import matplotlib.pyplot as plt

# Some parameters for TIM
# We'll just analyze the center 1/3 of the bandpass
fmin = 2.998e8 / 360e-6 # Hz
fmax = 2.998e8 / 300e-6
fmid = (fmin+fmax)/2
fcii = 1900.5369e9 # Hz

# Cosmology stuff
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
h = 0.6774   # Value of little h appropriate for TNG simulations
cosmo = FlatLambdaCDM(H0=100*h*(u.km/u.Mpc/u.s), Om0=0.309, Tcmb0=2.725*u.K, Neff=3.04, Ob0=0.0486)

# Load each grid and compute the power spectrum
from simim.map import load_grid

cube_path = '/data/rpkeenan/simim_resources/cubes/TIM/large_cube_0_'
paths = ['full_res.npz','CII_full_res.npz','NII205_full_res.npz','NII122_full_res.npz','OIII_full_res.npz',
         'tim_res.npz','CII_tim_res.npz','NII205_tim_res.npz','NII122_tim_res.npz','OIII_tim_res.npz']
names = ['Total Power','CII','NII 205um','NII 122um','OIII',
         'Total Power w/ Beam','CII w/ Beam','NII 205um w/ Beam','NII 122um w/ Beam','OIII w/ Beam',]

power_spectra = {}
for path, name in zip(paths,names):
    print("working on {}".format(name))

    # Load the grid
    grid = load_grid(cube_path+path)

    # Crop grid to central 1/3 of TIM frequency coverage
    grid.crop(ax=2,min=2.998e8/360e-6,max=2.998e8/300e-6)

    # Compute the power spectrum of the grid
    grid.power_spectrum(normalize=False,in_place=True)

    # Convert axes from uveta to k space - using transform apropriate for CII
    z_cii_mid = fcii / fmid - 1
    xfactor = cosmo.comoving_distance(z_cii_mid).value
    yfactor = 2.998e8/fcii * (1+z_cii_mid)**2 / (1000*cosmo.H(z_cii_mid).value)

    grid.fourier_axes[0] = 2*np.pi*grid.fourier_axes[0] / xfactor / h
    grid.fourier_axes[1] = 2*np.pi*grid.fourier_axes[1] / xfactor / h
    grid.fourier_axes[2] = 2*np.pi*grid.fourier_axes[2] / yfactor / h
    grid.fourier_axes_centers[0] = 2*np.pi*grid.fourier_axes_centers[0] / xfactor / h
    grid.fourier_axes_centers[1] = 2*np.pi*grid.fourier_axes_centers[1] / xfactor / h
    grid.fourier_axes_centers[2] = 2*np.pi*grid.fourier_axes_centers[2] / yfactor / h

    # Normalize power spectrum - units will be (Jy/Str)^2 Mpc^3 h^-3
    v = np.prod(grid.side_length) * xfactor**2 * yfactor
    grid.grid = grid.grid * (np.prod(grid.pixel_size) * xfactor**2 * yfactor)**2
    grid.grid = grid.grid / v * h**3

    # Do a spherical average
    k_edges,ps1d,n1d = grid.spherical_average(ax=[0,1,2],bins=np.logspace(-3,3,31),return_n=True)

    power_spectra[name] = ps1d.flatten()

k_centers = np.logspace(-2.9,2.9,30)
n1d = n1d.flatten()

# Plot the power spectrum
fig = plt.figure(figsize=(6,8))
gs = fig.add_gridspec(2, 1,  height_ratios=(5,2),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0, hspace=0)
ax = fig.add_subplot(gs[0])
ax_n = fig.add_subplot(gs[1])
ax_n.sharex(ax)
plt.setp(ax.get_xticklabels(),visible=False)

xlims = (2*np.pi/(grid.side_length[2]*yfactor*h)/1.2,2*np.pi/(grid.pixel_size[0]*xfactor*h)*1.2)
ylims = (np.nanmin(power_spectra['NII 205um'])/1.5,np.nanmax(power_spectra['Total Power'])*1.5)
ax.set(ylabel='P(k) [Jy$^2$ str$^{-2}$ h$^{-3}$ Mpc$^3$]',xscale='log',yscale='log',xlim=xlims,ylim=ylims)
ax_n.set(xlabel='k [h Mpc$^{-1}$]',ylabel='n cells',yscale='log',yticks=[1e2,1e4,1e6])

ax.plot(k_centers,power_spectra['Total Power w/ Beam'],label='w/ Beam Effect',ls='--',color='k',lw=3)
for name,color in zip(['CII w/ Beam','NII 205um w/ Beam','NII 122um w/ Beam','OIII w/ Beam'],['C0','C4','C1','C3']):
    ax.plot(k_centers,power_spectra[name],color=color,ls='--')
ax.plot(k_centers,power_spectra['Total Power'],label='Total Power',color='k',lw=3)
for name,color in zip(['CII','NII 205um','NII 122um','OIII'],['C0','C4','C1','C3']):
    ax.plot(k_centers,power_spectra[name],label=name,color=color)
ax.legend(fontsize='small')

ax_n.plot(k_centers,n1d,color='k',ls='none',marker='x')

for a in [ax,ax_n]:
    a.axvline(2*np.pi/(grid.pixel_size[0]*xfactor*h),color='.5',ls='--')
    a.axvline(2*np.pi/(1.2*330e-6/2*xfactor*h),color='k',ls='--')
    a.axvline(2*np.pi/(grid.side_length[0]*xfactor*h),color='k',ls='--')

    a.axvline(2*np.pi/(grid.pixel_size[2]*yfactor*h),color=(1,.5,.5),ls='--')
    a.axvline(2*np.pi/(2.998e8/330e-6/300*yfactor*h),color='r',ls='--')
    a.axvline(2*np.pi/(grid.side_length[2]*yfactor*h),color='r',ls='--')

plt.show()