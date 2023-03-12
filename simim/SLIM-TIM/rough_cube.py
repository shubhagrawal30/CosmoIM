import numpy as np
from scipy.interpolate import interp1d

# import simim.siminterface as sim
import simim.lightcone as lc
from simim._mplsetup import *
from simim.lineprops.lineprops import prop_delooze_cii
import simim.map as map

# Units stuff
from astropy.cosmology import WMAP9 as cosmo
c = 2.99792458e8 # m s^-1
k = 1.3806e-23 # m^2 kg s^-2 K^-1
L_sun = 3.828e26 # W
Mpc = 3.0857e22 # m

def L_to_F(L, z, cosmo=cosmo):
    L = np.array(L)*L_sun
    z = np.array(z)

    if z.ndim > 0 and z.size > 100:
        z_spline = np.linspace(np.amin(z), np.amax(z), 100)
        DL_spline = interp1d(z_spline, cosmo.luminosity_distance(z_spline).value * Mpc)
        DL = DL_spline(z)
    else:
        DL = cosmo.luminosity_distance(z).value * Mpc

    F = L/(4*np.pi*DL**2)

    return F

def F_to_I(F, dOm = 1, dnu = 1):
    I = F / (dOm * dnu)
    return I

def I_to_T(I, nu, z=None, mode='rest frequency'):
    if mode == 'observed frequency':
        nu_obs = np.array(nu)
    elif mode == 'rest frequency':
        nu_rest = np.array(nu)
        z = np.array(z)
        nu_obs = nu_rest/(1+z)

    I = np.array(I)
    T = I * c**2 / (2*k*nu_obs**2)

    return T

def L_to_T(L, nu, z, dOm, dnu, mode='rest frequency',cosmo=cosmo):
    F = L_to_F(L, z, cosmo=cosmo)
    I = F_to_I(F, dOm, dnu)
    T = I_to_T(I, nu, z, mode=mode)
    return T



lchandler = lc.handler.lightcone('TNG300-1','demo_suite',0)

if not lchandler.has_property('LCII'):
    lchandler.make_property(prop_delooze_cii)
    lchandler.write_property('LCII')

lchandler.set_property_range('redshift',0.2,1.9)


x = lchandler.return_property("ra")
y = lchandler.return_property("dec")
z = lchandler.return_property("redshift")
L = lchandler.return_property("LCII")

pos = np.array([x,y,z]).T
center = [0,0,1.05]
side = [np.pi/180,np.pi/180,1.7]
# pixel = [10/3600,10/3600,.001]
pixel = [10/3600*np.pi/180,10/3600*np.pi/180,.01]
dnu = (1500.5/(1+center[2]) - 1500.5/(1+center[2]+pixel[2])) * 1e9

T = L_to_T(L, 1900.5369*1e9, z, pixel[0]*pixel[1], dnu, mode='rest frequency', cosmo=cosmo)

print("gridding")
grid = map.gridder(pos,T,center_point=center,side_length=side,pixel_size=pixel)

print("convolving")
psf = map.psf([30/3600*np.pi/180,30/3600*np.pi/180],[grid.pixel_size[0],grid.pixel_size[1]],norm='peak')
grid.convolve(psf,pad=5)

plt.colorbar(plt.pcolor(grid.grid[:,:,-1,0],vmin=0,vmax=.00001))
# plt.show()
plt.savefig("test_rough_cube.png")
plt.close()

# grid.fourier_transform()
# grid.power_spectrum(in_place=True)
# grid.spherical_average(binmode='linear')

# print("animating")
# grid.animate(still=False,logscale=False)