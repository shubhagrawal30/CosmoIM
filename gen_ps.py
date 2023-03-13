import numpy as np
import os
import simim.siminterface as sim
from matplotlib import pyplot as plt
from simim.lineprops.lineprops import prop_li_co,prop_behroozi_sfr
from simim.map import gridder
from itertools import cycle
import time
import multiprocessing as mp
from pathlib import Path

plt.rcParams.update({'font.size': 8})

plots_dir = "./power_spectra/CO/20230313_no_std/plots/"
out_dir = "./power_spectra/CO/20230313_no_std/"
Path(plots_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)


num_snaps = 34
pixel_size = 0.25
bins = np.logspace(-4,4,81)
num_threads = 32

colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
markers = cycle(['o', 's', 'd', '^', 'v', 'p', '*', 'x'])

wl_co = 2.60076e-3      # m
Lsun = 3.828e26         # W
Mpc = 3.0857e22         # m
c = 2.99792458e8        # m s^-1
kB = 1.3806e-23         # m^2 kg s^-2 K^-1

def one_simulation(sim_name_camels, overwrite=False):
    if (not overwrite) and os.path.exists(os.path.join(out_dir, sim_name_camels + ".npz")):
        print(sim_name_camels, "already exists")
        return
    print(sim_name_camels)
    sim_name = f"camels/{sim_name_camels}"
    
    redshifts = {}
    curves = {}
    ks = {}
    
    simhandler = sim.simhandler.simhandler(sim_name)
    
    side_length = ((simhandler.box_edge_no_h) // pixel_size) * pixel_size
    center_point = [side_length/2,side_length/2,side_length/2]
    
    fig,ax = plt.subplots(1,2,figsize=(14,6),sharex=True)
    fig.subplots_adjust(wspace=.4)
    ax[0].set(xlabel='k [Mpc$^{-1}$]',ylabel='P(k) [$\mu$K$^2$ Mpc$^{3}$]',xscale='log',yscale='log')
    ax[0].grid()
    ax[1].set(xlabel='k [Mpc$^{-1}$]',ylabel='k$^3$P(k)/2$\pi$ [$\mu$K$^2$]',xscale='log',yscale='log')
    ax[1].grid()
    
    for snap_index in range(num_snaps):
        snap = simhandler.get_snap(snap_index)
        snap_z = snap.redshift

        snap.set_property_range()
        snap.make_property(prop_behroozi_sfr,rename='sfr_behroozi',other_kws={'scatter':False})
        snap.make_property(prop_li_co,rename='LCO_ns',kw_remap={'sfr':'sfr_behroozi'},other_kws={'scatter_lco':False})

        # Compute some cosmological factors
        xfactor = snap.cosmo.comoving_distance(snap_z).value                         # Mpc / rad
        yfactor = wl_co * (1+snap_z)**2 / (1000*snap.cosmo.H(snap_z).value)  # derivative of distance with respect to frequency in Mpc / Hz
        dl = (1+snap_z)*xfactor                                                      # luminosity distance to z in Mpc

        # Convert flux to luminosity for each point
        L_to_f = Lsun / (4*np.pi*(dl*Mpc)**2)                                                # Convert luminosity (Lsum) to flux (W/m^2)
        f_to_i = xfactor**2*yfactor/pixel_size**3                                            # Convert flux (W/m^2) to intensity (W/m^2/Str/Hz)
        i_to_T = ((1+snap_z)*wl_co)**2 / (2*kB) * 1e6                                # Convert intensity (W/m^2/Str/Hz) to brightness temp (uK)
        temps = snap.return_property('LCO_ns',in_h_units=False) * L_to_f * f_to_i * i_to_T

        # Get object positions
        x = snap.return_property('pos_x',in_h_units=False)
        y = snap.return_property('pos_y',in_h_units=False)
        z = snap.return_property('pos_z',in_h_units=False)
        positions = np.array([x,y,z]).T

        # Make the grid
        grid = gridder(positions,temps,center_point=center_point,side_length=side_length,pixel_size=pixel_size,axunits='Mpc',gridunits='uK')
        ps = grid.power_spectrum(in_place=False,normalize=True)

        # Spherical average
        bin_edges, ps1d = ps.spherical_average(ax=[0,1,2],bins=bins)
        ps1d = ps1d[:,0] / side_length**3                           # Normalize by volume
        k = 2*np.pi * (bin_edges[0][:-1] + bin_edges[0][1:]) / 2
        
        color, marker = next(colors), next(markers)
        # Plot the results
        ax[0].plot(k,ps1d,lw=0.5,label="{:.2f}".format(snap_z), color=color, marker=marker, ls="--")
        ax[1].plot(k,k**3/2/np.pi**2*ps1d,lw=0.5,label="{:.2f}".format(snap_z), color=color, marker=marker, ls="--")
        
        curves[snap_index] = ps1d
        ks[snap_index] = k
        redshifts[snap_index] = snap_z
    
    ax[0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
    ax[1].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6)
    plt.savefig(os.path.join(plots_dir, f"{sim_name_camels}.pdf"), dpi=300, bbox_inches="tight")
    plt.close()
    
    np.savez(os.path.join(out_dir, sim_name_camels), curves=curves, redshifts=redshifts, ks=ks)
    
if __name__ == "__main__":
    start = time.time()
    # one_simulation("CV_0")
    
    args = os.listdir("/global/cfs/cdirs/des/shubh/timsim/simim_resources/simulations/camels/")
    my_pool = mp.Pool(processes=num_threads)
    _ = my_pool.map(one_simulation, args)
    
    print(time.time() - start)
    
    