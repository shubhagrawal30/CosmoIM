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
from copy import copy

plt.rcParams.update({'font.size': 8})

plots_dir = "./grids/CO/20230327_1-2/plots/"
out_dir = "./grids/CO/20230327_1-2/"
Path(plots_dir).mkdir(parents=True, exist_ok=True)
Path(out_dir).mkdir(parents=True, exist_ok=True)

num_snaps = 34
pixel_size = 0.5
bins = np.logspace(-4,4,81)
num_threads = 32

wl_co = 2.60076e-3      # m
Lsun = 3.828e26         # W
Mpc = 3.0857e22         # m
c = 2.99792458e8        # m s^-1
kB = 1.3806e-23         # m^2 kg s^-2 K^-1

def one_simulation(sim_name_camels, overwrite=False):
    # try:
    if 1:
        if (not overwrite) and os.path.exists(os.path.join(out_dir, sim_name_camels + ".npz")):
            # print(sim_name_camels, "already exists")
            return
        print(sim_name_camels)
        sim_name = f"camels/{sim_name_camels}"

        redshifts = {}
        curves = {}

        simhandler = sim.simhandler.simhandler(sim_name)

        side_length = ((simhandler.box_edge_no_h) // pixel_size) * pixel_size
        center_point = [side_length/2,side_length/2,side_length/2]

        fig, axs = plt.subplots(3, 4, figsize=(12,8))
        fig.suptitle("mean along third spatial axis", fontsize=12)
        axs = axs.ravel()
        plot_index = 0
        plot_skip = 3
        fig.subplots_adjust(wspace=.4)

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

            curves[snap_index] = grid.grid[:,:,:,0]
            redshifts[snap_index] = snap_z
            
            if snap_index % plot_skip == 0:
                ax = axs[plot_index]
                ax.set_title(f"{snap_index}, {snap_z:.4f}")
                im = ax.imshow(np.nanmean(grid.grid[:,:,:,0], axis=0), vmax=2, cmap="cividis", origin="lower")
                ax.grid()
                cbar = fig.colorbar(im, ax=ax)
                plot_index += 1


        plt.savefig(os.path.join(plots_dir, f"{sim_name_camels}.pdf"), dpi=300, bbox_inches="tight")
        plt.close()

        np.savez(os.path.join(out_dir, sim_name_camels), curves=curves, redshifts=redshifts)
    # except:
    #     print("failed", sim_name_camels)
    
if __name__ == "__main__":
    start = time.time()
    one_simulation("LH_603", overwrite=True)
            
#     args = os.listdir("/global/cfs/cdirs/des/shubh/timsim/simim_resources/simulations/camels/")
    
#     overwrite = False
#     for sim_name_camels in copy(args):
#         if (not overwrite) and os.path.exists(os.path.join(out_dir, sim_name_camels + ".npz")):
#             args.remove(sim_name_camels)
#     print(f"running {len(args)} simulations")
#     my_pool = mp.Pool(processes=num_threads)
#     _ = my_pool.map(one_simulation, args)
    
    print(time.time() - start)
    
    