import os
import warnings

from astropy.cosmology import FlatLambdaCDM
from astropy.cosmology import z_at_value
import astropy.units as u
import h5py
import numpy as np

from simim import _paths
from simim.siminterface._sims import _checksim


class snapshot():
    def __init__(self,index,redshift,metadata):
        self.index = index
        self.redshift = redshift

        cosmo = FlatLambdaCDM(H0=100*metadata['cosmo_h']*u.km/u.s/u.Mpc,
                              Om0=metadata['cosmo_omega_matter'],
                              Ob0=metadata['cosmo_omega_baryon'],
                              Tcmb0=2.7255*u.K)

        self.cosmology = cosmo

    def dif_snap(self,other):
        time_snap = self.cosmology.age(self.redshift).value
        time_other = self.cosmology.age(other.redshift).value
        time_middle = (time_snap+time_other)/2

        redshift_middle = z_at_value(self.cosmology.age,time_middle*u.Gyr)

        return redshift_middle, time_middle

    def dif_lowerz_snap(self,other):

        if other == 0:
            self.redshift_min = 0
            self.time_end = self.cosmology.age(0).value
            self.distance_min = 0
            self.transverse_distance_min = 0

        else:
            if self.redshift < other.redshift:
                print("Not provided a lower redshift snapshot")
                raise Error

            redshift_middle, time_middle = self.dif_snap(other)
            self.redshift_min = redshift_middle
            self.time_end = time_middle
            self.distance_min = self.cosmology.comoving_distance(self.redshift_min).value
            self.transverse_distance_min = self.cosmology.comoving_transverse_distance(self.redshift_min).value

    def dif_higherz_snap(self,other):

        if other == 'max':
            self.redshift_max = self.redshift
            self.time_start = self.cosmology.age(self.redshift).value
            self.distance_max = self.cosmology.comoving_distance(self.redshift).value
            self.transverse_distance_max = self.cosmology.comoving_transverse_distance(self.redshift).value

        else:
            if self.redshift > other.redshift:
                print("Not provided a higher redshift snapshot")
                raise Error

            redshift_middle, time_middle = self.dif_snap(other)
            self.redshift_max = redshift_middle
            self.time_start = time_middle
            self.distance_max = self.cosmology.comoving_distance(self.redshift_max).value
            self.transverse_distance_max = self.cosmology.comoving_transverse_distance(self.redshift_max).value





# Notes about HDF5 file structure:
# file
# -- Atribute: associated metadata for sim
# -- Atribute: associated cosmology for sim
# -- Atribute: keys available
# -- Group: Snap {}
#    -- Atribute: associated metadata for snapshot
#    -- Atribute: redshifts and distances where the sim starts/ends
#    -- Atribute: structured array containing location of mass cuts
#    -- dataset:
#         note: mass units - Msun/h, lenghth units Mpc/h, time units Gyr/h

class sim_catalogs():
    def __init__(self,
                 sim, path='auto',
                 snaps='all',
                 updatepath=True,
                 ):

        # Check that we can handle the specified sim
        _checksim(sim)
        self.sim = sim

        # Set up a place to keep the data
        paths = _paths._paths()
        if path == 'auto':
            if self.sim in paths.paths.keys():
                self.path = paths.paths[self.sim]
            else:
                paths._newsimpath(self.sim)
                self.path = paths.paths[self.sim]
        else:
            if not os.path.exists(root):
                raise NameError("Specified path not found")
            self.path = os.path.abspath(path)
            if updatepath:
                paths._newsimpath(sim,new_path=path,checkoverwrite=True)

        self.meta_path = self.path + '/meta.npy'
        self.snap_meta_path = self.path + '/meta_snaps.npy'

        # Figure out the snapshots needed
        if self.sim[:3] == 'Ill':
            self.allsnaps = np.arange(136)

            # For Illustris-1 some snapshots were lost
            if self.sim == 'Illustris-1':
                self.allsnaps = np.setdiff1d(self.allsnaps,[53,55])

        elif self.sim[:3] == 'TNG':
            self.allsnaps =  np.arange(100)
            
        elif "camels" in self.sim:
            self.allsnaps = np.arange(34)

################ WATCH OUT FOR THIS!!! :
        elif self.sim == 'UniverseMachine-SMDPL':
            self.allsnaps = np.arange(117)

        if snaps == 'all':
            self.snaps = self.allsnaps
        else:
            self.snaps = snaps

        # Check that snapshots requested exist
        badsnaps = np.setdiff1d(self.snaps,self.allsnaps)
        if len(badsnaps) > 0:
            warnings.warn("Some requested snapshots do not exist: {}. Skipping them.".format(badsnaps))
        self.snaps = np.intersect1d(self.snaps,self.allsnaps)

        # Check what data might already exist
        if not os.path.exists(self.path+'/raw'):
            warnings.warn("No data for this simulation has been downloaded - run .download")
        if not os.path.exists(self.meta_path):
            warnings.warn("No meta data for this simulation is available - run .download_meta")
            self.metadata = None
            self.snap_meta = None
            self.h = None
            self.box_edge = None
        else:
            self.metadata = np.load(self.meta_path,allow_pickle='TRUE').item()
            self.snap_meta = np.load(self.snap_meta_path)
            self.h = self.metadata['cosmo_h']
            self.box_edge = self.metadata['box_edge']

        # Initialize the containers for various types of info
            # keys that require a unit conversion:
        self.mass_e8_keys = []
        self.mass_add_h_keys = []
        self.pos_kpc_keys = []
        self.inv_time_keys = []

            # Mapping key in original data type to new data structure
        self.basic_fields = {}
        self.dm_fields = {}
        self.matter_fields = {}

        # File name structure of raw data
        self.raw_fname = ''

        # Initialize a function to get halos - this must be modified when
        # creating the actual class, it's here as a template
        def loader(path, snapshot, fields):
            subhalos = {'anykey':[]}
            n_halos = len(subhalos['anykey'])
            return subhalos, n_halos
        self.loader = loader

    def clean_raw(self):

        # Confirm you really want to delete the stuff you spend two days downloading
        print("This will delete raw files for {}.".format(self.sim))
        answer = ("Are you sure you wish to proceed? y/n: ")
        while answer != 'y':
            if answer == 'n':
                print("Aborting cleanup")
                return
            print("Answer not recognized.\n")

            answer = ("Are you sure you wish to proceed? y/n: ")

        # Remove stuff
        file_path = self.path+'/raw'
        os.system("rm -r {}".format(file_path))

    def format(self,remake=False,basic=False):

        if not remake:
            if os.path.exists(self.path+'/data.hdf5'):
                warnings.warn("Formatted data appears to exist already")
                return

        if basic:
            self.other_fields = {}
        else:
            self.other_fields = {**self.dm_fields,**self.matter_fields}
        self.all_fields = {**self.basic_fields,**self.other_fields}

        # Create the file and give it some basic information
        with h5py.File(self.path+'/data.hdf5', "w") as file:
            # The simulation meta data
            for key in self.metadata.keys():
                file.attrs[key] = self.metadata[key]

            file.attrs['snapshots'] = self.snap_meta

        # Now get the data
        for snap in self.snaps:
            print("Formatting snap {}".format(snap))

            # Load stuff in from original file formats
            subhalos, n_halos = self.loader(path=self.path+'/raw/', snapshot=snap, fields=self.all_fields.keys())

            # Format values
            if n_halos > 0:
                # Put mass in Msun/h
                for key in self.mass_e8_keys:
                    if key in subhalos.keys():
                        subhalos[key] = subhalos[key] * 1e10
                for key in self.mass_add_h_keys:
                    if key in subhalos.keys():
                        subhalos[key] = subhalos[key] * self.h
                # Put position in Mpc/h
                for key in self.pos_kpc_keys:
                    if key in subhalos.keys():
                        subhalos[key] = subhalos[key] / 1000
                # Put time in Gyr/h
                for key in self.inv_time_keys:
                    if key in subhalos.keys():
                        subhalos[key] = subhalos[key] * .978

            # We want to keep a few basic properties together
            dtype_basic = []
            for key in self.basic_fields.keys():
                for subkey in range(len(self.basic_fields[key])):
                    dtype_basic.append(self.basic_fields[key][subkey][:2])

            basic_properties = np.empty(n_halos,dtype=dtype_basic)
            basic_units = {}
            basic_h = {}

            for key in self.basic_fields.keys():
                n_subkeys = len(self.basic_fields[key])
                if n_subkeys > 1:
                    for subkey in range(n_subkeys):
                        new_key = self.basic_fields[key][subkey][0]
                        if n_halos > 0:
                            basic_properties[new_key] = subhalos[key][:,subkey]
                        basic_units[new_key] = self.basic_fields[key][subkey][2]
                        basic_h[new_key] = self.basic_fields[key][subkey][3]
                else:
                    new_key = self.basic_fields[key][0][0]
                    if n_halos > 0:
                        basic_properties[new_key] = subhalos[key]
                    basic_units[new_key] = self.basic_fields[key][0][2]
                    basic_h[new_key] = self.basic_fields[key][0][3]

            # We'll sort everything by mass (descending)
            sorted_inds = np.argsort(basic_properties['mass'])[::-1]
            basic_properties = basic_properties[sorted_inds]
            for key in self.other_fields.keys():
                if n_halos > 0:
                    subhalos[key] = subhalos[key][sorted_inds]
                else:
                    subhalos[key] = np.zeros((0,len(self.other_fields[key])))

            # Now we want to know the indices where various mass cuts can
            # be applied. We'll do it in steps of 0.1 dex
            mass_cuts = np.zeros(141,dtype=[('min_mass','f'),('index','i')])
            mass_cuts['min_mass'] = np.logspace(6,20,141)[::-1]

            for i in range(len(mass_cuts)):
                inds = np.where(basic_properties['mass']>=mass_cuts['min_mass'][i])[0]
                if len(inds) == 0:
                    mass_cuts['index'][i] = 0
                else:
                    mass_cuts['index'][i] = max(inds)+1

            # Now put it in the file
            with h5py.File(self.path+'/data.hdf5', "a") as file:

                snap_grp = file.create_group("Snapshot {}".format(snap))

                snap_grp.create_dataset('mass_cuts',data=mass_cuts)

                for key in basic_properties.dtype.names:
                    snap_grp.create_dataset(key,data=basic_properties[key])
                    snap_grp[key].attrs['units'] = basic_units[key]
                    snap_grp[key].attrs['h dependence'] = basic_h[key]

                for key in self.other_fields.keys():
                    n_subkeys = len(self.other_fields[key])
                    if n_subkeys > 1:
                        for subkey in range(n_subkeys):
                            new_key = self.other_fields[key][subkey][0]
                            if new_key[:4] != 'none':
                                snap_grp.create_dataset(new_key,data=subhalos[key][:,subkey])
                                snap_grp[new_key].attrs['units'] = self.other_fields[key][subkey][2]
                                snap_grp[new_key].attrs['h dependence'] = self.other_fields[key][subkey][3]
                    else:
                        new_key = self.other_fields[key][0][0]
                        snap_grp.create_dataset(new_key,data=subhalos[key])
                        snap_grp[new_key].attrs['units'] = self.other_fields[key][0][2]
                        snap_grp[new_key].attrs['h dependence'] = self.other_fields[key][0][3]

                snap_meta = self.snap_meta[self.snap_meta['index'] == snap]
                snap_grp.attrs['metadata'] = snap_meta
