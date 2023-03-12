import warnings

import h5py
import numpy as np

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

from simim import _paths
from simim._handlers import handler
from simim.siminterface._sims import _checksim

class snaphandler(handler):
    def __init__(self,path,snap,redshift,cosmo,box_edge):
        super().__init__(path,objectname='Snapshot',groupname='Snapshot {}'.format(snap))

        self.redshift = redshift
        self.cosmo = cosmo
        self.box_edge = box_edge
        self.box_edge_no_h = box_edge / self.h

        self.extra_props['cosmo'] = self.cosmo
        self.extra_props['redshift'] = self.redshift
        self.extra_props['box_edge'] = self.box_edge
        self.extra_props['box_edge_no_h'] = self.box_edge / self.h

class simhandler():
    """Class to handle I/O for simulation hdf5 files"""

    def __init__(self,sim,init_snaps=False):
        """Initialize simulation access"""

        # Check that we can handle the specified sim
        _checksim(sim)

        # Set up a place to keep the data
        paths = _paths._paths()
        if sim in paths.paths.keys():
            self.path = paths.paths[sim]
        else:
            raise ValueError("Simulation {} not available. Try installing it or updating the path".format(sim))
        self.sim = sim

        # Get the metadata
        self.metadata = {}
        with h5py.File(self.path+'/data.hdf5','r') as file:
            for key in file.attrs.keys():
                self.metadata[key] = file.attrs[key]
            self.snap_meta = file.attrs['snapshots']
        self.h = self.metadata['cosmo_h']

        # Sort out redshift matching to stuff
        snaps_sorted = np.sort(np.copy(self.snap_meta),order='redshift')
        self.z_bins = [snap['redshift_min'] for snap in snaps_sorted]
        self.z_bins.append(snaps_sorted[-1]['redshift_max'])
        self.z_bin_snaps = [snap['index'] for snap in snaps_sorted]

        # Make the cosmology
        self.cosmo = FlatLambdaCDM(H0=100*self.metadata['cosmo_h']*u.km/u.s/u.Mpc,
                                   Om0=self.metadata['cosmo_omega_matter'],
                                   Ob0=self.metadata['cosmo_omega_baryon'],
                                   Tcmb0=2.7255*u.K)
        self.box_edge = self.metadata['box_edge']
        self.box_edge_no_h = self.box_edge / self.h

        # Get keys and units
        with h5py.File(self.path+'/data.hdf5','r') as file:
            snaps = [i for i in file.keys()]
            keys = [i for i in file[snaps[0]].keys() if i != 'mass_cuts']

            key_units = {key:file[snaps[0]][key].attrs['units'] for key in keys}
            key_h_dependence = {key:file[snaps[0]][key].attrs['h dependence'] for key in keys}

        self.keys = keys
        self.key_units = key_units
        self.key_h_dependence = key_h_dependence

        if init_snaps:
            print('Initializing snapshots, this may take a few seconds')
            self.snap_handlers = {}
            # Set up snapshot handlers
            for i in range(len(self.snap_meta)):

                snap = self.snap_meta['index'][i]
                redshift = self.snap_meta['redshift'][i]

                self.snap_handlers[str(snap)] = snaphandler(self.path+'/data.hdf5',snap,redshift,self.cosmo,self.box_edge)
            print("Snapshots initialized.")
        self.init_snaps = init_snaps

    def extract_snap_meta(self,snap):
        """Get the meta-data for a snapshot

        Parameters
        ----------
        snap : int
            Number of snapshot to be extracted

        Returns
        -------
        snap_meta
            The meta data for the requested snapshot
        """

        if snap in self.snap_meta['index']:
            snap_meta = self.snap_meta[self.snap_meta['index'] == snap][0]
            return snap_meta
        else:
            raise ValueError("Snapshot not found")

    def z_to_snap(self,z):
        """Determine the snapshot corresponding to a particular redshift

        Parameters
        ----------
        z : float
            Redshift to search for

        Returns
        -------
        snap_ind
            The index number of the snapshot matching the requested redshift
        """

        if z > np.amax(self.z_bins) or z < 0:
            raise ValueError("z out of range")
        bin = np.digitize(z,self.z_bins)-1
        bin_id = self.z_bin_snaps[bin]

        return bin_id

    def extract_snap_keys(self):
        """Get the fields a file

        Parameters
        ----------
        none

        Returns
        -------
        keys
            The fields of each snapshot

        """

        return self.keys

    def get_mass_index(self,mass,snap,in_h_units=False):
        """Find the indices above a specified mass

        Parameters
        ----------
        mass : float
            Minimum mass to access in Msun/h units
        snap : int
            Number of snapshot to be extracted
        in_h_units : bool (default=False)
            If True, mass will be taken to have units including little h,
            otherwise, it will be assumed to have units with no h dependence.

        Returns
        -------
        index : int
            The index
        """

        if not snap in self.snap_meta['index']:
            raise ValueError("Snapshot not found")

        with h5py.File(self.path+'/data.hdf5','r') as file:
            mass_cuts = file["Snapshot {}".format(snap)]['mass_cuts']

            vals = mass_cuts['min_mass'] - (mass / self.h)
            vals = vals[vals>0]
            if len(vals) < 1:
                index = 0
            elif len(vals) == len(mass_cuts):
                index = mass_cuts['index'][-1]
            else:
                index = mass_cuts['index'][len(vals)]

        return index

    def get_snap(self,snap):

        snap_meta = self.extract_snap_meta(snap)

        if self.init_snaps:
            return self.snap_handlers[str(snap_meta['index'])]
        else:
            snap = snap_meta['index']
            redshift = snap_meta['redshift']
            return snaphandler(self.path+'/data.hdf5',snap,redshift,self.cosmo,self.box_edge)

    def set_property_range(self,property_name=None,pmin=-np.inf,pmax=np.inf,reset=True, in_h_units=False):
        if not self.init_snaps:
            raise ValueError("This handler instance was not initialized with snapshots available")
        for i in range(len(self.snap_meta)):
            snap = self.snap_meta['index'][i]
            handler = self.get_snap(snap)
            handler.set_property_range(property_name=property_name,pmin=pmin,pmax=pmax,reset=reset,in_h_units=in_h_units)

    def make_property(self, property, rename=None, kw_remap={}, other_kws={}, overwrite=False, use_all_inds=False, write=False):
        if not self.init_snaps and not write:
            raise ValueError("This handler instance was not initialized with snapshots available")
        for i in range(len(self.snap_meta)):
            snap = self.snap_meta['index'][i]
            print("\033[1m"+"Assigning props for Snapshot {}.  ".format(snap)+"\033[0m",end='\r')
            handler = self.get_snap(snap)

            handler.make_property(property=property,
                                  rename=rename,
                                  kw_remap=kw_remap,
                                  other_kws=other_kws,
                                  overwrite=overwrite,
                                  use_all_inds=use_all_inds)

            if write:
                if isinstance(rename,str):
                    rename = [rename]
                if rename == None:
                    names = property.names
                elif len(rename) != property.n_props:
                    raise ValueError("Length of rename list doesn't match number of properties")
                else:
                    names = rename

                if i==0:
                    with h5py.File(handler.path,'a') as file:
                        for name in names:
                            if name in file[handler.groupname].keys():
                                if not overwrite:
                                    raise ValueError("Property {} already exists".format(name))
                                elif 'userdefined' not in file[handler.groupname][name].attrs.keys():
                                    raise ValueError("Property {} is not userd-defined and cannot be overwritten".format(name))
                                else:
                                    warnings.warn("Property {} already exists, overwriting".format(name))

                handler.write_property(*names,overwrite=overwrite)
                handler.unload_property
        print()

    def delete_property(self,*property_names):
        for i in range(len(self.snap_meta)):
            snap = self.snap_meta['index'][i]
            handler = self.get_snap(snap)

            if i==0:
                with h5py.File(handler.path,'a') as file:
                    for name in property_names:
                        if name in file[handler.groupname].keys():
                            if 'userdefined' not in file[handler.groupname][name].attrs.keys():
                                raise ValueError("Property {} is not userd-defined and cannot be deleted".format(name))
                            elif not file[handler.groupname][name].attrs['userdefined']:
                                raise ValueError("Property {} is not userd-defined and cannot be deleted".format(name))

            handler.delete_property(*property_names)

    def snap_stat(self, stat_function, kwargs, kw_remap={}, other_kws={},
                  give_args_in_h_units=False, use_all_inds=False):

        vals = []
        redshifts = []
        for i in range(len(self.snap_meta)):
            snap = self.snap_meta['index'][i]
            print("\033[1m"+"Collecting sources from Snapshot {}.  ".format(snap)+"\033[0m",end='\r')
            redshifts.append(self.snap_meta['redshift'][i])

            handler = self.get_snap(snap)
            vals.append(handler.eval_stat(stat_function,kwargs,kw_remap,other_kws=other_kws,use_all_inds=use_all_inds,give_args_in_h_units=give_args_in_h_units))
        print("")
            
        return vals, redshifts
