# To do list
# - add restart option
import matplotlib.pyplot as plt
from matplotlib import cm
cmap = cm.get_cmap('viridis_r')

import datetime
import os
import warnings

from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import h5py
import numpy as np
from scipy.interpolate import interp1d

from simim import _paths
from simim._timing import _timer
from simim.siminterface.simhandler import simhandler
from simim.siminterface._sims import _checksim

class lightcone():
    """class for handling lightcone generation."""

    def __init__(self,
                 sim,
                 name,
                 openangle, aspect=1,
                 redshift_min=0, redshift_max='max',
                 minimum_mass=0,
                 mode = 'box'
                 ):
        """Initialize light cone generator.

        Specify the simulation and light cone dimensions that will be used
        for creating light cones.

        Parameters
        ----------
        sim : str
            The name of the simulation to be used. The simulation must be
            supported by simim and downloaded in the simim_resources path
        name : str
            The name to give to generated light cones. They will be saved in
            the path [simim_resources]/lightcones/[sim]/[name]/lc_[number].hdf5
        openangle : float
            The opening angle of the light cone, specified in degrees.
            For circular light cones this will be the diameter of the
            light cone, for square light cones this will be the side length
            of the box, for rectangular light cones this will be the length
            of the longer side
        aspect : float beteen 0 and 1, optional
            The aspect ratio for the sides of a rectangular box, the long
            side is determined by openangle and the short side is aspect
            x openangle
        redshift_min : float, optional
            Minimum redshift to include in the lightcone, default is 0
        redshift_max : float or 'max', optional
            Maximum redshift to include in the lightcone, default is the
            maximum redshift of the simulation
        minimum_mass : float (optional)
            the minimum halo mass to include, in Msun/h. Default is 0
            (include all halos).
        mode : {'box', 'circle'}, optional
            defines the profile of the light cone - circular or rectangular

        Returns
        -------
        No return
        """

        # Check that we can handle the specified sim
        _checksim(sim)

        paths = _paths._paths()
        if sim in paths.paths.keys():
            self.sim_path = paths.paths[sim]
        else:
            raise ValueError("Simulation {} not available. Try installing it or updating the path".format(sim))
        self.sim = sim

        if sim not in paths.lightcones.keys():
            paths._newlcpath(self.sim)
        self.lc_path = paths.lightcones[sim] + '/' + name
        if os.path.exists(self.lc_path):
            print("A file for light cones of this name already exists.")
            print("Light cones already saved may be overwritten.")
            answer = input("Do you wish to proceed? y/n: ")

            while answer != 'y':
                if answer == 'n':
                    print("Aborting setup")
                    raise ValueError("name already in use")
                print("Response not recognized.")
                answer = input("Do you wish to proceed? y/n: ")
        else:
            os.mkdir(self.lc_path)

        # Save parameters
        if mode not in ['box','circle']:
            raise ValueError("Invalid mode")
        self.mode = mode
        self.openangle = openangle * np.pi/180 # Convert to rad

        if aspect > 1:
            raise ValueError("Invalid mode")
        self.aspect = aspect

        self.minimum_mass = minimum_mass

        # Load some sim data and the class to handle extracting data
        self.sim_handler = simhandler(self.sim)
        self.metadata = self.sim_handler.metadata
        self.snap_meta = self.sim_handler.snap_meta
        self.snap_keys_all = self.sim_handler.extract_snap_keys()
        self.snap_keys = np.copy(self.snap_keys_all)
        self.box_edge = self.metadata['box_edge']

        # Deal with redshift stuff
        if redshift_max == 'max':
            redshift_max = np.amax(self.snap_meta['redshift_max'])
        if redshift_min > redshift_max:
            raise ValueError("redshift_min > redshift_max")
        self.redshift_min = redshift_min
        self.redshift_max = redshift_max

        # Cosmology class
        self.cosmo = FlatLambdaCDM(H0=100*self.metadata['cosmo_h']*u.km/u.s/u.Mpc,
                                   Om0=self.metadata['cosmo_omega_matter'],
                                   Ob0=self.metadata['cosmo_omega_baryon'],
                                   Tcmb0=2.7255*u.K)
        z_steps = np.linspace(0,20.05,100000)
        D_steps = self.cosmo.comoving_distance(z_steps).value * self.metadata['cosmo_h']
        self.assign_z = interp1d(D_steps,z_steps)


        # Check that we haven't requested something too big to fit in the box
        backsize = self.cosmo.comoving_transverse_distance(self.redshift_max).value
        backsize = backsize * self.metadata['cosmo_h'] * self.openangle
        if backsize > self.box_edge:
            print(backsize, self.box_edge)
            raise ValueError("Open angle larger than simulation box at max redshift")

        # Figure out the range of snapshots needed
        self.snapshots_use = np.zeros(0,dtype=self.snap_meta.dtype)
        for i in self.snap_meta:
            if self.redshift_min <= i['redshift_max']:
                if self.redshift_max >= i['redshift_min']:
                    self.snapshots_use = np.concatenate((self.snapshots_use,[i]))
        self.snapshots_use = np.sort(self.snapshots_use,order='redshift')

        # Allow copying box?
        if self.sim[:3] in ['Ill']:
            self.copies_max = 2
        elif self.sim[:3] in ['TNG']:
            self.copies_max = 2
        else:
            self.copies_max = 1

    def _initialize_los(self,openradius,rng):
        """Generates a line of sight for simulated light cones

        Parameters
        ----------
        openradius : float
            The radius of a circle surrounding the whole light cone face.
            Should be specified in radians
        rng : numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        LOS_unit :
            A unit vector along the line of sight, in simulation box
            coordinate system
        LC_transformation_matrix :
            A matrix to transform from simulation box coordinates to
            line of sight coordinates
        buffer_coef : float
            Coeficient used in determining the LOS distance that must
            be buffered to avoid overflowing the simulation box edges
        """

        test_geom = False
        while not test_geom:

            # Pick a line of sight, make sure it fits within some limits
            test_a = False
            test_b = False
            while not test_a or not test_b:

                LOS_vec = rng.random(3)

                tan_1 = LOS_vec[0]/LOS_vec[1]
                tan_check_1 = np.tan(openradius + np.pi/180)
                tan_check_2 = np.tan(np.pi/2 - 3*openradius - np.pi/180)
                if tan_1 > tan_check_1 and tan_1 < tan_check_2:
                    test_a = True

                tan_2 = LOS_vec[0]/LOS_vec[2]
                tan_check_3 = np.tan(3*openradius)
                tan_check_4 = np.tan(np.pi/2 - 3*openradius)
                if tan_2 > tan_check_3 and tan_2 < tan_check_4:
                    test_b = True

            # Construct a unit vector along the LOS
            LOS_unit = LOS_vec/np.linalg.norm(LOS_vec)

            # Now test that there's enough room in the box for this thing...
            angles = np.arctan([LOS_unit[1]/LOS_unit[0], LOS_unit[2]/LOS_unit[0], LOS_unit[2]/LOS_unit[1],
                                LOS_unit[0]/LOS_unit[1], LOS_unit[0]/LOS_unit[2], LOS_unit[1]/LOS_unit[2]])

            buffer_coef = 1.01 * np.tan(openradius) / np.sin(np.amin(angles)) # 1.01 is just to provide some padding

            max_distance = np.amax(self.snapshots_use['transverse_distance_max'])
            buffer_distance_max = buffer_coef * max_distance
            in_box_distance = self.box_edge / np.amax(LOS_unit)

            if (self.copies_max*in_box_distance - 2*buffer_distance_max) > 10:
                test_geom = True

        # Construc two perpendicular unit vectors and generate the
        # transformation matrix required to go between them
        u2 = np.cross(LOS_unit,np.array([0,0,1]))
        u2 /= np.linalg.norm(u2)

        u3 = np.cross(LOS_unit,u2)

        LC_transformation_matrix = np.linalg.inv(np.array([LOS_unit,u2,u3]).T)

        return LOS_unit, LC_transformation_matrix, buffer_coef

    def _initialize_pa(self,rng):
        """Generates a position angle

        Parameters
        ----------
        rng : numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        pa : float
            The position angle, in radians
        pa_transformation_matrix :
            A rotation matrix for the position angle
        """

        pa = rng.random() * 2*np.pi
        pa_transformation_matrix = np.array(
            [[1,0,0],
             [0,np.cos(pa),-np.sin(pa)],
             [0,np.sin(pa),np.cos(pa)]])

        return pa, pa_transformation_matrix

    def build_lightcones(self,
                         n,
                         rng=np.random.default_rng()):
        """Code to construct light cones

        Parameters
        ----------
        n : int
            Number of light cones to make
        rng : numpy rng
            Random number generator instance used to generate lightcone
            useful for producing repeatable results.

        Returns
        -------
        none
            Light cones are saved in the lightcones file in the resources
            folder.
        """

        TIMER = _timer(longlongtime=0)
        TIMER.start("WHOLE THING!")

        TIMER2 = _timer(longlongtime=0)


        print_info = True

        self.n = n

        # Set up pointing parameters - box shape
        if self.mode == 'circle':
            radius = self.openangle / 2
        elif self.mode == 'box':
            angle1 = self.openangle
            angle2 = self.openangle * self.aspect
            radius = np.sqrt(angle1**2 + angle2**2) / 2
        else:
            raise ValueError("Invalid mode")

        # Print status
        print("Generating lines of sight.")

        # Set up pointing parameters - starting position
        pointing_start = rng.random((n,3)) * self.box_edge

        # Set up pointing parameters - PA for boxes
        pointing_pa = np.empty(n)
        pointing_pa_matrix = np.empty((n,3,3))

        for i in range(n):
            pointing_pa[i], pointing_pa_matrix[i] = self._initialize_pa(rng=rng)

        # Pick which dimensions to flip
        pointing_flips = rng.choice([-1,1],size=(n,3))

        # Set up pointing parameters - line of sight, buffer
        pointing_los = np.empty((n,3))
        pointing_los_matrix = np.empty((n,3,3))
        pointing_buffer_coef = np.empty(n)

        for i in range(n):
            pointing_los[i], pointing_los_matrix[i], pointing_buffer_coef[i] = self._initialize_los(openradius=radius,rng=rng)

        # Set up redshift indexing
        # index will be the index of the first halo above the specified redshift
        redshift_index = np.zeros(100,dtype=[('redshift','f'),('index','int')])
        redshift_index['redshift'] = (10**np.linspace(np.log10(1+0),np.log10(1+20.05),100))-1

        # Set up halo counter
        n_halos = np.zeros(n,dtype='int')

        # Data type for parent index properties
        simindex_dtype = [('snap','int'),('index','int')]

        # Start and end distances for the light cone,
        # since we'll primarily work in terms of distance
        distance_zmin = (self.cosmo.comoving_distance(self.redshift_min).value
                         * self.metadata['cosmo_h'])
        distance_zmax = (self.cosmo.comoving_distance(self.redshift_max).value
                         * self.metadata['cosmo_h'])

        # Initialize HDF5 file with metadata and a light cone dataset
        # Print status
        print("Creating files and adding metadata.")
        for i in range(n):
            with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(i),'w') as file:
                for key in self.metadata.keys():
                    file.attrs[key] = self.metadata[key]
                file.attrs['snapshots'] = self.snapshots_use
                file.attrs['parent sim'] = self.sim

                file.attrs['minimum redshift'] = self.redshift_min
                file.attrs['maximum redshift'] = self.redshift_max

                file.attrs['open angle'] = self.openangle
                file.attrs['shape'] = self.mode
                file.attrs['aspect ratio'] = self.aspect
                file.attrs['start position'] = pointing_start[i]
                file.attrs['position angle'] = pointing_pa[i]
                file.attrs['position angle transformation matrix'] = pointing_pa_matrix[i]
                file.attrs['box flips'] = pointing_flips[i]
                file.attrs['line of sight vector'] = pointing_los[i]
                file.attrs['line of sight transformation matrix'] = pointing_los_matrix[i]

                file.attrs['mass cut'] = self.minimum_mass

                dtime = datetime.datetime.today()
                file.attrs['created'] = dtime.isoformat(' ')
                file.attrs['finished'] = False
                file.attrs['status_initialized'] = True
                file.attrs['status_basic'] = False
                file.attrs['status_extra_initialized'] = False
                file.attrs['status_extra_completed'] = False


                idx = file.create_group('Indexing')
                idx.create_dataset('redshift_indices',data=redshift_index)

        # Open simulation file
        with h5py.File(self.sim_path+'/data.hdf5',"r") as sim_file:

            # Loop through boxes
            for snap_ind in range(len(self.snapshots_use)):

                snap_meta = self.snapshots_use[snap_ind]
                snap_name = 'Snapshot {}'.format(snap_meta['index'])
                print("\033[1m"+"Collecting sources from Snapshot {}.           ".format(snap_meta['index'])+"\033[0m")
                # TIMER2.start("SNAPSHOT {}".format(snap_meta['index']))

                # We will work in distance units mostly
                # Calculate distance to use in box
                start_distance = np.amax([snap_meta['distance_min'],distance_zmin])
                end_distance = np.amin([snap_meta['distance_max'],distance_zmax])
                use_distance = end_distance - start_distance

                # Load the snapshot data for the basic fields
                snap_grp = sim_file[snap_name]
                mass_index = self.sim_handler.get_mass_index(self.minimum_mass,snap_meta['index'])

                masses = snap_grp['mass'][:mass_index]
                mass_indices_final = np.where(masses >= self.minimum_mass)[0]
                if len(mass_indices_final) > 0:
                    mass_index_final = np.amax(mass_indices_final)+1
                else:
                    mass_index_final = 0
                masses = masses[:mass_index_final]

                positions0 = np.empty((mass_index_final,3))
                positions0[:,0] = snap_grp['pos_x'][:mass_index_final]
                positions0[:,1] = snap_grp['pos_y'][:mass_index_final]
                positions0[:,2] = snap_grp['pos_z'][:mass_index_final]

                # Iterate through each light cone
                for lc_ind in range(n):
                    print("   Working on light cone {}/{}".format(lc_ind+1,n), end='\r')

                    # Compute how far we can go in the box
                    buffer_distance = snap_meta['transverse_distance_max'] * pointing_buffer_coef[lc_ind]
                    in_box_distance = self.box_edge / np.amax(pointing_los[lc_ind])

                    iters = int(np.ceil(use_distance / (in_box_distance-2*buffer_distance)))

                    # Expand the box by making copies if necessary
                    positions = np.copy(positions0)

                    copies_n = int(1)
                    copies_n3d = int(1)
                    box_size = self.box_edge

                    while iters > 25 or iters <= 0:
                        if copies_n < self.copies_max:
                            for dimension_ind in range(3):
                                positions_copy = np.copy(positions)
                                positions_copy[:,dimension_ind] += box_size
                                positions = np.concatenate((positions,positions_copy),axis=0)

                            in_box_distance *= 2
                            box_size *= 2

                            iters = int(np.ceil(use_distance / (in_box_distance-2*buffer_distance)))
                            copies_n *= int(2)
                            copies_n3d *= int(8)
                        elif iters <= 0:
                            raise ValueError("Something went wrong calculating iterations")
                        else:
                            break

                    # Now shift the box to start where we want to start
                    positions = (positions * pointing_flips[lc_ind]) % box_size

                    positions = (positions - pointing_start[lc_ind]
                                 - start_distance * pointing_los[lc_ind]
                                 + buffer_distance * pointing_los[lc_ind])

                    if print_info:
                        print("   Working on light cone {}/{}... copies={}, iterations={}".format(lc_ind+1,n,copies_n,iters), end='\r')

                    for iter_ind in range(iters):

                        # Get everything into LOS units
                        iter_position = positions - iter_ind*use_distance/iters*pointing_los[lc_ind]
                        iter_position %= box_size
                        iter_position = np.matmul(pointing_los_matrix[lc_ind], iter_position.T).T

                        # And then get stuff into the basis of the position
                        # angle
                        iter_position = np.matmul(pointing_pa_matrix[lc_ind], iter_position.T).T

                        # Select halos within field
                        candidate_distances = np.linalg.norm(
                            iter_position
                            + np.array([start_distance + iter_ind*use_distance/iters - buffer_distance,0,0]),
                            axis = 1)

                        if self.mode == 'circle':
                            r = np.sqrt(iter_position[:,1]**2 + iter_position[:,2]**2) / candidate_distances
                            indices1 = np.where(r < radius)
                        elif self.mode == 'box':
                            d1 = np.abs(iter_position[:,1]/candidate_distances)
                            d2 = np.abs(iter_position[:,2]/candidate_distances)
                            indices1 = np.where((d1 <= angle1/2) & (d2 <= angle2/2))

                        iter_min_distance = start_distance + iter_ind*use_distance/iters
                        iter_max_distance = start_distance + (iter_ind+1)*use_distance/iters
                        iter_max_distance = np.amin([iter_max_distance,end_distance])

                        indices2 = np.where((candidate_distances < iter_max_distance)
                                            & (candidate_distances >= iter_min_distance))

                        selected_indices = np.intersect1d(indices1,indices2)
                        selected_distances = candidate_distances[selected_indices]

                        n_selected = len(selected_indices)
                        n_halos[lc_ind] += n_selected

                        # Sort by distance (which will become z)
                        sorted = np.argsort(selected_distances)
                        selected_indices = selected_indices[sorted]
                        selected_distances = selected_distances[sorted]

                        # Set the LC positions
                        pos_x = iter_position[:,1][selected_indices]
                        pos_y = iter_position[:,2][selected_indices]
                        pos_z = iter_position[:,0][selected_indices] + start_distance + iter_ind*use_distance/iters - buffer_distance

                        # Compute z, ra, dec for selected halos
                        ra = pos_x / selected_distances
                        dec = pos_y / selected_distances
                        halo_redshift = self.assign_z(selected_distances)

                        # Get masses of halos
                        halo_mass = masses[selected_indices % len(positions0)]

                        # Save indices
                        sim_index = np.zeros(
                            len(pos_x), dtype = simindex_dtype)
                        sim_index['snap'] = snap_meta['index']
                        sim_index['index'] = selected_indices % len(positions0)

                        # Add selected halos to some temporary storage
                        props = ['pos_x','pos_y','pos_z','ra','dec','redshift','mass']
                        vals = [pos_x,pos_y,pos_z,ra,dec,halo_redshift,halo_mass]
                        units = [self.sim_handler.key_units['pos_x'],
                                 self.sim_handler.key_units['pos_y'],
                                 self.sim_handler.key_units['pos_z'],
                                 'radians','radians','None',
                                 self.sim_handler.key_units['mass']]
                        h_dependence = [self.sim_handler.key_h_dependence['pos_x'],
                                        self.sim_handler.key_h_dependence['pos_y'],
                                        self.sim_handler.key_h_dependence['pos_z'],
                                        0,0,0,
                                        self.sim_handler.key_h_dependence['mass']]

                        with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(lc_ind),'a') as lc_file:

                            if iter_ind == 0 and snap_ind == 0:
                                lc = lc_file.create_group('Lightcone Basic')

                                for prop_ind in range(len(props)):
                                    prop = props[prop_ind]
                                    val = vals[prop_ind]
                                    lc.create_dataset(prop,data=val,maxshape=(None,),chunks=(2500,))
                                    lc[prop].attrs['units'] = units[prop_ind]
                                    lc[prop].attrs['h dependence'] = h_dependence[prop_ind]

                                lc_file['Indexing'].create_dataset('sim_indices',data=sim_index,maxshape=(None,),chunks=(2500,))

                            else:
                                for prop_ind in range(len(props)):
                                    prop = props[prop_ind]
                                    val = vals[prop_ind]

                                    lc_file['Lightcone Basic'][prop].resize((n_halos[lc_ind],))
                                    lc_file['Lightcone Basic'][prop][n_halos[lc_ind]-n_selected:] = val

                                lc_file['Indexing']['sim_indices'].resize((n_halos[lc_ind],))
                                lc_file['Indexing']['sim_indices'][n_halos[lc_ind]-n_selected:] = sim_index

                            # Calculate z indexing and save
                            redshift_index = lc_file['Indexing']['redshift_indices'][:]
                            for i in range(1,len(redshift_index)):
                                inds = np.where(halo_redshift<=redshift_index['redshift'][i])[0]
                                redshift_index['index'][i] += len(inds)
                            lc_file['Indexing']['redshift_indices'][:] = redshift_index

                            print(' '*80,end='\r')
                # TIMER2.lap()

        # Indicate that light cones are completed
        for i in range(n):
            with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(i),'a') as file:

                lc = file.create_group('Lightcone Full')
                for key in file['Lightcone Basic'].keys():
                    lc[key] = file['Lightcone Basic'][key]
                file.attrs['status_basic'] = True
                file.attrs['finished'] = dtime.isoformat(' ')


        TIMER.lap()

    def add_properties(self,keys):
        """Select additional properties to include in the light cone.
        Should not be used for properties with position information
        (ie velocity) which have to be transformed into the correct
        coordinates. Use add_pos_properties instead

        Parameters
        ----------
        keys : list
            list containing the keys used to identify fields to include in the
            light cone

        Returns
        -------
        none
        """

        # Make sure the keys exist, but aren't already included
        pos_keys = ['v_x','v_y','v_z',
                    'spin_x','spin_y','spin_z',
                    'cm_x','cm_y','cm_z']

        with h5py.File(self.lc_path+'/lc_0000.hdf5','r') as file:
            old_lc_keys = [i for i in file['Lightcone Full'].keys()]

        new_lc_keys = []
        for key in keys:
            if key in pos_keys:
                warnings.warn("Key '{}' is positional, use add_pos_properties instead".format(key))
            elif key in old_lc_keys:
                warnings.warn("Key '{}' already in light cone. Skipping".format(key))
            elif key not in self.snap_keys_all and key != 'masscheck':
                warnings.warn("Key '{}' not found".format(key))
            else:
                new_lc_keys.append(key)

        keys = np.sort(new_lc_keys)

        TIMER = _timer()
        TIMER2 = _timer()

        TIMER.start("WHOLE THING")
        # Note in LC files that we're adding properties
        for i in range(self.n):
            with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(i),'a') as file:
                file.attrs['status_extra_initialized'] = True
                file.attrs['status_extra_completed'] = False

                shape = file['Indexing']['sim_indices'].shape
                chunks = file['Indexing']['sim_indices'].chunks
                if shape[0] < chunks[0]:
                    chunks = shape
                for key in keys:
                    file['Lightcone Full'].create_dataset(key,shape=shape,chunks=chunks)
                    if key == 'masscheck':
                        file['Lightcone Full'][key].attrs['units'] = self.sim_handler.key_units['mass']
                        file['Lightcone Full'][key].attrs['h dependence'] = self.sim_handler.key_h_dependence['mass']
                    else:
                        file['Lightcone Full'][key].attrs['units'] = self.sim_handler.key_units[key]
                        file['Lightcone Full'][key].attrs['h dependence'] = self.sim_handler.key_h_dependence[key]

        # Open simulation file
        with h5py.File(self.sim_path+'/data.hdf5',"r") as sim_file:

            # Loop through snapshots
            for snap_ind in range(len(self.snapshots_use)):

                # Collect snap information
                snap_meta = self.snapshots_use[snap_ind]
                snap_name = 'Snapshot {}'.format(snap_meta['index'])
                print("\033[1m"+"Collecting sources from Snapshot {}.           ".format(snap_meta['index'])+"\033[0m")
                TIMER2.start("SNAPSHOT {}".format(snap_meta['index']))

                # Loop through keys
                for key in keys:

                    # Load the snapshot data for the basic fields
                    snap_grp = sim_file[snap_name]
                    mass_index = self.sim_handler.get_mass_index(self.minimum_mass,snap_meta['index'])

                    masses = snap_grp['mass'][:mass_index]
                    mass_indices_final = np.where(masses >= self.minimum_mass)[0]
                    if len(mass_indices_final) > 0:
                        mass_index_final = np.amax(mass_indices_final)+1
                    else:
                        mass_index_final = 0

                    if key == 'masscheck':
                        values = snap_grp['mass'][:mass_index_final]
                    else:
                        values = snap_grp[key][:mass_index_final]

                    for lc_ind in range(self.n):
                        print("   Working on light cone {}/{}          ".format(lc_ind+1,self.n), end='\r')

                        # Get indices of snap
                        with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(lc_ind),'a') as lc_file:
                            indexing = lc_file['Indexing']['sim_indices'][:]
                            start = np.where(indexing['snap'] == snap_meta['index'])[0]
                            if len(start) > 0:
                                start = np.amin(start)
                            else:
                                start = 0
                            indexing = indexing[indexing['snap']==snap_meta['index']]

                            # Append the data
                            lc_file['Lightcone Full'][key][start:start+len(indexing)] = values[indexing['index']]

                TIMER2.lap()

        # Note in LC files that we're done
        for i in range(self.n):
            with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(i),'a') as file:
                file.attrs['status_extra_initialized'] = True
                file.attrs['status_extra_completed'] = True

        TIMER.lap()

    def add_pos_properties(self,keys):
        """Select additional properties to include in the light cone for
        properties with position information (ie velocity).

        Parameters
        ----------
        keys : list
            list containing the keys used to identify fields to include in the
            light cone (exclude the _x, _y, _z components, ie for velocity
            give keys=['v'] not keys=['v_x','v_y','v_z'])

        Returns
        -------
        none
        """

        # Make sure the keys exist, but aren't already included
        pos_keys = ['pos','v','spin','cm']

        with h5py.File(self.lc_path+'/lc_0000.hdf5','r') as file:
            old_lc_keys = [i[:-2] for i in file['Lightcone Full'].keys() if i[:-2] in pos_keys]
        old_lc_keys = np.unique(old_lc_keys)

        new_lc_keys = []
        for key in keys:
            if key in old_lc_keys:
                warnings.warn("Key '{}' already in light cone. Skipping".format(key))
            elif key+'_x' not in self.snap_keys_all:
                warnings.warn("Key '{}' not found".format(key))
            else:
                new_lc_keys.append(key)

        keys = np.sort(new_lc_keys)

        TIMER = _timer()
        TIMER2 = _timer()

        TIMER.start("WHOLE THING")
        # Note in LC files that we're adding properties
        for i in range(self.n):
            with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(i),'a') as file:
                file.attrs['status_extra_initialized'] = True
                file.attrs['status_extra_completed'] = False

                shape = file['Indexing']['sim_indices'].shape
                chunks = file['Indexing']['sim_indices'].chunks
                if shape[0] < chunks[0]:
                    chunks = shape
                for key in keys:
                    for subkey in ['_x','_y','_z']:
                        file['Lightcone Full'].create_dataset(key+subkey,shape=shape,chunks=chunks)
                        file['Lightcone Full'][key+subkey].attrs['units'] = self.sim_handler.key_units[key+subkey]
                        file['Lightcone Full'][key+subkey].attrs['h dependence'] = self.sim_handler.key_h_dependence[key+subkey]

        # Open simulation file
        with h5py.File(self.sim_path+'/data.hdf5',"r") as sim_file:

            # Loop through snapshots
            for snap_ind in range(len(self.snapshots_use)):

                # Collect snap information
                snap_meta = self.snapshots_use[snap_ind]
                snap_name = 'Snapshot {}'.format(snap_meta['index'])
                print("\033[1m"+"Collecting sources from Snapshot {}.                        ".format(snap_meta['index'])+"\033[0m")
                TIMER2.start("SNAPSHOT {}".format(snap_meta['index']))

                # Loop through keys
                for key in keys:

                    # Load the snapshot data for the basic fields
                    snap_grp = sim_file[snap_name]
                    mass_index = self.sim_handler.get_mass_index(self.minimum_mass,snap_meta['index'])

                    masses = snap_grp['mass'][:mass_index]
                    mass_indices_final = np.where(masses >= self.minimum_mass)[0]
                    if len(mass_indices_final) > 0:
                        mass_index_final = np.amax(mass_indices_final)+1
                    else:
                        mass_index_final = 0

                    values = np.array([
                        snap_grp[key+'_x'][:mass_index_final],
                        snap_grp[key+'_y'][:mass_index_final],
                        snap_grp[key+'_z'][:mass_index_final]])

                    for lc_ind in range(self.n):
                        print("   Working on light cone {}/{}          ".format(lc_ind+1,self.n), end='\r')

                        with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(lc_ind),'a') as lc_file:
                            # Get indices of snap
                            indexing = lc_file['Indexing']['sim_indices'][:]
                            start = np.where(indexing['snap'] == snap_meta['index'])[0]
                            if len(start) > 0:
                                start = np.amin(start)
                            else:
                                start = 0
                            indexing = indexing[indexing['snap']==snap_meta['index']]

                            # Trim values to the indices
                            lc_values = values[:,indexing['index']]

                            # Get the matrices and do transformation
                            los_matrix = lc_file.attrs['line of sight transformation matrix']
                            pa_matrix = lc_file.attrs['position angle transformation matrix']
                            lc_values = np.matmul(los_matrix,lc_values)
                            lc_values = np.matmul(pa_matrix,lc_values)


                            # Append the data
                            lc_file['Lightcone Full'][key+'_x'][start:start+len(indexing)] = lc_values[1]
                            lc_file['Lightcone Full'][key+'_y'][start:start+len(indexing)] = lc_values[2]
                            lc_file['Lightcone Full'][key+'_z'][start:start+len(indexing)] = lc_values[0]

                TIMER2.lap()

        # Note in LC files that we're done
        for i in range(self.n):
            with h5py.File(self.lc_path+'/lc_{:04d}.hdf5'.format(i),'a') as file:
                file.attrs['status_extra_initialized'] = True
                file.attrs['status_extra_completed'] = True

        TIMER.lap()
