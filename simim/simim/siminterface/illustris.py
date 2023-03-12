"""Code to download Illustris TNG or Illustris group catalogs

Created: 31 March 2021, r p keenan
"""

import os
import requests
import warnings
import h5py
import re

import numpy as np

from simim.siminterface import illustris_datahandling as idh
from simim.siminterface._rawsiminterface import sim_catalogs, snapshot
from simim.siminterface._sims import _checksim

_IllustrisAPIKey = '07e2073a308296154fc216ae62145316'

class illustris_catalogs(sim_catalogs):
    def __init__(self,
                 sim, path='auto',
                 snaps='all',
                 updatepath=True,
                 ):
        sim_catalogs.__init__(self,sim, path, snaps, updatepath)

        # Identify keys that will require unit conversions
        self.mass_e8_keys = ['SubhaloBHMass','SubhaloBHMdot','SubhaloMass',
                          'SubhaloMassInHalfRad','SubhaloMassInHalfRadType',
                          'SubhaloMassInMaxRad','SubhaloMassInMaxRadType',
                          'SubhaloMassInRad','SubhaloMassInRadType',
                          'SubhaloMassType','SubhaloStellarPhotometricsMassInRad',
                          'SubhaloWindMass'
                          ]
        self.pos_kpc_keys = ['SubhaloCM','SubhaloHalfmassRad',
                         'SuhbaloHalfmassRadType','SubhaloPos','SubhaloSpin',
                         'SubhaloStellarPhotometricsRad','SubhaloVmaxRad'
                         ]
        self.inv_time_keys = ['SubhaloBHMdot']

        self.basic_fields = {# Spatial position within the periodic box (of the
                        # particle with the minium gravitational potential
                        # energy). Comoving coordinate.
                        'SubhaloPos':[('pos_x','f','Mpc/h',-1),('pos_y','f','Mpc/h',-1),('pos_z','f','Mpc/h',-1)],

                        # Comoving radius containing half of the total mass
                        # (SubhaloMass) of this Subhalo.
                        'SubhaloHalfmassRad':[('r_hm','f','Mpc/h',-1)],

                        # Total mass of all member particle/cells which are
                        # bound to this Subhalo, of all types. Particle/cells
                        # bound to subhaloes of this Subhalo are NOT accounted
                        # for.
                        'SubhaloMass':[('mass','f','Msun/h',-1)],

                        # Peculiar velocity of the group, computed as the sum
                        # of the mass weighted velocities of all
                        # particles/cells in this group, of all types. No unit
                        # conversion is needed.
                        'SubhaloVel':[('v_x','f','km/s',0),('v_y','f','km/s',0),('v_z','f','km/s',0)]
                        }

        if self.sim[:3] == 'TNG' and self.sim[-4:] != 'Dark':
            self.basic_fields['SubhaloFlag'] = [('flag','int','none',0)]
            # Flag field indicating suitability of this subhalo for
            # certain types of analysis. If zero, this subhalo should
            # generally be excluded, and is not thought to be of
            # cosmological origin. That is, it may have formed within
            # an existing halo, or is possibly a baryonic fragment of
            # a disk or other galactic structure identified by Subfind.
            # If one, this subhalo should be considered a 'galaxy' or
            # 'satellite' of cosmological origin. (Note: always true
            # for centrals). See the data release background for
            # details.

        self.dm_fields = {# Comoving center of mass of the Subhalo, computed as
                     # the sum of the mass weighted relative coordinates of
                     # all particles/cells in the Subhalo, of all types.
                     'SubhaloCM':[('cm_x','f','Mpc/h',-1),('cm_y','f','Mpc/h',-1),('cm_z','f','Mpc/h',-1)],

                     # Sum of masses of all particles/cells within the radius
                     # of Vmax
                     'SubhaloMassInMaxRad':[('mass_maxr','f','Msun/h',-1)],

                     # Total spin per axis, computed for each as the mass
                     # weighted sum of the relative coordinate times relative
                     # velocity of all member particles/cells.
                     'SubhaloSpin':[('spin_x','f','Mpc/h km/s',-1),('spin_y','f','Mpc/h km/s',-1),('spin_z','f','Mpc/h km/s',-1)],

                     # One-dimensional velocity dispersion of all the member
                     # particles/cells (the 3D dispersion divided by sqrt(3)
                     'SubhaloVelDisp':[('vdisp','f','km/s',0)],

                     # Maximum value of the spherically-averaged rotation
                     # curve.
                     'SubhaloVmax':[('vmax','f','km/s',0)],
                     # Comoving radius of rotation curve maximum (where Vmax
                     # is achieved).
                     'SubhaloVmaxRad':[('r_vmax','f','Mpc/h',-1)]
                     }

        if self.sim[-4:] == 'Dark':
            self.matter_fields = {}
        
        else:
            self.matter_fields = {# Sum of the masses of all blackholes in this subhalo.
                            'SubhaloBHMass':[('bhmass','f','Msun/h',-1)],

                            # Sum of the instantaneous accretion rates Mdot of all
                            # blackholes in this subhalo.
                            'SubhaloBHMdot':[('bhmdot','f','Msun/h / (Gyr/h)',0)],

                            # Mass-weighted average metallicity (Mz/Mtot, where
                            # Z = any element above He) of the gas cells bound to
                            # this Subhalo, but restricted to cells within twice
                            # the stellar half mass radius.
                            'SubhaloGasMetallicity':[('met','f','None',0)],
                            # Same as SubhaloGasMetallicity, but restricted to
                            # cells within the stellar half mass radius.
                            'SubhaloGasMetallicityHalfRad':[('met_hr','f','None',0)],
                            # Same as SubhaloGasMetallicity, but restricted to
                            # cells within the radius of Vmax.
                            'SubhaloGasMetallicityMaxRad':[('met_maxr','f','None',0)],

                            # Total mass of all member particle/cells which are
                            # bound to this Subhalo, separated by type.
                            # Particle/cells bound to subhaloes of this Subhalo are
                            # NOT accounted for. Note: Wind phase cells are counted
                            # as gas (type 0) for SubhaloMassType.
                            'SubhaloMassType':
                                [('m_gas','f','Msun/h',-1),('m_dm','f','Msun/h',-1),('none','f','Msun/h',-1),
                                ('m_tracers','f','Msun/h',-1),('m_stars_wind','f','Msun/h',-1),('m_bh','f','Msun/h',-1)],

                            # Sum of masses of all particles/cells within the
                            # stellar half mass radius.
                            'SubhaloMassInHalfRad':[('mass_hr','f','Msun/h',-1)],
                            # Sum of masses of all particles/cells (split by type)
                            # within the stellar half mass radius.
                            'SubhaloMassInHalfRadType':
                                [('m_gas_hr','f','Msun/h',-1),('m_dm_hr','f','Msun/h',-1),('none_hr','f','Msun/h',-1),
                                ('m_tracers_hr','f','Msun/h',-1),('m_stars_wind_hr','f','Msun/h',-1),
                                ('m_bh_hr','f','Msun/h',-1)],
                            # Sum of masses of all particles/cells (split by type)
                            # within the radius of Vmax
                            'SubhaloMassInMaxRadType':
                                [('m_gas_maxr','f','Msun/h',-1),('m_dm_maxr','f','Msun/h',-1),
                                ('none_maxr','f','Msun/h',-1),('m_tracers_maxr','f','Msun/h',-1),
                                ('m_stars_wind_maxr','f','Msun/h',-1),('m_bh_maxr','f','Msun/h',-1)],
                            # Sum of masses of all particles/cells within twice the
                            # stellar half mass radius.
                            'SubhaloMassInRad':[('mass_2hr','f','Msun/h',-1)],
                            # Sum of masses of all particles/cells (split by type)
                            # within twice the stellar half mass radius.
                            'SubhaloMassInRadType':
                                [('m_gas_2hr','f','Msun/h',-1),('m_dm_2hr','f','Msun/h',-1),('none_2hr','f','Msun/h',-1),
                                ('m_tracers_2hr','f','Msun/h',-1),('m_stars_wind_2hr','f','Msun/h',-1),
                                ('m_bh_2hr','f','Msun/h',-1)],

                            # Sum of the individual star formation rates of all gas
                            # cells in this subhalo.
                            'SubhaloSFR':[('sfr','f','Msun/yr',0)],
                            # Same as SubhaloSFR, but restricted to cells within
                            # the stellar half mass radius.
                            'SubhaloSFRinHalfRad':[('sfr_hr','f','Msun/yr',0)],
                            # Same as SubhaloSFR, but restricted to cells within
                            # the radius of Vmax.
                            'SubhaloSFRinMaxRad':[('sfr_maxr','f','Msun/yr',0)],
                            # Same as SubhaloSFR, but restricted to cells within
                            # twice the stellar half mass radius.
                            'SubhaloSFRinHalfRad':[('sfr_2hr','f','Msun/yr',0)],

                            # Mass-weighted average metallicity (Mz/Mtot, where
                            # Z = any element above He) of the star particles bound
                            # to this Subhalo, but restricted to stars within twice
                            # the stellar half mass radius.
                            'SubhaloStarMetallicity':[('met_stellar','f','None',0)],
                            # Same as SubhaloStarMetallicity, but restricted to
                            # stars within the stellar half mass radius.
                            'SubhaloStarMetallicityHalfRad':[('met_stellar_hr','f','None',0)],
                            # Same as SubhaloStarMetallicity, but restricted to
                            # stars within the radius of Vmax.
                            'SubhaloStarMetallicityMaxRad':[('met_stellar_maxr','f','None',0)],

                            # Eight bands: U, B, V, K, g, r, i, z. Magnitudes based
                            # on the summed-up luminosities of all the stellar
                            # particles of the group. For details on the bands, see
                            # snapshot table for stars.
                            'SubhaloStellarPhotometrics':
                                [('phot_U','f','mag',0),('phot_B','f','mag',0),('phot_V','f','mag',0),
                                ('phot_K','f','mag',0),('phot_g','f','mag',0),('phot_r','f','mag',0),
                                ('phot_i','f','mag',0),('phot_z','f','mag',0)]
                            }

        if self.sim[:3] == 'TNG' and self.sim[-4:] != 'Dark':
            tng_matter_fields ={# Individual abundances: H, He, C, N, O, Ne, Mg, Si,
                         # Fe, total (in this order). Each is the dimensionless
                         # ratio of the total mass in that species divided by
                         # the total gas mass, both restricted to gas cells
                         # within twice the stellar half mass radius. The tenth
                         # entry contains the 'total' of all other (i.e.
                         # untracked) metals.
                         'SubhaloGasMetalFractions':
                            [('metH','f','None',0),('metHe','f','None',0),('metC','f','None',0),
                             ('metN','f','None',0),('metO','f','None',0),('metNe','f','None',0),
                             ('metMg','f','None',0),('metSi','f','None',0),('metFe','f','None',0),
                             ('metOther','f','None',0)],
                         # Same as SubhaloGasMetalFractions, but restricted to
                         # cells within the stellar half mass radius.
                         'SubhaloGasMetalFractionsHalfRad':
                            [('metH_hr','f','None',0),('metHe_hr','f','None',0),('metC_hr','f','None',0),
                             ('metN_hr','f','None',0),('metO_hr','f','None',0),('metNe_hr','f','None',0),
                             ('metMg_hr','f','None',0),('metSi_hr','f','None',0),('metFe_hr','f','None',0),
                             ('metOther_hr','f','None',0)],
                         # Same as SubhaloGasMetalFractions, but restricted to
                         # cells within the radius of Vmax
                         'SubhaloGasMetalFractionsMaxRad':
                            [('metH_maxr','f','None',0),('metHe_maxr','f','None',0),('metC_maxr','f','None',0),
                             ('metN_maxr','f','None',0),('metO_maxr','f','None',0),('metNe_maxr','f','None',0),
                             ('metMg_maxr','f','None',0),('metSi_maxr','f','None',0),('metFe_maxr','f','None',0),
                             ('metOther_maxr','f','None',0)]
                         }
            self.matter_fields = {**self.matter_fields, **tng_matter_fields}


        # Check whether snapshots have been downloaded
        not_downloaded = []
        for i in self.snaps:
            file_path = self.path+'/raw/groups_{:03d}'.format(i)
            if not os.path.exists(file_path):
                not_downloaded.append(i)
        if len(not_downloaded) > 0:
            warnings.warn("No data exists for snapshots {} - run .download".format(not_downloaded))

        # File naming conventions for the raw simulation downloads:
        if self.sim[:3] == 'Ill':
            self.raw_fname = 'groups_'
        if self.sim[:3] == 'TNG':
            self.raw_fname = 'fof_subhalo_tab_'

        # Function to load the data in a format we want:
        def loader(path, snapshot, fields):
            subhalos = idh._loadSubhalos(self.path+'/raw/', snapshot, fields=self.all_fields.keys())
            n_halos = subhalos.pop('count')
            return subhalos, n_halos
        self.loader = loader

    def download(self, redownload=False, APIkey=_IllustrisAPIKey):

        if not os.path.exists(self.path+'/raw'):
            os.mkdir(self.path+'/raw')

        # Add a check for already downloaded files
        if redownload:
            self.download_snaps = np.copy(self.snaps)
        else:
            self.download_snaps = []
            for i in self.snaps:
                file_path = self.path+'/raw/groups_{:03d}'.format(i)
                if os.path.exists(file_path+'/'+self.raw_fname+'{:03d}.0.hdf5'.format(i)):
                    warnings.warn("Skipping snapshot {} as it appears to exist already".format(i))
                else:
                    self.download_snaps.append(i)

        # Download each snap
        for i in self.download_snaps:

            # Create destination for the files:
            file_path = self.path+'/raw/groups_{:03d}'.format(i)
            if not os.path.exists(file_path):
                os.mkdir(file_path)

            # Construcg wget command
            wget_cmd  = 'wget -nd -nc -nv -e robots=off '
            wget_cmd += '-l 1 -r -A hdf5 --content-disposition '
            wget_cmd += '--header="API-Key: {}" '.format(APIkey)
            wget_cmd += '"http://www.tng-project.org/api/{}'.format(self.sim)
            wget_cmd += '/files/groupcat-{}/?format=api" '.format(i)
            wget_cmd += '-P {}'.format(file_path)

            # Download the data and check that it all arrives, retry if it doesn't
            print("\nDownloading Snapshot {}".format(i))
            check_pass = False
            try:
                while not check_pass:
                    os.system(wget_cmd)

                    # If there's no meta-data saved we can't check if the right
                    # files have arrived
                    if not self.metadata:
                        check_pass = True
                    else:
                        check = [os.path.exists(file_path
                                                + '/' + self.raw_fname
                                                + '{:03d}.{}.hdf5'.format(i,j)
                                                )
                                 for j in range(self.metadata['groupcat_number_files'])]
                        check_pass = np.all(check)

                    if not check_pass:
                        print("Not all files downloaded correctly, retrying")
            except:
                print('Interupted!!!')
    
    
##########################################################################
    
    def download_meta_camels(self, redownload=False, \
                             sims_folder_path="/global/cfs/cdirs/des/shubh/camels/~camels/FOF_Subfind/IllustrisTNG/"):
        
        # Check that metadata doesn't already exist
        if not redownload:
            if os.path.exists(self.meta_path):
                warnings.warn("Metadata appears to exist already")
                return
        
        camels_sim_types = ["CV_", "LH_", "1P_", "EX_"]
        for cst in camels_sim_types:
            if cst in self.path:
                full_sim_path = os.path.join(sims_folder_path, self.path[self.path.index(cst):])
                break
        
        full_sim_path = "/global/cfs/cdirs/des/shubh/camels/~camels/FOF_Subfind/IllustrisTNG/CV_3/"
        
        fullsim_file = h5py.File(os.path.join(full_sim_path, "fof_subhalo_tab_000.hdf5"), 'r')
        fullsim_header = fullsim_file['Header'].attrs
        
        self.metadata = {'name':self.path[self.path.index(cst):],
                         'box_edge':fullsim_header['BoxSize']/1000,
                         'number_snaps':34,
                         'cosmo_name':self.path[self.path.index(cst):],
                         'cosmo_omega_matter':fullsim_header['Omega0'],
                         'cosmo_omega_lambda':fullsim_header['OmegaLambda'],
                         'cosmo_omega_baryon':0.049,
                         'cosmo_h':0.6711
                         }
        self.box_edge = self.metadata['box_edge']
        self.h = self.metadata['cosmo_h']
        
        fullsim_file.close()
        
        snap_meta_classes = []
        numbers = []
        for fullsim_filename in os.listdir(full_sim_path):
            if bool(re.match(r'fof_subhalo_tab_\d{3}\.hdf5', fullsim_filename)):
                with h5py.File(os.path.join(full_sim_path, fullsim_filename), 'r') as fullsim_file:
                    fullsim_header = fullsim_file['Header'].attrs
                    snap_meta_classes.append(snapshot(int(fullsim_filename[16:19]),fullsim_header['Redshift'],self.metadata))
                    numbers.append(int(fullsim_filename[16:19]))
        # snap_meta_classes = [snap_meta_classes[i] for i in np.argsort(numbers)]
        
        order = np.argsort(numbers)
        snap_meta_classes = [snap_meta_classes[i] for i in order if numbers[i] in self.snaps]
        
        snap_meta_classes[0].dif_higherz_snap('max')
        snap_meta_classes[-1].dif_lowerz_snap(0)
        for i in range(len(snap_meta_classes)-1):
            snap_meta_classes[i].dif_lowerz_snap(snap_meta_classes[i+1])
            snap_meta_classes[i+1].dif_higherz_snap(snap_meta_classes[i])

        snap_meta_dtype = [('index','i'),
                           ('redshift','f'),
                           ('redshift_min','f'),('redshift_max','f'),
                           ('time_start','f'),('time_end','f'),
                           ('distance_min','f'),('distance_max','f'),
                           ('transverse_distance_min','f'),('transverse_distance_max','f')]
        snap_meta = np.zeros(len(snap_meta_classes),
                             dtype = snap_meta_dtype)

        for i in range(len(snap_meta_classes)):
            snap_meta[i]['index'] = snap_meta_classes[i].index
            snap_meta[i]['redshift'] = snap_meta_classes[i].redshift
            snap_meta[i]['redshift_min'] = snap_meta_classes[i].redshift_min
            snap_meta[i]['redshift_max'] = snap_meta_classes[i].redshift_max
            snap_meta[i]['time_start'] = snap_meta_classes[i].time_start * self.h
            snap_meta[i]['time_end'] = snap_meta_classes[i].time_end * self.h
            snap_meta[i]['distance_min'] = snap_meta_classes[i].distance_min * self.h
            snap_meta[i]['distance_max'] = snap_meta_classes[i].distance_max * self.h
            snap_meta[i]['transverse_distance_min'] = snap_meta_classes[i].transverse_distance_min * self.h
            snap_meta[i]['transverse_distance_max'] = snap_meta_classes[i].transverse_distance_max * self.h
        self.snap_meta = snap_meta

        np.save(self.meta_path,self.metadata)
        np.save(self.snap_meta_path,self.snap_meta)

        
        
        

##########################################################################

        
    def download_meta(self, redownload=False, APIkey=_IllustrisAPIKey):

        # Check that metadata doesn't already exist
        if not redownload:
            if os.path.exists(self.meta_path):
                warnings.warn("Metadata appears to exist already")
                return

        # Get the stuff via HTML GET request (code adapted from Illustris project)
        base_url = 'http://www.tng-project.org/api/{}/'.format(self.sim)
        response = requests.get(base_url, params=None,
                                headers={'api-key':_IllustrisAPIKey})

        # raise exception if response code is not HTTP SUCCESS (200)
        response.raise_for_status()

        # Extract the stuff we want
        sim_meta_raw = response.json() # parse json responses automatically

        self.metadata = {'name':sim_meta_raw['name'],
                         'box_edge':sim_meta_raw['boxsize']/1000,
                         'number_snaps':sim_meta_raw['num_snapshots'],
                         'cosmo_name':sim_meta_raw['cosmology'],
                         'cosmo_omega_matter':sim_meta_raw['omega_0'],
                         'cosmo_omega_lambda':sim_meta_raw['omega_L'],
                         'cosmo_omega_baryon':sim_meta_raw['omega_B'],
                         'cosmo_h':sim_meta_raw['hubble'],
                         'groupcat_number_files':sim_meta_raw['num_files_groupcat']
                         }
        self.box_edge = self.metadata['box_edge']
        self.h = self.metadata['cosmo_h']

        # Now get stuff for individual snapshots:
        base_url = 'http://www.tng-project.org/api/{}/snapshots'.format(self.sim)
        response = requests.get(base_url, params=None,
                                headers={'api-key':_IllustrisAPIKey})

        # raise exception if response code is not HTTP SUCCESS (200)
        response.raise_for_status()

        # Extract the stuff we want - it's a list of dictionaries for each snap
        snap_meta_raw = response.json() # parse json responses automatically

        snap_meta_classes = []
        numbers = []
        for i in snap_meta_raw:
            snap_meta_classes.append(snapshot(i['number'],i['redshift'],self.metadata))
            numbers.append(i['number'])          
        snap_meta_classes = [snap_meta_classes[i] for i in np.argsort(numbers)]
           
        order = np.argsort(numbers)
        snap_meta_classes = [snap_meta_classes[i] for i in order if numbers[i] in self.snaps]

        snap_meta_classes[0].dif_higherz_snap('max')
        snap_meta_classes[-1].dif_lowerz_snap(0)
        for i in range(len(snap_meta_classes)-1):
            snap_meta_classes[i].dif_lowerz_snap(snap_meta_classes[i+1])
            snap_meta_classes[i+1].dif_higherz_snap(snap_meta_classes[i])

        snap_meta_dtype = [('index','i'),
                           ('redshift','f'),
                           ('redshift_min','f'),('redshift_max','f'),
                           ('time_start','f'),('time_end','f'),
                           ('distance_min','f'),('distance_max','f'),
                           ('transverse_distance_min','f'),('transverse_distance_max','f')]
        snap_meta = np.zeros(len(snap_meta_classes),
                             dtype = snap_meta_dtype)

        for i in range(len(snap_meta_classes)):
            snap_meta[i]['index'] = snap_meta_classes[i].index
            snap_meta[i]['redshift'] = snap_meta_classes[i].redshift
            snap_meta[i]['redshift_min'] = snap_meta_classes[i].redshift_min
            snap_meta[i]['redshift_max'] = snap_meta_classes[i].redshift_max
            snap_meta[i]['time_start'] = snap_meta_classes[i].time_start * self.h
            snap_meta[i]['time_end'] = snap_meta_classes[i].time_end * self.h
            snap_meta[i]['distance_min'] = snap_meta_classes[i].distance_min * self.h
            snap_meta[i]['distance_max'] = snap_meta_classes[i].distance_max * self.h
            snap_meta[i]['transverse_distance_min'] = snap_meta_classes[i].transverse_distance_min * self.h
            snap_meta[i]['transverse_distance_max'] = snap_meta_classes[i].transverse_distance_max * self.h
        self.snap_meta = snap_meta

        np.save(self.meta_path,self.metadata)
        np.save(self.snap_meta_path,self.snap_meta)
