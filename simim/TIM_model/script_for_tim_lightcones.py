import numpy as np
import matplotlib.pyplot as plt
import simim.lightcone as lc
from simim.lineprops.lineprops import prop_behroozi_sfr, prop_behroozi_mass
from simim.lineprops.lineprops import prop_spinoglio_fir, prop_zhao_nii
from simim.map.gridder import gridder,_grid,grid_from_axes_and_function

build_lc_large = True 
build_lc_small = False 
build_cube_large = True
build_cube_small = False

# Set the simulation to use
simname = 'TNG300-1'

# RNG instances
rng1 = np.random.default_rng(327903736)
rng2 = np.random.default_rng(rng1.integers(1000000000))

# Cube sizes
size_large = 2 # sq deg
size_small = 0.15 # sq deg
number_of_lc = 1

# Build the light cones
if build_lc_large or build_lc_small:
    # Line scatter
    line_scatter = 0.4 # based on DeLooze fit for whole CII sample

    # Setup of light cone
    mode = 'box'
    openangle_large = np.sqrt(size_large)
    openangle_small = np.sqrt(size_small)

    aspect = 1
    redshift_min = 0
    redshift_max = 4.0

    minimum_mass = 1e9

if build_lc_large:

    # For Behroozi et al. SFRs and stellar masses:
    kws_behroozi = {'rng':rng1}

    # For spinoglio et al:
    kws_spinoglio_cii = {'line':'[CII]','sig_scatter':line_scatter,'rng':rng1}
    kws_spinoglio_nii = {'line':'[NII]','sig_scatter':line_scatter,'rng':rng1}
    kws_spinoglio_oiii = {'line':'[OIII]88um','sig_scatter':line_scatter,'rng':rng1}

    # For Zhao et al.
    kws_zhao_nii = {'sig_scatter':line_scatter,'rng':rng1}
    
    # Generate the light cones
    maker = lc.make.lightcone(simname,'TIM-large',openangle=openangle_large,aspect=aspect,redshift_min=redshift_min,redshift_max=redshift_max,minimum_mass=minimum_mass,mode=mode)
    maker.build_lightcones(number_of_lc,rng=rng1)

    # Apply each property to the light cone
    props = [prop_behroozi_sfr,prop_behroozi_mass,prop_spinoglio_fir,prop_spinoglio_fir,prop_zhao_nii,prop_spinoglio_fir]
    names = ['sfr_behroozi','mstar_behroozi','LCII','LNII122','LNII205','LOIII',]
    kws = [kws_behroozi,kws_behroozi,kws_spinoglio_cii,kws_spinoglio_nii,kws_zhao_nii,kws_spinoglio_oiii]

    for i in range(number_of_lc):
        handler = lc.handler.lightcone(simname,'TIM-large',i)
        for prop,name,kw in zip(props,names,kws):
            handler.make_property(prop, rename=name,other_kws=kw,kw_remap={'sfr':'sfr_behroozi'})
        handler.write_property(*names)

if build_lc_small:
    # For Behroozi et al. SFRs and stellar masses:
    kws_behroozi = {'rng':rng2}

    # For spinoglio et al:
    kws_spinoglio_cii = {'line':'[CII]','sig_scatter':line_scatter,'rng':rng2}
    kws_spinoglio_nii = {'line':'[NII]','sig_scatter':line_scatter,'rng':rng2}
    kws_spinoglio_oiii = {'line':'[OIII]88um','sig_scatter':line_scatter,'rng':rng2}

    # For Zhao et al.
    kws_zhao_nii = {'sig_scatter':line_scatter,'rng':rng2}
    
    # Generate the light cones
    maker = lc.make.lightcone(simname,'TIM-small',openangle=openangle_small,aspect=aspect,redshift_min=redshift_min,redshift_max=redshift_max,minimum_mass=minimum_mass,mode=mode)
    maker.build_lightcones(number_of_lc,rng=rng2)

    # Apply each property to the light cone
    props = [prop_behroozi_sfr,prop_behroozi_mass,prop_spinoglio_fir,prop_spinoglio_fir,prop_zhao_nii,prop_spinoglio_fir]
    names = ['sfr_behroozi','mstar_behroozi','LCII','LNII122','LNII205','LOIII',]
    kws = [kws_behroozi,kws_behroozi,kws_spinoglio_cii,kws_spinoglio_nii,kws_zhao_nii,kws_spinoglio_oiii]

    for i in range(number_of_lc):
        handler = lc.handler.lightcone(simname,'TIM-small',i)
        for prop,name,kw in zip(props,names,kws):
            handler.make_property(prop, rename=name,other_kws=kw,kw_remap={'sfr':'sfr_behroozi'})
        handler.write_property(*names)


# Create simulated data cubes
if build_cube_large or build_cube_small:

    # Path to save cubes
    cube_path = '/data/rpkeenan/simim_resources/cubes/TIM/' 

    # Properties of our cube:
    df = 0.5*1e9 # Frequency resolution, Hz - puts about 2.5 cells across the channel widht of our longest wl detector
    da = 6.0/3600/180*np.pi # angular resolution, rad - puts at least 2.5 cells across the beam size

    fmin = 2.998e8 / 425e-6 # Frequency limits in Hz
    fmax = 2.998e8 / 215e-6
    fmid = (fmin+fmax)/2

    xyrange_large = np.sqrt(size_large) * np.pi/180 # angular extent of cube in rad
    xyrange_small = np.sqrt(size_small) * np.pi/180

    # Lines to grid
    lines = [{'name':'CII','prop':'LCII','freq':1900.53690000e9},
             {'name':'NII122','prop':'LNII122','freq':2459.38010085e9},
             {'name':'NII205','prop':'LNII205','freq':1461.13140620e9},
             {'name':'OIII','prop':'LOIII','freq':3393.00624400e9},
             ]

    # TIM PSF for convolution with maps
    def tim_psf(dx,dy,freq):
        """dx,dy in radians, freq in Hz"""

        x = dx.reshape(-1,1,1)
        y = dy.reshape(1,-1,1)
        fwhm = 1.2*(2.998e8/freq.reshape(1,1,-1))/2

        return np.exp(-4*np.log(2)*(x**2 + y**2)/fwhm**2)
        

if build_cube_large:
    for i in range(number_of_lc):
        handler = lc.handler.lightcone(simname,'TIM-large',i)

        # Make an empty grid to contain the sum
        grid_sum = _grid(n_properties=1,
                           center_point=[0,0,fmid],
                           side_length=[xyrange_large,xyrange_large,fmax-fmin],
                           pixel_size=[da,da,df],
                           axunits=['rad','rad','Hz'],
                           gridunits='Jy/str')

        grid_sum.init_grid()
        
        # Make the PSF
        ax1 = (np.arange(0,41)-20) * grid_sum.pixel_size[0]
        ax2 = (np.arange(0,41)-20) * grid_sum.pixel_size[0]
        ax3 = grid_sum.axes[2]

        tim_psf_kernel = grid_from_axes_and_function(tim_psf,ax1,ax2,ax3)
        # Renormalize the kernel such that the total flux is preserved
        tim_psf_kernel.grid = tim_psf_kernel.grid / np.sum(tim_psf_kernel.grid,axis=(0,1)).reshape(1,1,-1,1)

        # Iterate over lines
        for line in lines:
            # Restrict to relevant redshift range for faster computation
            zmax = line['freq']/fmin - 1
            zmin = line['freq']/fmax - 1

            handler.set_property_range('redshift',zmin,zmax)

            # Collect galaxy poistions
            z = handler.return_property('redshift',use_all_inds=False)
            dl = handler.cosmo.luminosity_distance(z).value
            f = line['freq']/(1+z)

            x = handler.return_property('ra',use_all_inds=False)
            y = handler.return_property('dec',use_all_inds=False)

            positions = np.array([x,y,f]).T

            # Determine intensity of each galaxy
            luminosity = handler.return_property(line['prop'],use_all_inds=False)
            intensity = 1e26 * (luminosity*3.828e26) / (4*np.pi*(dl*3.0857e22)**2) / (df * da**2) # Intensity contribution to a cell in Jy/str

            # Make and save grid
            grid = gridder(positions,intensity,
                           center_point=[0,0,fmid],
                           side_length=[xyrange_large,xyrange_large,fmax-fmin],
                           pixel_size=[da,da,df],
                           axunits=['rad','rad','Hz'],
                           gridunits='Jy/str')
            grid.save(cube_path+'large_cube_{}_{}_full_res.npz'.format(i,line['name']))
            
            # Add to sum grid
            grid_sum.grid += grid.grid
            grid_sum.n_objects += grid.n_objects

            # Apply TIM PSF and save
            grid.convolve(tim_psf_kernel,ax=[0,1],in_place=True,pad=20)
            grid.save(cube_path+'large_cube_{}_{}_tim_res.npz'.format(i,line['name']))

        # Save summed grid
        grid_sum.save(cube_path+'large_cube_{}_full_res.npz'.format(i))
        
        # Apply TIM PSF and save
        grid_sum.convolve(tim_psf_kernel,ax=[0,1],in_place=True,pad=20)
        grid_sum.save(cube_path+'large_cube_{}_tim_res.npz'.format(i))

if build_cube_small:
    for i in range(number_of_lc):
        handler = lc.handler.lightcone(simname,'TIM-small',i)

        # Make an empty grid to contain the sum
        grid_sum = _grid(n_properties=1,center_point=[0,0,fmid],
                         side_length=[xyrange_small,xyrange_small,fmax-fmin],
                         pixel_size=[da,da,df],
                         axunits=['rad','rad','Hz'],
                         gridunits='Jy/str')
       
        grid_sum.init_grid()
 
        # Make the PSF
        ax1 = (np.arange(0,41)-20) * grid_sum.pixel_size[0]
        ax2 = (np.arange(0,41)-20) * grid_sum.pixel_size[0]
        ax3 = grid_sum.axes[2]

        tim_psf_kernel = grid_from_axes_and_function(tim_psf,ax1,ax2,ax3)
        # Renormalize the kernel such that the total flux is preserved
        tim_psf_kernel.grid = tim_psf_kernel.grid / np.sum(tim_psf_kernel.grid,axis=(0,1)).reshape(1,1,-1,1)

        # Iterate over lines
        for line in lines:
            # Restrict to relevant redshift range for faster computation
            zmax = line['freq']/fmin - 1
            zmin = line['freq']/fmax - 1

            handler.set_property_range('redshift',zmin,zmax)

            # Collect galaxy poistions
            z = handler.return_property('redshift',use_all_inds=False)
            dl = handler.cosmo.luminosity_distance(z).value
            f = line['freq']/(1+z)

            x = handler.return_property('ra',use_all_inds=False)
            y = handler.return_property('dec',use_all_inds=False)

            positions = np.array([x,y,f]).T

            # Determine intensity of each galaxy
            luminosity = handler.return_property(line['prop'],use_all_inds=False)
            intensity = 1e26 * (luminosity*3.828e26) / (4*np.pi*(dl*3.0857e22)**2) / (df * da**2) # Intensity contribution to a cell in Jy/str

            # Make and save grid
            grid = gridder(positions,intensity,
                           center_point=[0,0,fmid],
                           side_length=[xyrange_small,xyrange_small,fmax-fmin],
                           pixel_size=[da,da,df],
                           axunits=['rad','rad','Hz'],
                           gridunits='Jy/str')
            grid.save(cube_path+'small_cube_{}_{}_full_res.npz'.format(i,line['name']))
            
            # Add to sum grid
            grid_sum.grid += grid.grid
            grid_sum.n_objects += grid.n_objects

            # Apply TIM PSF and save
            grid.convolve(tim_psf_kernel,ax=[0,1],in_place=True,pad=20)
            grid.save(cube_path+'small_cube_{}_{}_tim_res.npz'.format(i,line['name']))

        # Save summed grid
        grid_sum.save(cube_path+'small_cube_{}_full_res.npz'.format(i))
        
        # Apply TIM PSF and save
        grid_sum.convolve(tim_psf_kernel,ax=[0,1],in_place=True,pad=20)
        grid_sum.save(cube_path+'small_cube_{}_tim_res.npz'.format(i))
