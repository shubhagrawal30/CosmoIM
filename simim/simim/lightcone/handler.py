import os
import warnings

import h5py
import numpy as np
from astropy.cosmology import FlatLambdaCDM, z_at_value
import astropy.units as u

from matplotlib import animation

from simim import _paths
from simim._handlers import handler
from simim._mplsetup import *

"""TO DO LIST:
1. Write tests and debug
2. Compute statistics over a set of properties (mean, power spectrum, distribution functions)
"""

class lightcone(handler):
    """Class to handle I/O and basic analysis for light cone hdf5 files"""

    def __init__(self,sim,name,number):
        """Initialize lightcone access"""

        # Handle file path stuff
        paths = _paths._paths()
        if not sim in paths.lightcones.keys():
            raise ValueError("Lightcones for simulation {} not found. Try generating them or updating the path".format(sim))
        path = paths.lightcones[sim]

        if not os.path.exists(path+'/'+name):
            raise ValueError("Lightcones named '{}' not found. Try generating them or updating the path".format(name))
        path = path + '/' + name

        if not os.path.exists(path+'/lc_{:04d}.hdf5'.format(number)):
            raise ValueError("Lightcone number '{}' not found.".format(number))
        path = path + '/lc_{:04d}.hdf5'.format(number)

        super().__init__(path=path,objectname='Lightcone',groupname='Lightcone Full')

        # Get the metadata
        self.metadata = {}
        with h5py.File(self.path,'r') as file:
            for key in file.attrs.keys():
                self.metadata[key] = file.attrs[key]

        # Make the cosmology
        self.cosmo = FlatLambdaCDM(H0=100*self.metadata['cosmo_h']*u.km/u.s/u.Mpc,
                                   Om0=self.metadata['cosmo_omega_matter'],
                                   Ob0=self.metadata['cosmo_omega_baryon'],
                                   Tcmb0=2.7255*u.K)
        self.open_angle = self.metadata['open angle']
        self.shape = self.metadata['shape']
        self.aspect_ratio = self.metadata['aspect ratio']
        self.extra_props['cosmo'] = self.cosmo
        self.extra_props['open_angle'] = self.open_angle
        self.extra_props['shape'] = self.shape
        self.extra_props['aspect_ratio'] = self.aspect_ratio

    def volume(self,redshift_min=None,redshift_max=None,shape=None,open_angle=None,aspect_ratio=None,in_h_units=False):
        """Compute the comoving volume of the light cone

        Parameters
        ----------
        redshift_min, redshift_max : float, optional
            The minimum/maximum redshift to consider - the minimum/maximum
            redshift of the lightcone by default
        open_angle : float, optional
            The opening angle of the lightcone - by default matches the
            lightcone
        aspect_ratio : float, optional
            The aspect ratio of the box sides (for square lightcones) -
            by default matches the lightcone
        shape : 'box', 'circle', optional
            The shape of the box - by default matches the lightcone
        in_h_units : bool (default is False)
            If True, values will be returned in units including little h.
            If False, little h dependence will be removed.


        Returns
        -------
        volume : float
            The volume of the lightcone in comoving (Mpc/h)^3
        """

        # Set defaults
        if redshift_min == None:
            redshift_min=self.metadata['minimum redshift']
        if redshift_max == None:
            redshift_max=self.metadata['minimum redshift']
        if shape == None:
            shape=self.metadata['shape']
        if open_angle == None:
            open_angle=self.metadata['open angle']
        if aspect_ratio == None:
            aspect_ratio=self.metadata['aspect ratio']

        # Check everything
        if redshift_min < self.metadata['minimum redshift']:
            raise ValueError("redshift_min lower than minimum of lightcone")
        if redshift_max > self.metadata['maximum redshift']:
            raise ValueError("redshift_max higher than maximum of lightcone")

        if shape not in ['box','circle']:
            raise ValueError("Shape not recognized")

        if shape == self.metadata['shape'] or self.metadata['shape'] == 'box':
            if open_angle > self.metadata['open angle']:
                raise ValueError("openangle larger than lightcone")
            if open_angle*aspect_ratio > self.metadata['open angle']*self.metadata['aspect ratio']:
                raise ValueError("secondary angle (openangle * aspect) larger than lightcone")

        if shape == 'box' and self.metadata['shape'] == 'circle':
            radius2 = open_angle**2 + (open_angle*aspect_ratio)**2
            if radius2 > self.metadata['open angle']:
                raise ValueError("specified box does not fit in circular lightcone")

        # Figure out the volume
        dmin = self.cosmo.comoving_distance(redshift_min).value
        dmax = self.cosmo.comoving_distance(redshift_max).value
        length = dmax-dmin

        dtmin = self.cosmo.comoving_transverse_distance(redshift_min).value
        dtmax = self.cosmo.comoving_transverse_distance(redshift_max).value

        if shape == 'box':
            area0 = open_angle * open_angle * aspect_ratio
        elif shape == 'circle':
            area0 = np.pi * (open_angle/2)**2

        areamin = area0 * dtmin**2
        areamax = area0 * dtmax**2

        volume = (areamin+areamax)/2 * length
        if in_h_units:
            volume *= self.h**3

        return volume

    def eval_stat_evo(self, redshift_bins, stat_function, kwargs, kw_remap={}, other_kws={}, zmin_kw=False, zmax_kw=False, volume_kw=False, give_args_in_h_units=False):
        """Compute the evolution of a statistic over a specified set of
        redshift bins
        """

        redshift_bins = np.sort(redshift_bins)
        
        inds_active_save = np.copy(self.inds_active)

        vals = []
        for bin in range(len(redshift_bins)-1):
            self.set_property_range('redshift',
                                    pmin=redshift_bins[bin],
                                    pmax=redshift_bins[bin+1],
                                    reset=True)

            if zmin_kw:
                other_kws['zmin'] = redshift_bins[bin]
            if zmax_kw:
                other_kws['zmax'] = redshift_bins[bin+1]
            if volume_kw:
                other_kws['volume'] = self.volume(redshift_min = redshift_bins[bin],
                                                    redshift_max = redshift_bins[bin+1],
                                                    in_h_units=give_args_in_h_units)

            vals.append(self.eval_stat(stat_function,
                                       kwargs=kwargs,
                                       kw_remap=kw_remap,
                                       other_kws=other_kws,
                                       use_all_inds=False,
                                       give_args_in_h_units=give_args_in_h_units))

            other_kws.pop('zmin',None)
            other_kws.pop('zmax',None)

        # Reset active inds
        self.inds_active = np.copy(inds_active_save)

        return redshift_bins, vals

    def animate(self, save=None, use_all_inds=False, colorpropname='mass',colorscale='log', sizepropname='mass',sizescale='log',in_h_units=False):
        """Make an animation of the light cone

        Parameters
        ----------
        use_all_inds : bool, optional
            If True values will be assigned for all halos, otherwise only
            active halos will be evaluated, and others will be assigned nan.
            Default is False.
        save : str, optional
            If specified, the plot will be saved to the given location
        colorpropname : str
            Name of the property used to determine marker color for each
            galaxy, default is 'mass'
        colorpropscale : 'log' or 'linear'
            Determines how the colorbar will be applied. Default is 'log'
        in_h_units : bool (default is False)
            If True, values will be plotted in units including little h.
            If False, little h dependence will be removed.

        Returns
        -------
        None
        """

        x = self.return_property('pos_x',use_all_inds=use_all_inds,in_h_units=in_h_units)
        y = self.return_property('pos_y',use_all_inds=use_all_inds,in_h_units=in_h_units)
        z = self.return_property('pos_z',use_all_inds=use_all_inds,in_h_units=in_h_units)

        ra = self.return_property('ra',use_all_inds=use_all_inds,in_h_units=in_h_units)
        dec = self.return_property('dec',use_all_inds=use_all_inds,in_h_units=in_h_units)
        redshift = self.return_property('redshift',use_all_inds=use_all_inds,in_h_units=in_h_units)

        # Get limits
        phys_lim_xy = self.cosmo.comoving_transverse_distance(self.metadata['maximum redshift']).value * self.h * self.metadata['open angle']
        phys_lim_zmax = self.cosmo.comoving_distance(self.metadata['maximum redshift']).value * self.h
        phys_lim_zmin = self.cosmo.comoving_distance(self.metadata['minimum redshift']).value * self.h
        obs_lim_xy = self.metadata['open angle'] * 180/np.pi*60
        obs_lim_zmax = self.metadata['maximum redshift']
        obs_lim_zmin = self.metadata['minimum redshift']

        # Values along a circle, outline type
        outline = self.metadata['shape']
        edgepaternx = np.array([-1,1,1,-1,-1])
        edgepaterny = np.array([-1,-1,1,1,-1])

        theta = np.linspace(0,2*np.pi,1000)
        theta = np.concatenate((theta,[2*np.pi]))

        # Set up color map
        colorprop = self.return_property(colorpropname,use_all_inds=use_all_inds,in_h_units=in_h_units)
        if colorscale == 'log':
            colorprop = np.log10(colorprop)
        elif colorscale != 'linear':
            raise ValueError("colorscale not recognized")
        colormin = np.amin(colorprop)
        colorrange = np.ptp(colorprop)
        colors = cmap((colorprop-colormin)/colorrange)

        sm = plt.cm.ScalarMappable(cmap=cmap)
        sm.set_array([colormin,(colormin+colorrange)])

        # Set up sizes
        sizeprop = self.return_property(sizepropname,use_all_inds=use_all_inds,in_h_units=in_h_units)
        if sizescale == 'log':
            sizeprop = np.log10(sizeprop)
        elif sizescale != 'linear':
            raise ValueError("sizescale not recognized")
        sizemin = np.amin(sizeprop)
        sizerange = np.ptp(sizeprop)
        sizes = ((sizeprop-sizemin)/sizerange + .2) * 5

        # set up plots
        figure = plt.figure(figsize=(8,8))
        title = plt.suptitle('')
        # figure.legend(handles=[p0,p1,p2,p3,p4,p5],loc=8,bbox_to_anchor=(.5,0),ncol=3,markerscale=.5,fontsize='small')

        # cone plot
        plot_cone = plt.subplot(212)
        plot_cone.set_title('Light Cone')

        cone = plot_cone.scatter(z, x, s=1, c=colors)
        box, = plot_cone.plot([], [], color='#ff7f0e')

        if colorscale == 'log':
            figure.colorbar(sm,label='log '+colorpropname)
        else:
            figure.colorbar(sm,label=colorpropname)


        # physical cross section plot
        plot_xy = plt.subplot(222)
        plot_xy.set_title('Physical Cross-section (Mpc)')
        plot_xy.axis('equal')
        plot_xy.set(xlim = (-.55*phys_lim_xy,.55*phys_lim_xy), ylim = (-.55*phys_lim_xy,.55*phys_lim_xy))

        xy_scatter = plot_xy.scatter(x, y, s=np.zeros(len(x)), c=colors)
        xy_outline, = plot_xy.plot([], [], color='#ff7f0e')

        ### angular cross section plot
        plot_radec = plt.subplot(221)
        plot_radec.set_title('Angular Cross-section (arcmin)')
        plot_radec.axis('equal')
        plot_radec.set(xlim = (-.55*obs_lim_xy,.55*obs_lim_xy), ylim = (-.55*obs_lim_xy,.55*obs_lim_xy))

        radec_scatter = plot_radec.scatter(ra*180/np.pi*60, dec*180/np.pi*60, s=np.zeros(len(x)), c=colors)
        if outline == 'circle':
            radec_outline, = plot_radec.plot(obs_lim_xy/2*np.cos(theta), obs_lim_xy/2*np.sin(theta), color='#ff7f0e')
        else:
            radec_outline, = plot_radec.plot(obs_lim_xy/2*edgepaternx, obs_lim_xy/2*edgepaterny, color='#ff7f0e')


        # Initialization for animation_init
        def animation_init():
            title.set_text('')

            box.set_data([], [])

            # cone.set_sizes(np.zeros(len(x)))

            xy_outline.set_data([], [])

            radec_scatter.set_sizes(np.zeros(len(x)))

        step = 50
        nsteps = np.ceil((phys_lim_zmax-phys_lim_zmin)/step).astype('int')
        step = np.ceil((phys_lim_zmax-phys_lim_zmin)/nsteps).astype('int')

        def animate(i):
            #title
            redshift = z_at_value(self.cosmo.comoving_distance, (phys_lim_zmin+step*(i+.5))*u.Mpc)
            title.set_text('redshift = {:.2f}'.format(redshift))

            #animate p1
            box.set_data([phys_lim_zmin+step*i,phys_lim_zmin+step*i,phys_lim_zmin+step*(i+1),phys_lim_zmin+step*(i+1),phys_lim_zmin+step*i],
                         [-phys_lim_xy/2,phys_lim_xy/2,phys_lim_xy/2,-phys_lim_xy/2,-phys_lim_xy/2])

            #select halos to show in p2 and p3
            inds = np.where((z > phys_lim_zmin + step*i) & (z < phys_lim_zmin + step*(i+1)))
            new_sizes = np.zeros(len(z))
            new_sizes[inds] = sizes[inds]

            #animate p2
            xy_scatter.set_sizes(new_sizes)

            #animate p3
            radec_scatter.set_sizes(new_sizes)

            r = self.cosmo.comoving_transverse_distance(redshift) * self.metadata['open angle']/2
            if outline == 'circle':
                xy_outline.set_data(r*np.cos(theta),r*np.sin(theta))
            else:
                xy_outline.set_data(r*edgepaternx, r*edgepaterny)

        anim = animation.FuncAnimation(figure, animate, init_func=animation_init, frames=nsteps, interval=100, blit=False)
        if save != None:
            anim.save(save+'.mp4')

        plt.show()
