import os
import warnings

import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d

from simim import _paths
from simim.lineprops.log10normal import log10normal
from simim._mplsetup import *
import simim.siminterface as sim

class behroozi13_base():
    def __init__(self):

        paths = _paths._paths()

        if 'behroozi13' not in paths.sfrs.keys():
            paths._newsfrpath('behroozi13')

        self.path = paths.sfrs['behroozi13']
        if not os.path.exists(self.path+'/sfr.npy'):
            self._make_spline_files()

        # Load spline stuff
        self.grid_mass = np.load(self.path+'/mass_axis.npy')
        self.grid_redshift = np.load(self.path+'/redshift_axis.npy')
        grid_sfr = np.load(self.path+'/sfr.npy').T
        grid_stellar = np.load(self.path+'/stellarmass.npy').T

        self.sfr_function = RectBivariateSpline(self.grid_redshift,self.grid_mass,grid_sfr,kx=1,ky=1)
        self.stellarmass_function = RectBivariateSpline(self.grid_redshift,self.grid_mass,grid_stellar,kx=1,ky=1)

    def _make_spline_files(self):
        """Generate the needed files from the distribution data from
        Behroozi's website.
        """

        data = np.loadtxt(self.path+'/release.dat',
                          dtype=[('redshift','f'),('mass','f'),('sfr','f'),('stellarmass','f')])

        # Put everything in the right units (1+z --> z),
        # log10(mass) --> mass
        data['redshift'] -= 1
        data['mass'] = np.power(10,data['mass'])
        data['stellarmass'] = np.power(10,data['stellarmass'])

        # Get unique values of halo mass and redshift
        redshift = np.unique(data['redshift'])
        mass = np.unique(data['mass'])

        # Make fields to put everything in
        sfr = np.empty((len(mass),len(redshift)))
        stellarmass = np.empty((len(mass),len(redshift)))

        # And put things where they belong
        for i in range(len(mass)):
            for j in range(len(redshift)):
                inds1 = np.where(data['mass'] == mass[i])
                inds2 = np.where(data['redshift'] == redshift[j])
                ind = np.intersect1d(inds1,inds2)

                sfr[i,j] = data['sfr'][ind]
                stellarmass[i,j] = data['stellarmass'][ind]

        # Figure out where we stop getting values (limits)
        for j in range(len(redshift)):
            sfr_at_limit = sfr[:,j][~np.isclose(sfr[:,j],-1000.0)][-1]
            sfr[:,j][np.isclose(sfr[:,j],-1000.0)] = sfr_at_limit

        sfr = np.power(10,sfr)

        np.save(self.path+'/sfr.npy',sfr)
        np.save(self.path+'/stellarmass.npy',stellarmass)
        np.save(self.path+'/mass_axis.npy',mass)
        np.save(self.path+'/redshift_axis.npy',redshift)

    def plot_grid(self,prop='sfr'):
        """Plot the grid.

        Parameters
        ----------
        prop : 'sfr' or 'stellar'
            The property to show
        """

        mass = np.logspace(np.log10(np.amin(self.grid_mass)),np.log10(np.amax(self.grid_mass)),1000)
        redshift = np.linspace(np.amin(self.grid_redshift),np.amax(self.grid_redshift),1000)

        x,y = np.meshgrid(redshift,mass)

        if prop == 'sfr':
            val = self.sfr_function.ev(x.flatten(),y.flatten()).reshape((1000,1000))
        elif prop == 'stellar':
            val = self.stellarmass_function.ev(x.flatten(),y.flatten()).reshape((1000,1000))
        else:
            raise ValueError("val not recognized")

        fig,ax = plt.subplots()
        ax.set(xlabel='Redshift',ylabel=r'Halo Mass [M$_\odot$]',yscale='log')

        map = ax.pcolor(x,y,val)
        cbar = fig.colorbar(map)
        if prop == 'sfr':
            cbar.set_label(r'SFR [M$_\odot$/yr]')
        if prop == 'stellar':
            cbar.set_label(r'M$_*$ [M$_\odot$]')

        plt.show()

    def sfr(self, redshift, mass, scatter=True, sigma_scatter=0.3, rng=np.random.default_rng()):
        """Function to assign star formation rates based on the model
        from Behroozi et al. 2013.

        Parameters
        ----------
        redshift : float or array
            The redshift(s) of the halo(s)
        mass : float or array
            The mass(es) of the halo(s) in Msun
        scatter : bool
            If True, a lognormal scatter will be added around the mean SFR
            assigned. Default is True
        sigma_scatter : float
            The scatter, in dex, to add around the mean. Default is 0.3
        rng : optional, numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        sfr : float or array
            The assigned SFRs in Msun/yr
        """

        if np.any(redshift > np.amax(self.grid_redshift)):
            warnings.warn('redshift exceeds maximum in Behroozi model range')
        if np.any(mass > np.amax(self.grid_mass)):
            warnings.warn('mass exceeds maximum in Behroozi model range')

        # Get SFRs and return them
        sfrval = self.sfr_function.ev(redshift,mass)

        if scatter:
            sfrval = log10normal(sfrval, sigma_scatter)

        return sfrval

    def stellarmass(self, redshift, mass, scatter=True, sigma_scatter=0.3, rng=np.random.default_rng()):
        """Function to assign stellar masses based on the model
        from Behroozi et al. 2013.

        Parameters
        ----------
        redshift : float or array
            The redshift(s) of the halo(s)
        mass : float or array
            The mass(es) of the halo(s) in Msun
        scatter : bool
            If True, a lognormal scatter will be added around the mean SFR
            assigned. Default is True
        sigma_scatter : float
            The scatter, in dex, to add around the mean. Default is 0.3
        rng : optional, numpy.random.Generator object
            Can be used to specify an rng, useful if you want to control the seed

        Returns
        -------
        stellar : float or array
            The assigned stellar mass in Msun
        """

        if np.any(redshift > np.amax(self.grid_redshift)):
            warnings.warn('redshift exceeds maximum in Behroozi model range')
        if np.any(mass > np.amax(self.grid_mass)):
            warnings.warn('mass exceeds maximum in Behroozi model range')

        # Get masses and return them
        stellarmassval = self.stellarmass_function.ev(redshift,mass)

        if scatter:
            stellarmassval = log10normal(stellarmassval, sigma_scatter)

        return stellarmassval

behroozi13 = behroozi13_base().sfr
behroozi13_stellarmass = behroozi13_base().stellarmass

class illustris_base():
    def __init__(self,sim):

        self.sim = sim

        paths = _paths._paths()

        if 'illustris' not in paths.sfrs.keys():
            paths._newsfrpath('illustris')

        self.path = paths.sfrs['illustris']

        if not os.path.exists(self.path+'/{}.npy'.format(self.sim)):
            self._make_scale_files()

        # Load ratio stuff
        loaded_values = np.load(self.path+'/{}.npy'.format(self.sim))
        redshift = loaded_values['redshift']
        ratio = loaded_values['ratio']
        self.ratio_function = interp1d(redshift,ratio)

    def _make_scale_files(self):
        """Generate the needed files from the Illustris snapshots
        """

        handler = sim.simhandler.simhandler(self.sim)

        # Get SFR density from snapshots
        def sfrdensity(sfr):
            return np.sum(sfr)/(handler.metadata['box_edge']/handler.h)**3

        sfrdensity_sim, z_sim = handler.snap_stat(sfrdensity,['sfr'])
        z_sim = np.array(z_sim)
        sfrdensity_sim = np.array(sfrdensity_sim)

        # Compute Madau+Dickinson fit values
        sfrdensity_madau = 0.015 * (1+z_sim)**2.7 / (1+((1+z_sim)/2.9)**5.6)

        # Compute the ratio
        ratio = sfrdensity_madau / sfrdensity_sim

        # Save
        save_values = np.empty(len(ratio),dtype=[('redshift','f'),('ratio','f')])
        save_values['redshift'] = z_sim
        save_values['ratio'] = ratio
        np.save(self.path+'/{}.npy'.format(self.sim), save_values)

    def sfr_corrected(self,sfr,redshift):
        """Function to rescale Illustris simulation SFRs to produce SFRDs
        matched with the Madau and Dickinson 2014 fitting function.

        Parameters
        ----------
        sfr : float or array
            The simulation SFRs
        redshift : float or array
            The redshift(s) of the halo(s)

        Returns
        -------
        sfr_rescaled : float or array
            The scaled SFRs
        """

        ratio = self.ratio_function(redshift)
        return sfr * ratio

# tngcorrections = illustris_base('TNG300-1').sfr_corrected
