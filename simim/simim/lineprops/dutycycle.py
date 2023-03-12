import numpy as np
from scipy.interpolate import interp1d

def f_duty(sf_length,redshift,cosmo):
    """Compute the fraction of star formating galaxies given a star formation
    lifetime and some cosmological information. Model assumes a single starburst
    (ie each galaxy forms stars only once) - giving sf_length = n*starburst_length
    allows an arbitrary number of star formation events

    Parameters
    ----------
    sf_length : float or array
        The length of star formation events, in Gyr
    redshift : float or array
        The redshift of the galaxies in question
    cosmo : astropy.cosmology.flatlambdacdm instance
        The cosmology to use when calculating the age of the universe

    Returns
    -------
    fraction : array
        The fraction of the time during which a galaxy forms stars
    """

    redshift = np.array(redshift, ndmin=1)
    if redshift.size>100:
        redshift_nodes = np.linspace(np.amin(redshift),np.amax(redshift))
        age_nodes = cosmo.age(redshift_nodes).value
        age_spline = interp1d(redshift_nodes,age_nodes)
        ages = age_spline(redshift)
    else:
        ages = cosmo.age(redshift).value

    fraction = sf_length/ages
    fraction[fraction>1] = 1

    return fraction
