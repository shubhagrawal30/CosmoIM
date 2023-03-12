import numpy as np

from simim.lineprops.log10normal import log10normal
from simim.lineprops import linefreq
from simim.lineprops import dutycycle

def keating16(mass,
              m_min=1e10, A=6.3e-7,
              scatter_lco=True, sig_lco=0.3,
              rng=np.random.default_rng()):
    """Compute CO luminosities based on the simple model of Keating et al.
    2016. Note that scatter does not presever the linear mean here (preserves
    the median instead).

    Parameters
    ----------
    mass : float or array
        The mass of the halo(s), in Msun
    m_min : float
        The minimum mass for a halo to emit CO
    A : float
        Used in converting mass to luminosity: L_co = A*mass
    scatter_lco : bool, optional
        Toggles scattering on or off, default is on (True)
    sig_lco : float, optional
        The scatter in dex to add around the mean CO luminosity.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_co : float or array
        The assigned CO luminosities in Lsun
    """

    L_co = np.array(A*mass,ndmin=1)

    if scatter_lco:
        L_co = log10normal(L_co, sig_lco, preserve_linear_mean=False, rng=rng)

    L_co[mass < m_min] = 0

    return L_co


def li16(sfr,
         d_mf=1.0, a = 1.37, b = -1.74, nu_rest = linefreq.co10,
         scatter_lco = True, sig_lco = .3,
         rng = np.random.default_rng()):
    """Compute CO luminosities based on Li et al. 2016 model.

    See equations 1, 2, and 4 along with the steps outlined on page 4 of the
    Li et al. 2016 paper. We assign a CO luminosity based on a galaxy's SFR.
    Scatter can be introduced using a lognormal distribution.

    Parameters
    ----------
    sfr : float or array
        The star formation rate of the halo(s), in Msun/yr
    d_mf : float, optional
        Sets the conversion between SFR and IR luminosity (default is 1.0,
        correct for Chambrier IMF)
    a, b : floats, optional
        The slope and intercept of the integrated KS law (defaults are 1.37 and
        -1.74, taken for Li et al.'s fit to a collection of literature values)
    nu_rest : float, optional
        The rest frequency of the emission line in GHz (default is for CO(1-0)).
    scatter_lco : bool, optional
        Toggles scattering on or off, default is on (True)
    sig_lco : float, optional
        The scatter in dex to add around the mean CO luminosity, linear mean
        will be preserved.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_co : float or array
        The assigned CO luminosities in Lsun
    """

    L_ir = sfr/d_mf*1e10
    L_co_p = np.power(10,(np.log10(L_ir) - b) / a)
    L_co = L_co_p * 4.9e-5 * (nu_rest/linefreq.co10)**3

    if scatter_lco:
        L_co = log10normal(L_co, sig_lco, preserve_linear_mean=True, rng=rng)

    return np.array(L_co,ndmin=1)


def pullen13(mass, redshift, cosmo,
             rand_f_duty = True, sf_length = 0.1,
             rng=np.random.default_rng()):
    """Compute CO luminosities based on Pullen et al. 2013's Model A.

    We assign a CO luminosity based on a galaxy's mass. See equation 5 of
    Pullen. Note that the Pullen model includes a duty cycle in which only
    a fraction of the halos are assigned this luminosity and the rest are
    assigned 0 luminosity.

    Parameters
    ----------
    mass : float or array
        The mass of the halo(s), in Msun
    redshift : float or array
        The redshift(s) of the halo(s)
    cosmo : astropy.cosmology.flatlambdacdm instance
        The cosmology to use when calculating the age of the universe
    sf_length : float
        The length of star formation events for a galaxy, in Gyr. Used in
        computing the fraction of halos that are on. Default is 0.1 Gyr.
    rand_f_duty : bool, optional
        If True, a random set of galaxies will be turned on (with probability
        equal to f_duty). If False, all galaxies will be assigned f_duty times
        the computed L_co to provide the correct total luminosity but individual
        luminosities that are too low (on average). Default is True.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_co : float or array
        The assigned CO luminosities in Lsun
    """

    fraction_on = dutycycle.f_duty(sf_length,redshift,cosmo)
    mass = np.array(mass,ndmin=1)
    L_co = 1e6 * mass/5e11

    if rand_f_duty:
        f = rng.random(mass.shape)
        L_co[f>fraction_on] = 0
    else:
        L_co *= fraction_on

    return np.array(L_co,ndmin=1)

def lidz11(mass,
           f_duty=0.1, rand_f_duty=True,
           rng=np.random.default_rng()):
    """Compute CO luminosities based on the prescription of Lidz et al. 2011.

    We assign a CO luminosity based on a galaxy's mass. See equation 12 of
    Lidz. Note that the model includes a duty cycle in which only a fraction
    of the halos are assigned this luminosity and the rest are assigned 0
    luminosity. If the argument rand_f_duty is set to False, instead of
    selecting a random set of halos to be "on", all halos will be given
    f_duty * L_co as their final luminosity, resulting in a correct total
    luminosity, but individual galaxy luminosities that are too low.


    Parameters
    ----------
    mass : float or array
        The mass of the halo(s), in Msun
    f_duty : float, optional
        The fraction of galaxies which should be luminous at a given time. The
        default is 0.1. The Lids model is optimized for a universe ~10^9 years
        old and a star formation timescale of ~10^8 years, which gives 0.1
    rand_f_duty : bool, optional
        If True, a random set of galaxies will be turned on (with probability
        equal to f_duty). If False, all galaxies will be assigned f_duty times
        the computed L_co to provide the correct total luminosity but individual
        luminosities that are too low (on average)
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_co : float or array
        The assigned CO luminosities in Lsun
    """

    mass = np.array(mass,ndmin=1)
    L_co = np.zeros(mass.shape)
    if rand_f_duty:
        f = rng.random(mass.shape)
        L_co[f<f_duty] = 2.8e3 * (mass[f<f_duty]/1e8)
    else:
        L_co = 2.8e3 * (mass/1e8) * f_duty

    return L_co

# Notes - should figure out how to make scatter correlate
# May have the break for ULIRGS wrong (Lir vs Lfir)
def kamenetzky16(sfr,
                 lines=np.arange(1,14,dtype='int'),
                 d_mf=1.0,
                 scatter_lco=True,
                 sig_lco=0.3,
                 break_ulirgs=False,
                 rng=np.random.default_rng()):
    """Compute CO luminosities based on the SLEDs of Kamenetzky et al. 2016.

    We assign a CO luminosity based on a galaxy's SFR. Scatter can be
    introduced using a lognormal distribution.

    Parameters
    ----------
    sfr : float or array
        The star formation rate of the halo(s), in Msun/yr
    lines : list of ints
        The J-upper values for the transitions to generate. 1-0 through 13-12
        are available
    break_ulirgs : bool
        If True, galaxies with L_ir > 6e10 will be treated differently from
        those with a lower L_ir.
    d_mf : float, optional
        Sets the conversion between SFR and IR luminosity (default is 1.0,
        correct for Chambrier IMF)
    scatter_lco : bool, optional
        Toggles scattering on or off, default is on (True)
    sig_lco : float, optional
        The scatter in dex to add around the mean CO luminosity, linear mean
        will be preserved.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_co : float or array
        The assigned CO luminosities in Lsun
    """

    if np.any(~np.isin(lines,np.arange(1,14))):
        raise ValueError("some lines not recognized")

    lines = np.array(lines)-1

    j_upper = np.arange(1,14,dtype='int')
    f = np.array([linefreq.co10, linefreq.co21, linefreq.co32,
                  linefreq.co43, linefreq.co54, linefreq.co65,
                  linefreq.co76, linefreq.co87, linefreq.co98,
                  linefreq.co10to9, linefreq.co11to10, linefreq.co12to11,
                  linefreq.co13to12])

    # WHOLE SET:
    a =       np.array([1.27, 1.11, 1.18, 1.09, 1.05, 1.04, 0.98, 1.00, 1.03, 1.01, 1.06, 0.99, 1.12])
    a_sig =   np.array([0.04, 0.07, 0.03, 0.05, 0.03, 0.03, 0.03, 0.03, 0.04, 0.03, 0.04, 0.03, 0.04])
    b =       np.array([-1.0,  0.6,  0.1,  1.2,  1.8,  2.2,  2.9,  3.0,  2.9,  3.2,  3.1,  3.7,  2.9])
    b_sig =   np.array([ 0.4,  0.7,  0.3,  0.4,  0.3,  0.2,  0.2,  0.3,  0.3,  0.3,  0.3,  0.2,  0.3])

    # ULIRGS:
    a_ulirg       = np.array([1.15, 0.66, 0.94, 0.68, 0.96, 1.02, 0.92, 0.91, 0.92, 0.83, 0.86, 0.79, 0.85])
    a_sig_ulirg   = np.array([0.09, 0.15, 0.11, 0.13, 0.07, 0.05, 0.04, 0.04, 0.07, 0.03, 0.05, 0.04, 0.05])
    b_ulirg       = np.array([ 0.2,  4.9,  2.3,  4.9,  2.7,  2.4,  3.5,  3.7,  3.8,  4.7,  4.7,  5.3,  5.0])
    b_sig_ulirg   = np.array([ 0.8,  1.4,  1.1,  1.1,  0.6,  0.4,  0.4,  0.3,  0.5,  0.3,  0.4,  0.3,  0.4])

    # NON-ULIRGS
    a_not         = np.array([1.05, 1.12, 1.05, 1.09, 1.01, 1.01, 1.00, 1.07, 1.06, 1.12, 1.10, 1.03, 1.23])
    a_sig_not     = np.array([0.09, 0.11, 0.05, 0.07, 0.06, 0.05, 0.06, 0.07, 0.08, 0.07, 0.09, 0.07, 0.08])
    b_not         = np.array([ 0.8,  0.4,  0.9,  1.2,  2.1,  2.3,  2.8,  2.5,  2.7,  2.5,  2.7,  3.4,  2.2])
    b_sig_not     = np.array([ 0.8,  0.9,  0.4,  0.5,  0.5,  0.4,  0.4,  0.5,  0.5,  0.5,  0.6,  0.4,  0.5])

    L_ir = sfr/d_mf*1e10

    a_gal = np.zeros((len(sfr), len(lines)))
    b_gal = np.zeros((len(sfr), len(lines)))

    if break_ulirgs:
        inds = np.where(L_ir>6e10)

        a_gal[inds,:] = a_ulirg[lines]
        b_gal[inds,:] = b_ulirg[lines]

        inds = np.where(L_ir<=6e10)
        a_gal[inds,:] = a_not[lines]
        b_gal[inds,:] = b_not[lines]

    else:
        a_gal[:,:] = a[lines]
        b_gal[:,:] = b[lines]

    L_ir = L_ir.reshape(len(sfr),1) * np.ones((1,len(lines)))
    log_L_co_p = (np.log10(L_ir) - b_gal) / a_gal
    L_co_p = np.power(10,log_L_co_p)
    L_co = L_co_p * 4.9e-5 * (f[lines]/linefreq.co10)**3

    if scatter_lco:
        L_co = log10normal(L_co, sig_lco, preserve_linear_mean=True, rng=rng)

    L_co_list = []
    for i in range(len(lines)):
        L_co_list.append(L_co[:,i])

    return L_co_list
