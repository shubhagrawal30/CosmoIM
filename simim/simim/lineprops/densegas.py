import numpy as np

from simim.lineprops.log10normal import log10normal
from simim.lineprops import linefreq

def hcn_gao(sfr,
            d_mf=1.0, a = 1.0, b = 2.9, nu_rest = linefreq.hcn10,
            scatter_lhcn = True, sig_lhcn = .3,
            rng = np.random.default_rng()):
    """Compute the HCN(1-0) luminosity based on Chung et al. 2017's
    implementation of the Gao and Solomon 2004 fit.

    Parameters
    ----------
    sfr : float or array
        The star formation rate of the halo(s), in Msun/yr
    d_mf : float, optional
        Sets the conversion between SFR and IR luminosity (default is 1.0,
        correct for Chambrier IMF)
    a, b : floats, optional
        The slope and intercept of the power law relating L_IR to L'_HCN.
        (defaults are 1.0 and 2.9, based on Gao and Solomon's fit
    nu_rest : float, optional
        The rest frequency of the emission line in GHz (default is for HCN(1-0)).
    scatter_lhcn : bool, optional
        Toggles scattering on or off, default is on (True)
    sig_lhcn : float, optional
        The scatter in dex to add around the mean HCN luminosity, linear mean
        will be preserved.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_hcn : float or array
        The assigned HCN luminosities in Lsun

    """

    L_ir = sfr/d_mf*1e10
    L_hcn_p = np.power(10,(np.log10(L_ir) - b) / a)
    L_hcn = L_hcn_p * 2.2e-5 * (nu_rest/linefreq.hcn10)**3

    if scatter_lhcn:
        L_hcn = log10normal(L_hcn, sig_lhcn, preserve_linear_mean=True, rng=rng)

    return np.array(L_hcn,ndmin=1)

def hcn_breysse(mass,
                f_duty=0.1, rand_f_duty=True,
                rng=np.random.default_rng()):

    """Compute the HCN(1-0) luminosity based on Breysse et al.'s prescription.

    Parameters
    ----------
    mass : float or array
        The mass of the halo(s), in Msun
    f_duty : float, optional
        The fraction of galaxies which should be luminous at a given time. The
        default is 0.1.
    rand_f_duty : bool, optional
        If True, a random set of galaxies will be turned on (with probability
        equal to f_duty). If False, all galaxies will be assigned f_duty times
        the computed L_co to provide the correct total luminosity but individual
        luminosities that are too low (on average)
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_hcn : float or array
        The assigned HCN luminosities in Lsun

    """

    mass = np.array(mass,ndmin=1)
    L_hcn = np.zeros(mass.shape)
    if rand_f_duty:
        f = rng.random(mass.shape)
        L_hcn[f<f_duty] = 1.7e-15 * mass[f<f_duty]**1.67
    else:
        L_hcn = 1.7e-15 * mass**1.67 * f_duty

    return np.array(L_hcn,ndmin=1)
