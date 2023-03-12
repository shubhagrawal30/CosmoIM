import numpy as np

from simim.lineprops.log10normal import log10normal
from simim.lineprops import linefreq
import simim.constants as const

def delooze14(sfr,
              line='[CII]', set=None,
              scatter=True,
              rng=np.random.default_rng()):
    """Compute FIR line luminosities based on the fits from De Looze et
    al. 2014.

    Most lines and galaxy subsets are included. Options for line are:
        '[CII]', '[OI]', and '[OIII]'
    Options for galaxy sets are
        None, 'DGS', 'dwarfs', 'starbursts', 'comp/agn'
    For line='[CII]' you can also select the galaxy set 'high z'.

    Parameters
    ----------
    sfr : float or array
        The star formation rage of the halo(s), in Msun/yr
    line : '[CII]', '[OI]', or '[OIII]'
        The line for which the luminosity should be computed, default is [CII]
    scatter : bool, optional
        Toggles scattering on or off, default is on (True). Scatter is determined
        based on the fits provided.
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_line : array
        The assigned line luminosity in Lsun

    """

    if line not in ['[CII]','[OI]','[OIII]']:
        raise ValueError("line not recognized")
    if set is not None:
        if set not in ['DGS','dwarfs','starbursts','comp/AGN']:
            if line == '[CII]' and set == 'high z':
                pass
            else:
                raise ValueError("set not recognized")
    if line=='[CII]':
        if set==None:
            m = 1.01
            b = -6.99
            sigma = .42

        if set=='DGS':
            m = .84
            b = -5.29
            sigma = .40

        if set=='dwarfs':
            m = .80
            b = -5.73
            sigma = .37

        if set=='starbursts':
            m = 1.00
            b = -7.06
            sigma = .27

        if set=='comp/AGN':
            m = .90
            b = -6.09
            sigma = .37

        if set=='high z':
            m = 1.18
            b = -8.52
            sigma = .40

    if line=='[OI]':
        if set==None:
            m = 1.00
            b = -6.79
            sigma = .42

        if set=='DGS':
            m = .94
            b = -6.37
            sigma = .25

        if set=='dwarfs':
            m = .91
            b = -6.23
            sigma = .27

        if set=='starbursts':
            m = .89
            b = -6.05
            sigma = .20

        if set=='comp/AGN':
            m = .76
            b = -5.08
            sigma = .35

    if line=='[OIII]':
        if set==None:
            m = 1.12
            b = -7.48
            sigma = .66

        if set=='DGS':
            m = .92
            b = -6.71
            sigma = .30

        if set=='dwarfs':
            m = .92
            b = -6.71
            sigma = .30

        if set=='starbursts':
            m = .69
            b = -3.89
            sigma = .23

        if set=='comp/AGN':
            m = .87
            b = -5.46
            sigma = .35

    sfr = np.array(sfr, ndmin=1)
    L = np.power(10, (np.log10(sfr)-b)/m)
    if scatter:
        L = log10normal(L, sigma, preserve_linear_mean=True, rng=rng)

    return L

def uzgil14(sfr,
            d_mf=1.0,
            line='[CII]', set=None,
            scatter=True, sig_scatter=0.4,
            rng=np.random.default_rng()):
    """Compute FIR line luminosities based on the fits used by Uzgil et
    al. 2014. Note that these fits were incorrect, this function is here
    for historical comparison. Use the spinoglio12 function for the corrected
    fits.

    The lines available are:
        '[CII]', '[NII]', '[OI]', '[OIII]88um', '[OIII]52um',
        '[SiII]', '[SiIII]33um', '[SiIII]19um', '[NeII]',
        and '[NeIII]'

    Parameters
    ----------
    sfr : float or array
        The star formation rate of the halo(s), in Msun/yr
    d_mf : float, optional
        Sets the conversion between SFR and IR luminosity (default is 1.0,
        correct for Chambrier IMF)
    line : see above
        The line for which the luminosity should be computed, default is [CII]
    scatter : bool, optional
        Toggles scattering on or off, default is on (True)
    sig_scatter : float, optional
        The scatter in dex to add around the mean luminosity, linear mean
        will be preserved. The default is 0.4 based on the scatter found by
        DeLooze et al. 2014 for [CII]
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_line : float or array
        The assigned line luminosities in Lsun
    """

    # Note sigma_A and sigma_B are fit uncertainties, not used in calculations
    if line=='[CII]':
        A = 0.89
        sigma_A = 0.03
        B = 2.44
        sigma_B = 0.07
    elif line=='[NII]':
        A = 1.01
        sigma_A = 0.04
        B = 3.54
        sigma_B = 0.11
    elif line=='[OI]':
        A = 0.98
        sigma_A = 0.03
        B = 2.70
        sigma_B = 0.10
    elif line=='[OIII]88um':
        A = 0.98
        sigma_A = 0.10
        B = 2.86
        sigma_B = 0.30
    elif line=='[OIII]52um':
        A = 0.88
        sigma_A = 0.10
        B = 2.54
        sigma_B = 0.31
    elif line=='[SiII]':
        A = 1.04
        sigma_A = 0.05
        B = 3.15
        sigma_B = 0.16
    elif line=='[SIII]33um':
        A = 0.99
        sigma_A = 0.05
        B = 3.21
        sigma_B = 0.14
    elif line=='[SIII]19um':
        A = 0.97
        sigma_A = 0.06
        B = 3.47
        sigma_B = 0.20
    elif line=='[NeII]':
        A = 0.99
        sigma_A = 0.06
        B = 3.26
        sigma_B = 0.20
    elif line=='[NeIII]':
        A = 1.10
        sigma_A = 0.07
        B = 3.72
        sigma_B = 0.23

    sfr = np.array(sfr, ndmin=1)
    L_ir = sfr/d_mf*1e10

    # Converg to 10^41 erg/s to match units of the Uzgil fits
    L_ir = L_ir * (const.Lsun * 1e-34)

    log_L_line = A*np.log10(L_ir) - B
    L_line = 10**log_L_line

    L_line = L_line * 1e34 / const.Lsun

    if scatter:
        L_line = log10normal(L_line, sig_scatter, preserve_linear_mean=False, rng=rng)

    return L_line

def spinoglio12(sfr,
                d_mf=1.0,
                line='[CII]', set=None,
                scatter=True, sig_scatter=0.4,
                rng=np.random.default_rng()):
    """Compute FIR line luminosities based on the fits used of Spinoglio et
    al. 2012, updated to account for the erratum published in 2014.

    The lines available are:
        '[CII]', '[NII]', '[OI]', '[OIII]88um', '[OIII]52um',
        '[SiII]', '[SiIII]33um', '[SiIII]19um', '[NeII]',
        and '[NeIII]'

    Parameters
    ----------
    sfr : float or array
        The star formation rate of the halo(s), in Msun/yr
    d_mf : float, optional
        Sets the conversion between SFR and IR luminosity (default is 1.0,
        correct for Chambrier IMF)
    line : see above
        The line for which the luminosity should be computed, default is [CII]
    scatter : bool, optional
        Toggles scattering on or off, default is on (True)
    sig_scatter : float, optional
        The scatter in dex to add around the mean luminosity, linear mean
        will be preserved. The default is 0.4 based on the scatter found by
        DeLooze et al. 2014 for [CII]
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_line : float or array
        The assigned line luminosities in Lsun
    """

    if line=='[CII]':
        A = 0.89
        sigma_A = 0.03
        B = 2.67
        sigma_B = 0.08
    elif line=='[NII]':
        A = 1.01
        sigma_A = 0.04
        B = 3.80
        sigma_B = 0.12
    elif line=='[OI]':
        A = 0.98
        sigma_A = 0.03
        B = 2.95
        sigma_B = 0.11
    elif line=='[OIII]88um':
        A = 0.98
        sigma_A = 0.10
        B = 3.11
        sigma_B = 0.33
    elif line=='[OIII]52um':
        A = 0.88
        sigma_A = 0.10
        B = 2.76
        sigma_B = 0.33
    elif line=='[SiII]':
        A = 1.04
        sigma_A = 0.05
        B = 3.42
        sigma_B = 0.17
    elif line=='[SIII]33um':
        A = 0.99
        sigma_A = 0.05
        B = 3.46
        sigma_B = 0.16
    elif line=='[SIII]19um':
        A = 0.97
        sigma_A = 0.06
        B = 3.72
        sigma_B = 0.22
    elif line=='[NeII]':
        A = 0.99
        sigma_A = 0.06
        B = 2.94
        sigma_B = 0.20
    elif line=='[NeIII]':
        A = 1.10
        sigma_A = 0.07
        B = 4.00
        sigma_B = 0.25

    sfr = np.array(sfr, ndmin=1)
    L_ir = sfr/d_mf*1e10

    # Converg to 10^41 erg/s to match units of the Uzgil fits
    L_ir = L_ir * (const.Lsun * 1e-34)
    log_L_line = A*np.log10(L_ir) - B
    L_line = 10**log_L_line

    L_line = L_line * 1e34 / const.Lsun

    if scatter:
        L_line = log10normal(L_line, sig_scatter, preserve_linear_mean=False, rng=rng)

    return L_line

def schaerer20(sfr,
               set=None,
               scatter=True,sig_scatter=0.4,
               rng=np.random.default_rng()):
    
    """Compute CII line luminosities based on the fits from Schaerer et
    al. 2020 (ALPINE).

    Parameters
    ----------
    sfr : float or array
        The star formation rage of the halo(s), in Msun/yr
    set : 'ALPINE-uvir','ALPINE-sed','ALPINE-uvirx','high z-uvirx'
        The fit which should be used (see Schaerer's table A.1). All implemented
        fits are for the 3sigma limits. Default is 'ALPINE-uvirx'.
    scatter : bool, optional
        Toggles scattering on or off, default is on (True). Scatter is determined
        based on the fits provided.
    sig_scatter : float, optional
        The scatter in dex to add around the mean luminosity, linear mean
        will be preserved. The default is 0.4 based on the scatter found by
        DeLooze et al. 2014 for [CII]
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_line : array
        The assigned line luminosity in Lsun
    """

    if set is None:
        set = 'ALPINE-uvirx'
    if set not in ['ALPINE-uvir','ALPINE-sed','ALPINE-uvirx','high z-uvirx']:
        raise ValueError("set not recognized")

    if set == 'ALPINE-uvir':
        m = 1.00
        b = 7.00
    if set == 'ALPINE-sed':
        m = 0.84
        b = 7.09
    if set == 'ALPINE-uvirx':
        m = 1.17
        b = 6.61
    if set == 'high z-uvirx':
        m = 1.28
        b = 6.43

    L = 10**(m*np.log10(np.array(sfr))+b)
    if scatter:
        L = log10normal(L, sig_scatter, preserve_linear_mean=True, rng=rng)

    return L

def zhao13(sfr,
           d_mf=1.0,
           scatter=True,sig_scatter=0.4,
           rng=np.random.default_rng()):
    
    """Compute NII 205um line luminosities based on the fits from Zhao et
    al. 2013.

    Parameters
    ----------
    sfr : float or array
        The star formation rage of the halo(s), in Msun/yr
    d_mf : float (default = 1.0)
        The relation between IR luminosity and SFR is taken to be SFR=d_mf*L_IR/1e10.
        1 is the apropriate value for a Chambrier IMF. 1.7 is apropriate for a 
        Salpeter IMF.
    scatter : bool, optional
        Toggles scattering on or off, default is on (True). Scatter is determined
        based on the fits provided.
    sig_scatter : float, optional
        The scatter in dex to add around the mean luminosity, linear mean
        will be preserved. The default is 0.4 based on the scatter found by
        DeLooze et al. 2014 for [CII]
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    L_nii : array
        The assigned line luminosity in Lsun
    """

    LIR = sfr*1e10/d_mf
    L = 10**((np.log10(LIR)-4.51)/0.95)

    if scatter:
        L = log10normal(L, sig_scatter, preserve_linear_mean=True, rng=rng)

    return L

