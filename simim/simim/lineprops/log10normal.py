import numpy as np

def log10normal(x, dex, preserve_linear_mean = True, rng=np.random.default_rng()):
    """Helper function for introducing a log normal scatter. While preserving
    the linear mean of the distribution

    Parameters
    ----------
    x : float or array
        The linear mean of the distribution (the scatter will be added
        around log10(x)).
    dex : float
        The scale parameter for the distribution, in log10 units
    rng : optional, numpy.random.Generator object
        Can be used to specify an rng, useful if you want to control the seed

    Returns
    -------
    x_scattered : array
        The scattered values
    """

    mu = np.array(x)

    # Adjust value so the linear mean is preserved
    mu = mu / 10**(dex**2 * np.log(10)/2)

    x_scattered = mu * np.power(10,dex*rng.normal(loc=0,scale=1,size=mu.shape))

    return x_scattered
