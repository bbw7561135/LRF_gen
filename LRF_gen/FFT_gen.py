import numpy
import numpy.fft
import numpy.ma
import math
import scipy.stats
import scipy.interpolate
import warnings
import mpmath

from .powerspec import SM14Powerspec

warnings.filterwarnings("ignore", category=RuntimeWarning)


def cube_make_FFT(cube_half_length, ps, m_func=None, s_func=None, 
                  scale_ratio=None, mag_randoms=None, arg_randoms=None):
    half_cube_shape = (cube_half_length*2,
                       cube_half_length*2,
                       cube_half_length+1)
    cube_shape = (cube_half_length*2, cube_half_length*2,
                  cube_half_length*2)
    double_cube_shape = (cube_half_length*4, cube_half_length*4,
                         cube_half_length*4)

    if mag_randoms is None:
        mag_randoms = numpy.random.randn(half_cube_shape[0],
                                         half_cube_shape[1],
                                         half_cube_shape[2])

    if arg_randoms is None:
        arg_randoms = scipy.stats.uniform.rvs(loc=0, scale=2*math.pi,
                                              size=half_cube_shape)

    cube = (numpy.cos(arg_randoms) + 1j*numpy.sin(arg_randoms)) * mag_randoms

    if scale_ratio:
        k_cube = numpy.fromfunction(
                   lambda i, j, k:
                   numpy.sqrt(pow((i-cube_half_length)*scale_ratio, 2)
                              + pow(j-cube_half_length, 2)+pow(k, 2)),
                   half_cube_shape)

    else:
        k_cube = numpy.fromfunction(
                   lambda i, j, k: numpy.sqrt(pow(i-cube_half_length, 2)
                                              + pow(j-cube_half_length, 2)
                                              + pow(k, 2)),
                   half_cube_shape)

    k_cube = numpy.ma.filled(numpy.ma.masked_invalid(k_cube),
                             fill_value=0)

    ps_cube = ps(k_cube)
    ps_cube = numpy.fft.ifftshift(ps_cube, axes=(0, 1))

    # correction for non-filling when L << 1/kmax
    ps_cube /= ps.fill_correction(cube_half_length)

    ps_cube = numpy.sqrt(numpy.abs(ps_cube))
    cube = cube*ps_cube[:, :, :cube_half_length+1]

    # take fft

    fft_cube = numpy.fft.irfftn(cube)
    fft_cube = numpy.fft.fftshift(fft_cube)

    # exponentiate

    mean = 0
    std = math.sqrt(ps.var)/pow(2*cube_half_length, 3)

    if m_func:
        m = numpy.fromfunction(m_func, cube_shape)    # 0
    else:
        m = 0

    if s_func:
        s = numpy.fromfunction(s_func, cube_shape)    # 1
        s *= math.sqrt(ps.var)/numpy.mean(s)
    else:
        s = math.sqrt(ps.var)

    cube = numpy.exp(m + ((fft_cube-mean)/(std))*s)

    return cube


def corr_func(i, j, k, cube_half_length, beta, outer_scale, sigma):

    mpmath.mp.dps = 10

    x = math.sqrt(pow(i-cube_half_length, 2)
                  + pow(j-cube_half_length, 2)
                  + pow(k-cube_half_length, 2))

    y = ((math.exp(pow(sigma, 2))-1)/2
         * mpmath.re(pow(x, beta-3) * (pow(1j, 3-beta)
                     * mpmath.gammainc(2-beta, a=-x*1.0j/2.54)
                     + pow(-1j, 3-beta)
                     * mpmath.gammainc(2-beta, a=x*1.0j/2.54))) + 1)

    y = mpmath.log(y)
    return float(mpmath.nstr(y))
