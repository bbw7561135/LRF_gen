import numpy
import numpy.fft
import numpy.ma
import math
import scipy.stats
import scipy.interpolate
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


def rect_make_FFT(rect_half_length, beta, outer_scale, sigma,
                  m_func=None, s_func=None, scale_ratio=None,
                  x_half_length=None):
    if x_half_length is None:
       x_half_length = rect_half_length
    half_rect_shape = rect_half_length*2, x_half_length+1
    rect_shape = rect_half_length*2, x_half_length*2
    double_rect_shape = rect_half_length*4, x_half_length*4

    outer_scale_L = 1./outer_scale

    rect = scipy.stats.uniform.rvs(loc=0, scale=2*math.pi,
                                   size=half_rect_shape)

    rect = numpy.cos(rect) + 1j*numpy.sin(rect)    


    index =- beta    # gives-10/3 for logN power spec

    if scale_ratio:
        k2_rect = numpy.fromfunction(lambda i,k:
                                     pow((i-rect_half_length)
                                         * scale_ratio,2)+pow(k,2),
                                     half_rect_shape)

    else:
        k2_rect = numpy.fromfunction(lambda i,k: 
                                       pow(i-rect_half_length,2)
                                     + pow(k,2),half_rect_shape)

    k_rect = numpy.power(1 + k2_rect*numpy.power(outer_scale_L,2),
                         -beta/2)

    k_rect = numpy.fft.ifftshift(k_rect, axes=(0))
    var1 = math.sqrt(numpy.sum(numpy.abs(k_rect)))


    k_rect = numpy.sqrt(numpy.abs(k_rect))

    m1 = numpy.real(rect[0,0])
    rect = rect * k_rect[:,:x_half_length+1]

    # take fft

    fft_rect = numpy.fft.irfftn(rect)
    fft_rect = numpy.fft.fftshift(fft_rect)

    # exponentiate

    mean = 0
    std = var1 / (4*rect_half_length*x_half_length)

    if m_func:
        m = numpy.fromfunction(m_func , rect_shape) #0
    else:
        m = 0

    if s_func:
        s = numpy.fromfunction(s_func, rect_shape) #1
        s *= sigma/numpy.mean(s)
    else:
        s = sigma

    rect = numpy.exp(m + ((fft_rect-mean)/(std))*s)

    return rect
