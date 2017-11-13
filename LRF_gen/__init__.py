import numpy
import numpy.fft
import math
import scipy.stats

from . import FFT_gen
from . import SRA_gen
from . import plots
from . import FFT_rect
from .powerspec import SM14Powerspec



def get_corr(cube):
    bins = int(pow(cube.shape[0]/2., 2) + pow(cube.shape[1]/2., 2)
               + pow(cube.shape[2]/2., 2) + 1)
    covs = [[] for x in xrange(bins)]

    ff_c = numpy.fft.rfftn(cube)
    ff_c = ff_c*numpy.conj(ff_c)
    c = numpy.fft.irfftn(ff_c)
    c = numpy.fft.fftshift(c)

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                dist = (pow(i-c.shape[0]/2, 2) + pow(j-c.shape[0]/2, 2)
                        + pow(k-c.shape[0]/2, 2))
                covs[dist].append(c[i, j, k])

    return covs


def get_1DPS(cube):
    bins = int(pow(cube.shape[0]/2., 2) + pow(cube.shape[1]/2., 2)
               + pow(cube.shape[2]/2., 2) + 1)
    covs = [[] for x in xrange(bins)]

    ff_c = numpy.fft.rfftn(cube)
    ff_c = ff_c*numpy.conj(ff_c)

    c = numpy.abs(numpy.fft.fftshift(ff_c, axes=[0, 1]))

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            for k in range(c.shape[2]):
                dist = (pow(i-c.shape[0]/2, 2) + pow(j-c.shape[0]/2, 2)
                        + pow(k, 2))
                covs[dist].append(c[i, j, k])
    return covs


def get_colPS(cube):
    bins = int(pow(cube.shape[0]/2., 2) + pow(cube.shape[1]/2., 2) + 1)
    covs = [[] for x in xrange(bins)]

    cube2 = numpy.sum(cube, axis=2)

    ff_c = numpy.fft.rfftn(cube2)
    ff_c = ff_c*numpy.conj(ff_c)

    c = numpy.abs(numpy.fft.fftshift(ff_c, axes=[0]))
    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            dist = pow(i-c.shape[0]/2, 2)+pow(j, 2)
            covs[dist].append(c[i, j])
    return covs


def get_colcorr(cube):
    bins = int(pow(cube.shape[0]/2., 2) + pow(cube.shape[1]/2., 2) + 1)
    covs = [[] for x in xrange(bins)]

    cube2 = numpy.sum(cube, axis=2)

    ff_c = numpy.fft.rfftn(cube2)
    ff_c = ff_c*numpy.conj(ff_c)
    c = numpy.fft.irfftn(ff_c)/(cube2.shape[0]*(cube2.shape[1]+2))
    c = numpy.fft.fftshift(c)

    for i in range(c.shape[0]):
        for j in range(c.shape[1]):
            dist = pow(i-c.shape[0]/2, 2)+pow(j-c.shape[0]/2, 2)
            covs[dist].append(c[i, j])
    return covs


class LRF_cube:

    def __init__(self, res, sigma, gamma, method, omega=0, outer=1.,
                 m_func=None, s_func=None, scale_ratio=None, mag_randoms=None,
                 arg_randoms=None):
        # set up cubes sizes, etc
        self.res = res
        self.cube_size = pow(2, res)
        self.half_cube_size = pow(2, res-1)
        self.cube_shape = (self.cube_size, self.cube_size,
                           self.cube_size)

        self.method = method

        # set up for power spec
        self.gamma = gamma
        self.outer_scale = outer
        self.sigma = sigma

        if method == "FFT":
            self.ps = SM14Powerspec(gamma=self.gamma, omega=omega, 
                                    L=self.outer_scale,
                                    var=pow(self.sigma,2))
            self.cube = FFT_gen.cube_make_FFT(self.half_cube_size, self.ps,
                                              m_func=m_func,
                                              s_func=s_func,
                                              scale_ratio=scale_ratio,
                                              arg_randoms=arg_randoms,
                                              mag_randoms=mag_randoms)
        elif method == "SRA":
            self.cube = SRA_gen.cube_make_SRA(self.res, sigma,
                                              (self.gamma-3)/2.)
        elif method == "FFT2D":
            self.cube = FFT_rect.rect_make_FFT(self.half_cube_size, self.gamma,
                                               self.outer_scale, sigma, m_func,
                                               s_func, scale_ratio)
        self.method = method

        self.log_cube = None

    def normalise(self):

        m = numpy.mean(self.cube[:self.halfsize, :, :])
        s = numpy.std(self.cube[:self.halfsize, :, :])
        self.cube -= m
        self.cube /= s

    def dump(self, filename):

        fileobj = open(filename, "w")

        if self.method == "FFT2D":

            for i in range(self.cube.shape[0]):
                for k in range(self.cube.shape[1]):
                    fileobj.write("{0:d}\t{1:d}\t{2:.3f}\n".
                                  format(i, k, self.cube[i, k]))
        else:

            for i in range(self.cube.shape[0]):
                for j in range(self.cube.shape[1]):
                    for k in range(self.cube.shape[2]):

                        fileobj.write("{0}\t{1}\t{2}\t{3:.3f}\n".
                                      format(i, j, k, self.cube[i, j, k]))

        fileobj.close()

    def log_cube_make(self):
        if self.log_cube is None:
            self.log_cube = numpy.log(self.cube)

    def moving_av(self, size, log):

        if size == 1:
            if log is False:
                self.means_array = self.cube
            else:
                if self.log_cube is None:
                    self.log_cube_make()
                self.means_array = self.log_cube

        else:

            if self.method == "FFT":
                # As periodic, can wrap around
                self.means_array = numpy.zeros(self.cube_shape)

                if log is False:
                    # wrapping round edges of array
                    cubeA = numpy.append(self.cube, self.cube[:size, :, :],
                                         axis=0)
                else:
                    if self.log_cube is None:
                        self.log_cube_make()

                    # wrapping round edges of array
                    cubeA = numpy.append(self.log_cube,
                                         self.log_cube[:size, :, :],  axis=0)

                cubeB = numpy.append(cubeA, cubeA[:, :size, :], axis=1)
                cube2 = numpy.append(cubeB, cubeB[:, :, :size], axis=2)

                for i in range(self.cube.shape[0]):
                    for j in range(self.cube.shape[1]):
                        for k in range(self.cube.shape[2]):
                            self.means_array[i, j, k] = numpy.mean(
                                   cube2[i:i+size, j:j+size, k:k+size])
