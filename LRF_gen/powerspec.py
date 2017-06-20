from __future__ import print_function, division
import abc
import math
import numpy as np
import scipy.special


class IsmPowerspec(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def __call__(self):
        return


class SM14Powerspec(IsmPowerspec):
    """ This class holds power-spectra of the form described in
        Sale & Magorrian (2014), i.e.:

        p(k) = R * (|k| L)^{2 \omega} / (1 + (|k| L)^2)^{\gamma/2+\omega} .

        Such power take a Kolmogorov-like form: at k>>1/L

        p(k) \propto |k|^{-\gamma} ,

        but are tapered towards 0 for k<<1/L

        Attributes
        ----------
        gamma : float
            The power law slope of the power-spectrum at
            |k| >> 1/L.
            Must be greater than 0.
        omega : float
            Sets the form of the tapering/rollover of the
            power spectrum at |k| << 1/L .
            Must be greater than 0.
        L : float
            The scale corresponding to the roll-over of the
            power-spectrum.
            In a turbulent conext this corresponds to the outer
            scale, i.e. the scale of energy injection.
            All resulting distance and wavenumbers produced in this
            class will be expressed as multiples of L or 1/L
            respectively.
        R : float
            A normalistaion constant
        param_names : list
            The names of the parameters required to uniquely define the
            instance
    """

    param_names = ["gamma", "omega", "L"]

    def __init__(self, gamma=11/3, omega=0, L=1., var=1.):
        """ __init__(gamma=11/3, omega=0, L=1.)

            Initialise a Sale & Magorrian (2014) power spectrum object.

            Attributes
            ----------
            gamma : float, optional
                The power law slope of the power-spectrum at
                |k| >> 1/L.
                Must be greater than 0.
            omega : float, optional
                Sets the form of the tapering/rollover of the
                power spectrum at |k| << 1/L .
                Must be greater than 0.
            L : float, optional
                The scale corresponding to the roll-over of the
                power-spectrum.
                In a turbulent conext this corresponds to the outer
                scale, i.e. the scale of energy injection.
                All resulting distance and wavenumbers produced in this
                class will be expressed as multiples of L or 1/L
                respectively.
            var : float, optional
                The variance implied by the power-spectrum, i.e. the
                integral of the non-DC component over all wavenumbers
        """

        if gamma < 0:
            raise AttributeError("gamma<=0 implies infinite variance!")
        if omega < 0:
            raise AttributeError("omega<=0 implies infinite variance!")
        if L < 0:
            raise AttributeError("Scale length cannot be negative!")

        self.gamma = gamma
        self.omega = omega
        self.L = L
        self.var = var

        # Normalisation

        self.R = 1 / self.norm_const()

    def norm_const(self):
        """ norm_const()

            Determine the normalisation constant as in eqn 13 of
            Sale & Magorrian (2014)

            Returns
            -------
            R : float
                normalisation constant
        """
        norm_const = 4*math.pi * (scipy.special.beta((self.gamma-3)/2,
                                                     1.5+self.omega)
                                  / (2 * math.pow(self.L, 3)))
        return norm_const

    def fill_correction(self, cube_half_length):
        """ fill_correction(kmax)

            Determine approximately what proportion of the total power
            is contained within a cubic array of maximum wavenumber kmax
            in any direction

            Attributes
            ----------
            cube_half_length : float
                half the width/length/height of the cube

            Returns
            -------
            fill_correction : float
                The (approximate) proportion of the total power contained
                within the array.
        """
        # factor of 1.25 is a fudge for cube -> sphere approximation
        kmax = cube_half_length * 1.25
        fill_correction = (scipy.special.hyp2f1(1.5 + self.omega,
                                                self.gamma/2 + self.omega,
                                                2.5 + self.omega,
                                                -pow(self.L * kmax, 2))
                           * pow(kmax, 3) * pow(self.L*kmax, 2*self.omega)
                           / (3 + 2*self.omega)) * 4 * math.pi
        fill_correction /= self.norm_const()
        return fill_correction

    def __call__(self, k):
        """ __call__(k)

            Give the (3D) power spectrum for some wavenumber(s) k

            Attributes
            ----------
            k : int, ndarray
                The wavenumbers for which the power-spectrum is needed
        """

        ps = (self.var * self.R * np.power(k*self.L, 2*self.omega)
              / np.power(1 + np.power(k*self.L, 2), self.gamma/2+self.omega))
        return ps


class SM14Powerspec_2D(SM14Powerspec):
    """ This class holds power-spectra of the form described in
        Sale & Magorrian (2014), but for 2D fields i.e.:

        p(k) = R * (|k| L)^{2 \omega} / (1 + (|k| L)^2)^{\gamma/2+\omega} .

        Such power take a Kolmogorov-like form: at k>>1/L

        p(k) \propto |k|^{-\gamma} ,

        but are tapered towards 0 for k<<1/L

        Attributes
        ----------
        gamma : float
            The power law slope of the power-spectrum at
            |k| >> 1/L.
            Must be greater than 0.
        omega : float
            Sets the form of the tapering/rollover of the
            power spectrum at |k| << 1/L .
            Must be greater than 0.
        L : float
            The scale corresponding to the roll-over of the
            power-spectrum.
            In a turbulent conext this corresponds to the outer
            scale, i.e. the scale of energy injection.
            All resulting distance and wavenumbers produced in this
            class will be expressed as multiples of L or 1/L
            respectively.
        R : float
            A normalistaion constant
        param_names : list
            The names of the parameters required to uniquely define the
            instance
    """

    def norm_const(self):
        """ norm_const()

            Determine the normalisation constant as in eqn 13 of
            Sale & Magorrian (2014)

            Returns
            -------
            R : float
                normalisation constant
        """
        raise NotImplementedError("The integral needed for the 2D power-"
                                  "spectrum is not defined")
