# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from tqdm import tqdm
from scipy.optimize import leastsq
from astropy.stats import LombScargle
import matplotlib.pyplot as plt

class Balmung(object):
    def __init__(self, t, y):
        self.t = t
        self.y = y
        self.residual = y

        #self.prewhitened_values = []
        self.prewhitened_f = []
        self.prewhitened_amp = []
        self.prewhitened_phase = []

    def run(self, steps=1, fmin=None, fmax=None, harmonics=0, verbose=False, memory=False,
            **kwargs):

        for j in tqdm(range(steps)):
            # Calculate periodogram
            freq, amp = self.periodogram(self.t, self.residual, fmin=fmin, fmax=fmax)

            # Find frequency of highest peak
            max_freq = self.find_highest_peak(freq, amp)

            # Prewhiten frequency and harmonics specified
            for i in np.arange(0,harmonics+1):
                # Current frequency
                f0 = max_freq * (i+1)

                # Use DFT to estimate values for least squares for current frequency
                amp0, phase0 = self.dft(self.t, self.residual, f0)
                
                self.prewhitened_f.append(f0)
                self.prewhitened_amp.append(amp0)
                self.prewhitened_phase.append(phase0)

                if memory:
                    # Fit time series USING ALL PREVIOUS FREQUENCIES FOR BETTER FIT
                    f_fit, amp_fit, phase_fit = self.fit_timeseries(self.t, self.y, self.prewhitened_f,
                                                                                    self.prewhitened_amp,
                                                                                    self.prewhitened_phase)
                    self.residual = self.prewhiten(self.t, self.y, f_fit, amp_fit, phase_fit)

                else:
                    f_fit, amp_fit, phase_fit = self.fit_timeseries(self.t, self.residual, f0,amp0,phase0)
                    # Prewhiten time series
                    self.residual = self.prewhiten(self.t, self.residual, f_fit, amp_fit, phase_fit)

                # Log removed frequencies
                self.prewhitened_f = f_fit
                self.prewhitened_amp = amp_fit
                self.prewhitened_phase = phase_fit

        if verbose:
            print('Frequency')
            print(self.prewhitened_f)

    def plot(self, ax=None, **kwargs):

        if ax is None:
            fig, ax = plt.subplots()
        
        freq, power = self.periodogram(self.t, self.residual, **kwargs)
        ax.plot(freq, power)

        ax.set_xlim([freq[0], freq[-1]])
        ax.set_ylim([0, None])
        return ax

    def periodogram(self, t, y, fmin=None, fmax=None, oversample=2, mode='amplitude'):
        
        tmax = t.max()
        tmin = t.min()
        dt = np.median(np.diff(t))

        df = 1.0 / (tmax-tmin)
        ny = 0.5 / dt

        if fmin is None:
            fmin = df
        if fmax is None:
            fmax = ny
        freq = np.arange(fmin, fmax, df/oversample)
        sc = LombScargle(t, y).power(freq, method="fast", normalization="psd")
        fct = np.sqrt(4./len(t))

        if mode == 'amplitude':
            sc = np.sqrt(np.abs(sc)) * fct
        elif mode == 'power':
            sc = fct**2. * sc
        return freq, sc

    def prewhiten(self, t, a, freq, amp, phi):
        """
        Prewhitens a time series using a harmonic function with the given
        frequencies, amplitudes and phases.
        """
        t, a, freq, amp, phi = np.atleast_1d(t, a, freq, amp, phi)
        return a - self.harmfunc(t, freq, amp, phi)

    def fit_timeseries(self, t, a, freq0, amp0, phase0, **kwargs):
        """
        Fit a time series using a harmonic function with a fixed set of
        frequencies to determine corresponding amplitudes and phases.
        """

        # Perform leastsq fit.
        x, _, _, _, _ = leastsq(
            self._minfunc, x0=np.array([freq0, amp0, phase0]), args=(t, a),
            Dfun=None, full_output=1, col_deriv=1, **kwargs)
        x = x.tolist()

        # Extract amplitudes and phases from the fit result.
        nn = len(x) // 3
        nu, amp, phi = x[:nn], x[nn:2*nn], x[2*nn:]

        # Normalizing the results

        #idx = amp < 0
        #amp[idx] *= -1.0
        #phi[idx] += 0.5
        #phi = np.mod(phi, 1)

        #nu = nu.tolist()
        #amp = amp.tolist()
        #phi = phi.tolist()
        return nu, amp, phi

    def find_highest_peak(self, nu, p):
        """
        Find the frequency of the highest peak in the periodogram, using a
        3-point parabolic interpolation.
        """
        nu, p = np.atleast_1d(nu, p)

        # Get index of highest peak.
        imax = np.argmax(p)

        # Determine the frequency value by parabolic interpolation
        if imax == 0 or imax == p.size - 1:
            nu_peak = p[imax]
        else:
            # Get values around the maximum.
            frq1 = nu[imax-1]
            frq2 = nu[imax]
            frq3 = nu[imax+1]
            y1 = p[imax-1]
            y2 = p[imax]
            y3 = p[imax+1]

            # Parabolic interpolation formula.
            t1 = (y2-y3) * (frq2-frq1)**2 - (y2-y1) * (frq2-frq3)**2
            t2 = (y2-y3) * (frq2-frq1) - (y2-y1) * (frq2-frq3)
            nu_peak = frq2 - 0.5 * t1/t2
        return nu_peak

    def harmfunc(self, t, nu, amp, phi):
        """
        Harmonic model function. Returns a time series of harmonic 
        functions with frequencies nu, amplitudes amp and phases phi.
        """
        t, nu, amp, phi = np.atleast_1d(t, nu, amp, phi)
        n = len(nu)
        res = np.zeros(len(t))
        for i in range(n):
            res += amp[i] * np.sin(2 * np.pi * (nu[i] * t + phi[i]))
        return res

    def _minfunc(self, theta, xdat, ydat):
        """ Don't ask me why least squares flattens the x0 array.."""
        nn = len(theta) // 3
        nu, amp, phi = theta[:nn], theta[nn:2*nn], theta[2*nn:]

        return ydat - self.harmfunc(xdat, nu, amp, phi)

    def dft(self, x, y, freq, verbose=False):
        """Slow DFT for estimating amplitude and phase at single frequency"""
        freq = np.asarray(freq)        
        x = np.array(x)
        y = np.array(y)

        expo = 2.0 * np.pi * freq * x
        ft_real = np.sum(y * np.cos(expo))
        ft_imag = np.sum(y * np.sin(expo))
        #phase = np.arctan2(ft_imag,ft_real)
        phase = np.arctan(ft_imag/ft_real)
        amp = np.abs((ft_real+ft_imag) / (len(x)/2))
        return amp, phase
