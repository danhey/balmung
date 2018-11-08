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

    def run(self, steps=1, fmin=None, fmax=None):
        for i in tqdm(range(steps)):
            """
            # Calculate periodogram
            freq, amp = self.periodogram(self.t, self.prewhitened_y, 6)

            # Find frequency of highest peak
            fmax = self.find_highest_peak(freq, amp)

            # Fit original time series
            amp, phase = self.fit_timeseries(self.t, self.y, fmax)

            # Prewhiten original time series
            self.prewhitened_y = self.prewhiten(self.t, self.y, fmax, amp,  phase)
            """
            self.step(fmin, fmax)

    def step(self, fmin, fmax):
        # Calculate periodogram
        freq, amp = self.periodogram(self.t, self.residual, fmin=fmin, fmax=fmax, oversample=6)

        # Find frequency of highest peak
        fmax = self.find_highest_peak(freq, amp)

        # Fit time series
        amp, phase = self.fit_timeseries(self.t, self.residual, fmax)

        # Prewhiten time series
        self.residual = self.prewhiten(self.t, self.residual, fmax, amp,  phase)

    def plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        
        freq, power = self.periodogram(self.t, self.residual)
        ax.plot(freq, power)

        ax.set_xlim([freq[0], freq[-1]])
        ax.set_ylim([0, None])
        return ax

    def periodogram(self, t, y, fmin=None, fmax=None, oversample=6, mode='amplitude'):
        
        tmax = t.max()
        tmin = t.min()
        dt = np.median(np.diff(t))

        df = 1.0 / (tmax-tmin)
        ny = 0.5 / dt

        if fmin is None:
            fmin = df
        if fmax is None:
            fmax = ny
        freq = np.arange(fmin, fmax, df / oversample)
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

    def fit_timeseries(self, t, a, freq, amp0=None, phi0=None, **kwargs):
        """
        Fit a time series using a harmonic function with a fixed set of
        frequencies to determine corresponding amplitudes and phases.
        """

        t, a, freq = np.atleast_1d(t, a, freq)
        amp0 = np.ones_like(freq) if amp0 == None else np.atleast_1d(amp0)
        phi0 = 0.5 * np.ones_like(freq) if phi0 == None else np.atleast_1d(phi0)

        # Fill missing initial values for amplitudes and phases.
        if amp0.size < freq.size:
            amp0 = np.concatenate((amp0, np.ones(freq.size - amp0.size)))
        if phi0.size < freq.size:
            phi0 = np.concatenate((phi0, 0.5 * np.ones(freq.size - phi0.size)))

        # Perform leastsq fit.
        x, _, _, _, _ = leastsq(
            self._minfunc, np.concatenate((amp0, phi0)), args=(t, a, freq),
            Dfun=self._dfunc, full_output=1, col_deriv=1, **kwargs)

        # Extract amplitudes and phases from the fit result.
        n = len(x) // 2
        amp, phi = x[:n], x[n:]

        # Normalizing the results
        idx = (amp < 0)
        amp[idx] *= -1.0
        phi[idx] += 0.5
        phi = np.mod(phi, 1)

        return amp, phi

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
        Harmonic model function. Returns a time series of harmonic funtions with frequencies nu, amplitudes amp and phases phi.
        """
        
        t, nu, amp, phi = np.atleast_1d(t, nu, amp, phi)
        n = len(nu)
        res = np.zeros(len(t))
        for i in range(n):
            res += amp[i] * np.sin(2 * np.pi * (nu[i] * t + phi[i]))
        return res

    def _minfunc(self, amph, xdat, ydat, nu):
        n = len(amph) // 2
        return ydat - self.harmfunc(xdat, nu, amph[:n], amph[n:])

    def _dfunc(self, amph, xdat, ydat, nu):
        n = len(amph) // 2
        am, ph = amph[:n], amph[n:]
        res = np.zeros((len(amph), len(xdat)))
        for i in range(n):
            res[i] = -np.sin(2*np.pi * (nu[i]*xdat + ph[i]))
        for i in range(n):
            res[n+i] = -am[i] * np.cos(2*np.pi * (nu[i]*xdat + ph[i])) * 2*np.pi
        return res


