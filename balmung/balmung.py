from __future__ import division, print_function

import numpy as np
from astropy.timeseries import LombScargle
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import tqdm


class Balmung:
    def __init__(self, time: np.ndarray, flux: np.ndarray):
        """The big cheese. does prewhitening

        Args:
            time (np.ndarray): Time values
            flux (np.ndarray): Flux value corresponding to `time`
        """
        self.time = time
        self.flux = flux - np.median(flux)
        self.residual = np.copy(self.flux)
        self.removed = []

    def prewhiten(self, fmin=None, fmax=None, minimum_snr=5, maxiter=100, diagnose=True):
        # Calculate initial amplitude spectrum
        freq, amp = self.amplitude_spectrum(self.time, self.residual, fmin=fmin, fmax=fmax)

        # # Estimate noise level:
        # bkg = self.estimate_background(freq, amp)
        # snr = amp / bkg

        # Get first guess:
        # idx = np.nanargmax(snr)
        # f0, a0 = freq[idx], amp[idx]
        # phi0 =

        noise_level = minimum_snr * np.median(amp)
        f0, a0, phi0 = self.initialize_guess(fmin=fmin, fmax=fmax)

        for i in tqdm.tqdm(range(maxiter)):
            if a0 > noise_level:
                # Fit theta to lc
                popt = self.fit([f0, a0, phi0])
                self.removed.append(popt.tolist())

                # Subtract off the fitted model
                self.residual -= self.model(self.time, *popt)

                # Get new params for next iteration
                f0, a0, phi0 = self.initialize_guess(fmin=fmin, fmax=fmax)
            else:
                break

        if diagnose:
            pass
            # Do some diagnostic shit

    def fit(self, theta: list) -> np.ndarray:
        """Small wrapper for curve_fit.

        Args:
            time (np.ndarray): Time values
            flux (np.ndarray): Flux values
            theta (list): Array-like of initial guesses

        Returns:
            list: Fitted parameters
        """
        popt, _ = curve_fit(
            self.model, self.time, self.residual, p0=theta, jac=self.grad_model
        )

        # I don't expect this to happen.. but if the amplitude goes negative let's fix it:
        if popt[1] < 0:
            popt[1] *= -1.
            popt[2] += np.pi

        return popt

    def grad_model(
        self, time: np.ndarray, freq: float, amp: float, phi: float
    ) -> np.ndarray:
        """Gradient function of our pulsation model

        Args:
            time (np.ndarray): Time values
            freq (float): Frequency
            amp (float): Amplitude
            phi (float): Phase

        Returns:
            np.ndarray: Gradient vector (dModel/d_{freq,amp,phi})
        """
        factor = 2 * np.pi * freq * time + phi
        return np.array(
            [
                -2 * np.pi * amp * time * np.sin(factor),
                np.cos(factor),
                -1 * amp * np.sin(factor),
            ]
        ).T

    def model(self, time:np.ndarray, freq:float, amp:float, phi:float) -> np.ndarray:
        """And at the heart of it all, a tiny model function.

        Args:
            time (np.ndarray): Time values
            freq (float): Frequency
            amp (float): Amplitude
            phi (float): Phase

        Returns:
            np.ndarray: Sinusoid at the given parameters
        """
        return amp * np.cos((2 * np.pi * freq * time) + phi)

    def estimate_background(
        self, x: np.ndarray, y: np.ndarray, log_width: float = 0.01
    ) -> np.ndarray:
        """Estimates the background signal

        Args:
            x (np.ndarray): [description]
            y (np.ndarray): [description]
            log_width (float, optional): [description]. Defaults to 0.01.

        Returns:
            [type]: [description]
        """
        count = np.zeros(len(x), dtype=int)
        bkg = np.zeros_like(x)
        x0 = np.log10(x[0])
        while x0 < np.log10(x[-1]):
            m = np.abs(np.log10(x) - x0) < log_width
            bkg[m] += np.median(y[m])
            count[m] += 1
            x0 += 0.5 * log_width
        return bkg / count

    def find_highest_peak(self, f: np.ndarray, a: np.ndarray) -> float:
        """Uses three point parabolic interpolation to find the highest peaks in the amplitude spectrum

        Args:
            f (np.ndarray): Frequency values
            a (np.ndarray): Amplitude values

        Returns:
            float: Maximum frequency
        """
        nu, p = f, a
        nu, p = np.atleast_1d(nu, p)

        # Get index of highest peak.
        imax = np.argmax(p)

        # Determine the frequency value by parabolic interpolation
        if imax == 0 or imax == p.size - 1:
            nu_peak = p[imax]
        else:
            # Get values around the maximum. This is kinda gross
            frq1 = nu[imax - 1]
            frq2 = nu[imax]
            frq3 = nu[imax + 1]
            y1 = p[imax - 1]
            y2 = p[imax]
            y3 = p[imax + 1]

            # Parabolic interpolation formula.
            t1 = (y2 - y3) * (frq2 - frq1) ** 2 - (y2 - y1) * (frq2 - frq3) ** 2
            t2 = (y2 - y3) * (frq2 - frq1) - (y2 - y1) * (frq2 - frq3)
            nu_peak = frq2 - 0.5 * t1 / t2
        return nu_peak

    def amplitude_spectrum(
        self, t, y, fmin: float = None, fmax: float = None, oversample_factor: float = 5.0,
    ) -> tuple:
        """Calculates the amplitude spectrum at a given time and flux input

        Args:
            t (np.ndarray): Time values
            y (np.ndarray): Flux values
            fmin (float, optional): Minimum frequency. Defaults to None.
            fmax (float, optional): Maximum frequency. Defaults to None.
            oversample_factor (float, optional): Amount by which to oversample the light curve. Defaults to 5.0.

        Returns:
            tuple: Frequency and amplitude arrays
        """
        # t, y = self.time, self.residual
        tmax = t.max()
        tmin = t.min()
        df = 1.0 / (tmax - tmin)

        if fmin is None:
            fmin = df
        if fmax is None:
            fmax = 0.5 / np.median(np.diff(t))  # *nyq_mult

        freq = np.arange(fmin, fmax, df / oversample_factor)
        model = LombScargle(t, y)
        sc = model.power(freq, method="fast", normalization="psd")

        fct = np.sqrt(4.0 / len(t))
        amp = np.sqrt(sc) * fct

        return freq, amp

    def dft_phase(self, x: np.ndarray, y: np.ndarray, f: float) -> float:
        """Calculates the phase at a single frequency using the Discrete Fourier Transform

        Args:
            x (np.ndarray): Time values
            y (np.ndarray): Flux values
            f (float): Frequency

        Returns:
            float: Phase at given frequency
        """
        expo = 2.0 * np.pi * f * x
        return np.arctan2(np.sum(y * np.sin(expo)), np.sum(y * np.cos(expo)))

    def initialize_guess(self, fmin: float, fmax: float):
        time, flux = self.time, self.residual
        f, a = self.amplitude_spectrum(time, flux, fmin=fmin, fmax=fmax, oversample_factor=5.0)

        # Get freq of max power using parabolic interpolation
        f0 = self.find_highest_peak(f, a)
        # Calculate a0 at f0
        a0 = np.sqrt(
            LombScargle(time, flux).power(f0, method="fast", normalization="psd")
        ) * np.sqrt(4.0 / len(time))
        # Calculate phi0, since ASTC needs to be negative
        phi0 = -1 * self.dft_phase(time, flux, f0)
        return f0, a0, phi0

    def plot_lc(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.time, self.flux)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        return ax

    def plot_residual(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.time, self.residual)
        ax.set_xlabel("Time")
        ax.set_ylabel("Flux")
        return ax

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots()
        f, a = self.amplitude_spectrum(self.time, self.flux)
        ax.plot(f, a, lw=0.7, c="black")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("Amplitude")
        rem = np.array(self.removed)
        ax.plot(rem[:,0], rem[:,1], 'v', alpha=0.7)
        return ax

