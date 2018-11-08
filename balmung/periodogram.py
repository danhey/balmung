# -*- coding: utf-8 -*-

from __future__ import division, print_function

import numpy as np
from astropy.stats import LombScargle

__all__ = ['periodogram']

def periodogram(t, y, oversample):
        """
        Args:
            mode: amplitude or power
        """
        tmax = t.max()
        tmin = t.min()
        dt = np.median(np.diff(t))

        df = 1.0 / (tmax-tmin)
        ny = 0.5 / dt

        freq = np.arange(df, ny, df / oversample)
        power = LombScargle(t, y).power(freq)
        #fct = np.sqrt(4./len(t))

        return freq, power