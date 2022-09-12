import numpy as np
from scipy.stats import skew
from matplotlib.ticker import Locator
import matplotlib.pyplot as plt
import math


def find_threshold_minnonzero(values, min_threshold=0.00001):
    values = np.array(values)
    min_nonzero = min([abs(v) for v in values if (v != 0) & (v is not None)])
    min_nonzero_rounded_log10 = 10 ** math.floor(np.log10(min_nonzero))
    threshold = max(min_nonzero_rounded_log10, min_threshold)
    return(threshold)


def find_best_yscale(values, lim_var=0.8, lim_skew=0.8):
    values = np.array(values)
    scale = 'linear'
    if len(values) == 0:
        print('!!! no values')
        return(scale)
    elif len(values) <= 2:
        print('!!! only 2 values')
        return(scale)
    vskewness = skew(values)
    vstd = np.std(values)
    vmedian = np.median(values)
    vstdmedian = vstd / vmedian
    if (abs(vstdmedian) > lim_var) | (abs(vskewness) > lim_skew):
        if any(values <= 0):
            scale = 'symlog'
        else:
            scale = 'log'
    return(scale, vstdmedian, vskewness)


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on
    the positions of major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the symlog major ticks.

        The placement is:
        * linear for x between -linthresh and +linthresh,
        * logarithmic below -linthresh and above this +linthresh
        """
        self.linthresh = linthresh

    def __call__(self):
        'Return the locations of the ticks'
        majorlocs = self.axis.get_majorticklocs()
        # my changes to previous solution
        # this adds one majortickloc below and above the axis range
        # to extend the minor ticks outside the range of majorticklocs
        # bottom of the axis (low values)
        first_major = majorlocs[0]
        if first_major == 0:
            outrange_first = -self.linthresh
        else:
            outrange_first = first_major * float(10) ** (- np.sign(first_major))
        # top of the axis (high values)
        last_major = majorlocs[-1]
        if last_major == 0:
            outrange_last = self.linthresh
        else:
            outrange_last = last_major * float(10) ** (np.sign(last_major))
        majorlocs = np.concatenate(([outrange_first], majorlocs, [outrange_last]))

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            major_current = majorlocs[i]
            major_previous = majorlocs[i - 1]
            majorstep = major_current - major_previous
            # print('major curernt: ', major_current, ' major previous: ', major_previous)
            if abs(major_previous + majorstep / 2) < self.linthresh:
                ndivs = 10  # linear gets 10 because it starts from 0 (i.e., 0 to threshold)
            else:
                ndivs = 9  # log gets 9 because there is no zero (e.g., 1 to 10)
            minorstep = majorstep / ndivs
            locs = np.arange(major_previous, major_current, minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            'Cannot get tick locations for a {} type.'.format(type(self))
        )


if __name__ == '__main__':

    # add here some tests
    values_list = [
        np.array([-10, 0.1, 1, 2, 1.5, 1.2, 0.5, 0.7, 10.5]),
        np.array([0.1, 1, 2, 1.5, 1.2, 0.5, 0.7, 10.5]),
        np.array([1, 2, 3, 4, 5]),
    ]
    for values in values_list:
        scale, vvariation, vskewness = find_best_yscale(values)
        print(scale, vvariation, vskewness)
        fig, ax = plt.subplots()
        plt.plot(values, 'o')
        if scale == 'symlog':
            threshold = find_threshold_minnonzero(values)
            print('threshold: ', threshold)
            ax.set_yscale('symlog', linthresh=threshold)
            plt.minorticks_on()
            ax.yaxis.set_minor_locator(MinorSymLogLocator(threshold))
        else:
            plt.yscale(scale)
        ax.grid(which='both', axis='both')
        plt.show()
