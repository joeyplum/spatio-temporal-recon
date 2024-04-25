import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import hilbert, find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter
# plt.style.use('dark_background')

class HilbertBinning:
    """
        Sorting respiratory signals into arbitrary bins for respiratory cycle analysis. Takes a given respiratory signal from
        bellows or diaphragm height, reorganizes the data into a single cosine cycle using a hilbert transform, excludes data
        that deviate too far from the average respiratory cycle and sorts the data into bins.

        Input arguments:
        signal      > the raw respiratory data in units of amplitude
        smoothing   > the degree to smooth the data for better hilbert results using a uniform convolution
        plots       > displays plots for debugging to see goodness of fit
        bin_type    > determines how to bin data
            'Fixed'    > divides all points into a fixed number of bins
            'Dynamic'  > divides all points into bins with a given number of data points
        fixed_bins  > number of bins if using fixed method
        dynamic_pts > number of points per bin using dynamic method

        Outputs:
        binned_indx >
        perc_ex     > percent of data excluded from binning

    """

    def __init__(self, signal, smoothing=130):
        # Smooth signal with an average filter
        if smoothing==None:
            self.signal = signal
        else:
            self.signal = np.convolve(signal, np.ones(smoothing), mode='same')

        # Hilbert transform
        signal_hilbert = hilbert(signal)
        self.phase = np.angle(signal_hilbert)
        self.order = np.argsort(self.phase)
        self.order_inv = np.argsort(self.order)
        self.signal_sorted = self.signal[self.order]

        # Fit data to sinusoidal phase
        self.fit_data()

    def cos_func(self, x, a, b):
        return a*np.cos(b*x)

    def fit_data(self):
        """
        Fit the signal to a cosine to determine the mean and standard deviation of the signal amplitude
        """

        sorted_signal_median = median_filter(self.signal_sorted, 2000)

        params, params_cov = curve_fit(
            self.cos_func, self.phase[self.order], self.signal_sorted, p0=[200000, 1/(2*np.pi)])
        self.amp = params[0]
        self.freq = params[1]

        stdev = np.diag(params_cov)
        self.amp_err = stdev[0]

    def breathing_rate(self, TR=3.71655E-3): # Default TR estimated from average of lots of subjects
        peaks, peak_props = find_peaks(self.signal, prominence=4000,distance=500)
        vals, val_props = find_peaks(-1 * self.signal, prominence=4000,distance=500)

        print("TR: " + str(TR) + " seconds.")
        print("Total peaks: " + str(peaks.shape[0]))
        print("Total troughs: " + str(vals.shape[0]))
        print("Total breaths: " + str(((peaks.shape[0]) + (vals.shape[0]))/2))
        N_proj = self.signal.shape[0]
        total_scan_duration = TR * N_proj
        print("Total scan duration: " + str(total_scan_duration/60) + " minutes.")
        average_breath_time = total_scan_duration/(((peaks.shape[0]) + (vals.shape[0]))/2)
        print("Average breath duration: " + str(average_breath_time) + " seconds.")

        return average_breath_time

    def filter_peaks(self):
        peaks, peak_props = find_peaks(self.signal)
        vals, val_props = find_peaks(-1 * self.signal)

        # identify low peaks
        low_peaks = (self.signal[peaks] < (0.2 * np.max(self.signal[peaks]))
                     ) & (self.signal[peaks] > (0.2 * np.min(self.signal[vals])))
        low_vals = (self.signal[vals] > (0.2 * np.min(self.signal[vals]))
                    ) & (self.signal[vals] < (0.2 * np.max(self.signal[peaks])))

        if peaks.size > vals.size:
            peaks = peaks[:-1]
        elif peaks.size < vals.size:
            vals = vals[:-1]

        # below is unideal, since 50% of the time the first peak will not be filtered, but should have minor impact

        if low_peaks.size > low_vals.size:
            low_peaks = low_peaks[:-1]
        elif low_peaks.size < low_vals.size:
            low_vals = low_vals[:-1]

        # algorithm assumes val is first, might be a better way to do this other than "left" and "right"
        if vals[0] > peaks[0]:
            peaks = peaks[1:]
            vals = vals[:-1]

        # identify low peaks
        low_peaks = (self.signal[peaks] < (0.2 * np.max(self.signal[peaks]))
                     ) & (self.signal[peaks] > (0.2 * np.min(self.signal[vals])))
        low_vals = (self.signal[vals] > (0.2 * np.min(self.signal[vals]))
                    ) & (self.signal[vals] < (0.2 * np.max(self.signal[peaks])))

        # Identify peaks / valleys adjacent to the points of interest
        peaks_right = np.append(vals[1:], vals[0], )
        vals_left = np.append(peaks[-1], peaks[:-1])

        # Determine the midpoint to the next peak
        mid_peaks_left = peaks[low_peaks] - \
            (peaks[low_peaks] - vals[low_peaks]) // 2 - 1
        mid_peaks_right = peaks[low_peaks] + \
            (peaks_right[low_peaks] - peaks[low_peaks]) // 2 + 1
        mid_vals_left = vals[low_vals] - \
            (vals[low_vals] - vals_left[low_vals]) // 2 - 1
        mid_vals_right = vals[low_vals] + \
            (peaks[low_vals] - vals[low_vals]) // 2 + 1

        # Exclude data points so that peaks/valleys to remove are in the middle of a phase
        exclude_points = np.zeros_like(self.signal, dtype=bool)
        for i, garb in enumerate(mid_peaks_left):
            exclude_points[mid_peaks_left[i]:mid_peaks_right[i]] = True

        for i, garb in enumerate(mid_vals_left):
            exclude_points[mid_vals_left[i]:mid_vals_right[i]] = True

        return exclude_points

    def outliers(self, n_std):
        """
        Determines signal points to exclude from binning. Excludes points if they are more than n_std standard deviations from the mean signal
        """
        # Identify outliers
        indx = (self.signal > (self.cos_func(self.phase, self.amp, self.freq)+n_std*self.amp_err)) |\
            (self.signal < (self.cos_func(self.phase,
             self.amp, self.freq)-n_std*self.amp_err))
        self.outliers = indx

        self.outliers = (self.outliers | self.filter_peaks())

        self.perc_ex = np.sum(self.outliers) / self.signal.size * 100

    def sort_fixed_bin(self, n_bins, stdev=1):
        """
        Sorts the signal into n bins according to both the phase and amplitude of the signal. First sorts all points into n bins
        of equal phase witdth, with bins spaced evenly starting at -pi (the bin at +pi is the same as at -pi). Next removes 
        outliers with amplituds outside of 2 standard deviations from the mean.
        Input:
        n_bins  > the number of bins to sort the signal. Should be even to ensure inspiration and expiration both have their own bin

        Output:
        bin_hot > a Nxn binary array, where N is the signal length and n the number of bins. A 1 in the nth column indicates that
                  point goes in the nth bin. If a row has no column with a 1, that row is an outlier
        """
        self.outliers(stdev)

        self.bin_array = np.zeros_like(self.signal)

        N = self.bin_array.size
        bin_centers = np.linspace(-np.pi, np.pi, n_bins+1)

        # Vector_method to put in bins
        # Make a Nxn_bins matrix to find which bin center the phase is closest to
        cent_matrix = np.tile(bin_centers, [N, 1])
        phase_matrix = np.tile(np.reshape(
            self.phase, [self.phase.size, 1]), [1, n_bins+1])
        self.bin_array = np.argmin(np.abs(cent_matrix-phase_matrix), -1)
        # bin centered on -pi is the same as +pi
        self.bin_array[self.bin_array == n_bins] = 0

        # Hot-encode the array
        bin_hot = np.zeros(
            [self.bin_array.size, self.bin_array.max()+1], dtype=int)
        bin_hot[np.arange(self.bin_array.size), self.bin_array] = 1

        # Make column 0 be for outliers
        outlier_tile = np.tile(np.reshape(
            self.outliers, [self.outliers.size, 1]), [1, n_bins])
        bin_hot[outlier_tile == 1] = 0
        bin_hot = np.concatenate(
            (np.reshape(self.outliers, [self.outliers.size, 1]), bin_hot,), axis=1)

        self.bin_hot = bin_hot
        self.n_bins = n_bins

        return bin_hot

    def sort_dynamic_bin(self, n_bins, n_pts, stdev=1):
        """
        Sorts the signal into bins according to a sliding window such that each bin contains the same number of n_pts points. Points
        can belong to multiple bins due to overlapping windows. 
        Input:
        n_bins  > Number of bins/windows
        n_pts   > Number of signal points to include per window
        stdev   > How far a point can deviate from the mean oscillation amplitude and still be included. 

        Output:
        bin_hot > a Nxn binary array, where N is the signal length and n the number of windows. A 1 in the nth column indicates that 
                  point is included in the nth window. A point may be included in multiple windows

        Current version does not include window wraparound, this needs to be included
        """
        self.outliers(stdev)

        N = self.signal.size
        self.bin_array = np.zeros([2*N, n_bins+1], dtype=bool)

        # Vector method to assign bins
        # Method requires ordered phase data -> return to signal order at the end
        bin_centers = np.linspace(-np.pi, np.pi, n_bins+1)
        cent_matrix = np.tile(bin_centers, [N, 1])
        phase_matrix = np.tile(np.reshape(self.phase[self.order], [
                               self.phase.size, 1]), [1, n_bins+1])
        # should give the indices of closest sorted phase points
        phase_centers = np.argmin(np.abs(phase_matrix-cent_matrix), 0)

        # Unwrap phase (needed since N/2 points may wrap around -pi)
        phase_centers += N//2
        outlier_matrix = np.tile(np.expand_dims(
            self.outliers[self.order], 1), [1, n_bins+1])
        outlier_matrix_unwrapped = np.concatenate(
            (outlier_matrix[N//2:, :], outlier_matrix, outlier_matrix[:N//2, :]), 0)

        # Find n/2 points from those centroids
        keep_counts = np.cumsum(outlier_matrix_unwrapped != 1, axis=0)
        keep_center = keep_counts[phase_centers, 0]
        keep_center_matrix = np.tile(keep_center, [2*N, 1])
        neg_bin = np.argmin(
            np.abs(keep_counts - (keep_center_matrix - n_pts//2)), 0)
        pos_bin = np.argmin(
            np.abs(keep_counts - (keep_center_matrix + n_pts//2)), 0)

        # Fill in binning array
        for i in range(n_bins+1):
            self.bin_array[neg_bin[i]:pos_bin[i], i] = True

        # Re-wrap phase
        self.bin_array += np.concatenate(
            [self.bin_array[N-1:2*N, :], self.bin_array[:N-1, :]])
        self.bin_array[outlier_matrix_unwrapped] = False

        # Crop to phase, drop last bin, same as first bin
        self.bin_array = self.bin_array[N//2:3*N//2, :-1]
        self.bin_array = np.concatenate([np.reshape(self.outliers[self.order], [
                                        self.outliers.size, 1]), self.bin_array], -1)  # 1st column is outliers

        # Return to signal order
        self.bin_array = self.bin_array[self.order_inv, :]

        self.n_bins = n_bins
        self.bin_low = neg_bin
        self.bin_high = pos_bin
        self.bin_hot = self.bin_array

        return self.bin_array

    def plot_signal(self):
        """
        Mainly for debugging to see how the fits / bins perform. Plots the original signal and instantaneous phase.
        """
        plt.figure(figsize=(16, 3))
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(self.signal)

        ax2 = plt.subplot(2, 1, 2, sharex=ax1)
        ax2.plot(self.phase)
        plt.xlabel("Acquisition Number")

    def plot_phase_resample(self, n_std=2):
        """
        Plots the original signal, colored by acquisition time. Also plots data re-sampled along a single phase with the same
        color codes to show where individual oscillations fit on the phase.
        """
        color = np.linspace(1, 10, np.size(self.signal))

        plt.figure(figsize=(9, 9))
        plt.subplot(2, 1, 1)
        plt.scatter(np.linspace(0, self.signal.size,
                    self.signal.size), self.signal, c=color, s=0.5)

        plt.subplot(2, 1, 2)
        plt.scatter(self.phase, self.signal, c=color, s=0.5)
        x = np.linspace(-np.pi, np.pi, self.signal.size)
        plt.plot(x, self.cos_func(x, self.amp, self.freq), c='k')
        plt.plot(x, self.cos_func(x, self.amp, self.freq) +
                 n_std*self.amp_err, c='k')
        plt.plot(x, self.cos_func(x, self.amp, self.freq) -
                 n_std*self.amp_err, c='k')
        plt.xlabel('Phase')
        plt.xlabel('Amplitude')

    def plot_fixed_bin(self, n_bins):
        """
        Plots the data interpolated to a single phase. Data are color-coded by bin and set to gray for outliers. 
        """
        hsv = matplotlib.colormaps['prism']
        c = [[0.5, 0.5, 0.5]]
        for i in range(self.n_bins):
            c.append(hsv(i/self.n_bins))

        cmap = LinearSegmentedColormap.from_list("Bin Colors", c)

        plt.figure()
        plt.subplot(2, 1, 1)
        plt.scatter(np.arange(self.signal.size), self.signal,
                    c=np.argmax(self.bin_hot, -1), cmap=cmap, s=0.5)
        plt.xlabel('Acquisition number')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.scatter(self.phase, self.signal, c=np.argmax(
            self.bin_hot, -1), cmap=cmap, s=0.5)
        plt.xlabel('Phase')
        plt.ylabel('Amplitude')

    def plot_dynamic_bin(self, n_bins):
        hsv = matplotlib.colormaps['prism']
        color = [[1, 1, 1]] # 1 = white, 0 = black
        for i in range(self.n_bins):
            color.append(hsv(i/self.n_bins))

        cmap = LinearSegmentedColormap.from_list("Bin Colors", color)

        fig, axs = plt.subplots(2, 2, figsize=(9, 6))
        gs = axs[0, 1].get_gridspec()
        axs[0, 1].remove()
        axs[1, 1].remove()
        axbig = fig.add_subplot(gs[0:2, 1])
        # fig.tight_layout()

        for i in range(self.n_bins+1):
            axs[0, 0].scatter(np.arange(self.signal.size)[
                              self.bin_hot[:, i]], self.signal[self.bin_hot[:, i]], color=color[i], s=0.5, alpha=1)
            # axs[0, 0].scatter(np.arange(self.signal.size)[
            #                   self.bin_hot[:, i]], self.signal[self.bin_hot[:, i]], color='k', s=0.5, alpha=1)
            axs[1, 0].scatter(self.phase[self.bin_hot[:, i]],
                              self.signal[self.bin_hot[:, i]], color=color[i], s=0.5, alpha=1)

         # Adding labels to subplots
        axs[0, 0].set_ylabel('Respiratory motion', fontsize=15)
        axs[1, 0].set_ylabel('Respiratory motion', fontsize=15)
        axs[0, 0].set_xlabel('Excitation number', fontsize=15)
        axs[1, 0].set_xlabel('Phase', fontsize=15)
        axs[0, 0].set_yticklabels([])
        axs[1, 0].set_yticklabels([])        
        
        scale_vector = np.arange(self.n_bins+1)
        scale_vector[0] = 1
        scale_matrix = np.tile(
            scale_vector, [self.signal.size, 1])  # for colormapping
        axbig.imshow(np.multiply(self.bin_hot[self.order], scale_matrix)[
                     :, 1:].transpose(), cmap=cmap, aspect='auto', interpolation='none')
        # plt.gcf().set_facecolor("lightgray")

        plt.ylabel('Respiratory bin', fontsize=15)
        plt.xlabel('Excitation number', fontsize=15)
        
