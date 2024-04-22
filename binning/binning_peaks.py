import argparse
import scipy
import sigpy as sp
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("dark_background")
matplotlib.use('TkAgg')
from scipy.signal import find_peaks

if __name__ == "__main__":

    # IO parameters
    parser = argparse.ArgumentParser(
        description='motion compensated low rank constrained recon.')

    parser.add_argument('--nbins', type=int, default=10,
                        help='number of respiratory phases to separate data into.')
    parser.add_argument('--fname', type=str,
                        help='folder name (e.g. data/floret-neonatal/).')
    # TODO: Fix this bool to actually work (arg parse does not support bool as written below)
    parser.add_argument('--plot', type=int, default=1,
                        help='show plots of waveforms, 1=True or 0=False.')
    args = parser.parse_args()

    N_bins = args.nbins
    folder = args.fname
    show_plot = args.plot

    def bin_waveform(resp_in, n_bins, resp_min, resp_max, prominence):
        """bin_waveform

        Args:
            resp_in (Array): _description_
            n_bins (Int): _description_
            resp_min (Float): _description_
            resp_max (Float): _description_
            prominence (Float): _description_

        Raises:
            ValueError: _description_

        Returns:
            x (Array): _description_
        """

        if n_bins % 2:
            raise ValueError(
                f"Number of bins should be even: Current value: {n_bins}!")

        # Assumed normalized
        if resp_max is None:
            resp_max = 1
            print("resp_max assumed to be = ", resp_max)
        if resp_min is None:
            resp_min = 0
            print("resp_min assumed to be = ", resp_min)

        # Copy input data
        resp = copy.deepcopy(resp_in)

        # Interpolate resp to spline
        # Find Peaks and Valleys
        if prominence is None:
            prominence = 2000
            print("prominence assumed to be = ", prominence)
        peak_idx, p_prop = find_peaks(resp, prominence=prominence)
        valley_idx, v_prop = find_peaks(resp * -1, prominence=prominence)

        if peak_idx.size < valley_idx.size:
            valley_idx = valley_idx[:-1]

        resp_smol = resp[0:4000]
        peak_idx_smol, _ = find_peaks(resp_smol, prominence=prominence)
        valley_idx_smol, _ = find_peaks(resp_smol * -1, prominence=prominence)
        
        if show_plot:
            plt.figure(figsize=(15, 4))
            plt.plot(resp_smol, "#1f77b4")
            plt.scatter(
                peak_idx_smol, resp_smol[peak_idx_smol], color="red", marker="x", label="peaks")
            plt.scatter(valley_idx_smol,
                        resp_smol[valley_idx_smol], color="gold", marker="x", label="valleys")
            try:
                plt.plot(waveform[0:4000], 'm*', markersize=0.2)
            except:
                print("Original waveform not shown.")
            plt.legend()
            plt.grid()
            plt.show()
        # plt.close()
        bins = n_bins * np.ones_like(resp)

        # Check if first location is a minima or maxima
        if (peak_idx[0] - valley_idx[0]) < 0:
            s_idx = 0
        else:
            s_idx = 1

        # Find the amplitude between peak and the base (minima)
        min_amp = resp_min
        max_amp = resp_max
        for k in range(valley_idx.size - 1):
            # Exclude if amplitude is too small or too big
            if (
                (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] > min_amp)
                & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] > min_amp)
                & (resp[peak_idx[k]] - resp[valley_idx[k + s_idx]] < max_amp)
                & (resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]] < max_amp)
            ):
                amp_left = resp[peak_idx[k]] - resp[valley_idx[k + s_idx]]
                amp_right = resp[peak_idx[k + 1]] - resp[valley_idx[k + s_idx]]

                # Find the number of data points between peak and the base (minima)
                n_left = valley_idx[k + s_idx] - peak_idx[k] 
                n_right = peak_idx[k + 1] - valley_idx[k + s_idx] 

                # Select the area of interest to find the intersection point
                resp_left = resp[peak_idx[k]: peak_idx[k] + n_left]
                resp_right = resp[valley_idx[k + s_idx]: valley_idx[k + s_idx] + n_right]

                # Intersection points
                bin_amp = amp_left / (n_bins // 2)  # Bin size for each compartment
                y_left = []
                for b in range(n_bins // 2):
                    y_left.append(resp[peak_idx[k]] - (0.5 + b) * bin_amp)

                # Bin size for each compartment
                bin_amp = amp_right / (n_bins // 2)
                y_right = []
                for b in range(n_bins // 2):
                    y_right.append(
                        resp[valley_idx[k + s_idx]] + (0.5 + b) * bin_amp)

                # Find the indices for each partition (left)
                n = [peak_idx[k]]
                m = [valley_idx[k + s_idx]]
                for b in range(n_bins // 2):
                    n.append(
                        np.argmin(np.abs(resp_left - y_left[b])) + peak_idx[k])
                    m.append(
                        np.argmin(np.abs(resp_right - y_right[b])) + valley_idx[k + s_idx])
                n.append(n_left + peak_idx[k])
                m.append(n_right + valley_idx[k + s_idx])
                # Bin assignment
                # Left
                for b in range(1 + n_bins // 2):
                    bins[n[b]: n[b + 1]] = b

                # Right
                for b in range(n_bins // 2):
                    bins[m[b]: m[b + 1]] = b + (n_bins // 2)
                bins[m[-2]: m[-1]] = 0

        if show_plot:
            plt.figure(figsize=(15, 4))
            # plt.rcParams["figure.figsize"] = (15, 4)
            # plt.rcParams["figure.facecolor"] = "black"
            # plt.rcParams["axes.facecolor"] = "black"
            plt.style.use("dark_background")
            colors = plt.cm.rainbow(np.linspace(0, 1, n_bins))
            plt.gca().set_prop_cycle(color=colors)
            resp_sub = resp[5000:15000]
            bins_sub = bins[5000:15000]
            resp_sub = resp[:len(resp)//10]
            bins_sub = bins[:len(resp)//10]
            for b in range(n_bins):
                resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
                plt.plot(np.arange(resp_sub.size), resp_array, label=f"Bin {b}")
            resp_array = np.ma.masked_where(bins_sub != n_bins, resp_sub)
            plt.plot(np.arange(resp_sub.size), resp_array,
                    label=f"Excluded", color="g")
            plt.legend()
            plt.title("Respiratory Binning")
            plt.xlabel("RF Excitation")
            plt.ylabel("Amplitude")
            plt.savefig(folder + 'resp_bellows_wf_binned.png', dpi=100)
            plt.show()
            # plt.close()

        # Bin Data
        resp_gated = []
        # print(bins.shape)
        for b in range(n_bins):
            idx = bins == b
            # resp_gated.append(resp_in[idx])
            resp_gated.append(idx)
        return resp_gated


    # Load motion
    motion_load = np.array(np.load(folder + "motion.npy"))
    motion_load = np.squeeze(motion_load)
    if np.size(np.shape(motion_load)) != 2:
        print('Unexpected motion data dimensions.')
    waveform = np.reshape(motion_load, (np.shape(motion_load)[
                            0]*np.shape(motion_load)[1]))

    # Optional, normalize waveform
    def normalize_data(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    waveform_normalized = normalize_data(waveform)


    # Smooth motion waveform
    sos = scipy.signal.iirfilter(1, Wn=[0.001, 10], fs=500, btype="bandpass",
                                    ftype="butter", output="sos")
    waveform_filt = scipy.signal.sosfilt(sos, waveform)
    # waveform_filt = scipy.signal.medfilt(waveform,3) # median filter

    if show_plot == 1:
        fig = plt.figure(figsize=(15, 4), dpi=100)
        plt.plot(sp.to_device(
            waveform_filt[:np.shape(waveform_filt)[0]], -1), color='c')
        plt.plot(sp.to_device(waveform[:np.shape(waveform_filt)[0]], -1), 'm.', markersize=0.2)
        plt.xlabel('Excitation number')
        plt.ylabel('Respiratory bellows amplitude')
        plt.title('Filtered motion according to respiratory bellows amplitude')
        fig.savefig(folder + 'resp_bellows_wf.png', dpi=100)
        plt.show()

    # Start binning
    resp_gated = bin_waveform( waveform_filt, n_bins=N_bins, resp_min=-10000, resp_max=10000, prominence=2000)
    print(np.sum(resp_gated, axis=1))

    # Estimate "goodness of breathing"
    range_bins = np.ptp(np.sum(resp_gated, axis=1))
    range_norm = range_bins/np.max(np.sum(resp_gated, axis=1))
    print("Normalized variability of projections in each bin: " +
            str(np.round(range_norm, 3)))
    print("(normalized to max number of projections per bin)")
    print("(0 = incredible)")
    print("(1 = awful)")

    # Subset value to have same number proj in each insp exp
    # k = np.min(np.sum(resp_gated, axis=1))
    # print("Number of points per bin selected for use: " + str(k))

    k = np.max(np.sum(resp_gated, axis=1))
    print("WARNING: USING THE MAX of points per bin selected for use: " + str(k))

    # Load data
    ksp = np.load(folder + "ksp.npy")
    ksp = np.reshape(ksp, (np.shape(ksp)[0], np.shape(ksp)[
        1]*np.shape(ksp)[2], np.shape(ksp)[3]))
    print(np.shape(ksp))
    coord = np.load(folder + "coord.npy")
    coord = coord.reshape(
        (np.shape(coord)[0]*np.shape(coord)[1], np.shape(coord)[2], np.shape(coord)[3]))
    dcf = np.load(folder + "dcf.npy")
    dcf = dcf.reshape((np.shape(dcf)[0] * np.shape(dcf)[1], np.shape(dcf)[2]))


# Subset
ksp_save = np.zeros(
    (N_bins, np.shape(ksp)[0], k, np.shape(ksp)[2]), dtype="complex")
coord_save = np.zeros((N_bins, k, np.shape(coord)[1], np.shape(coord)[2]))
# coord_save += 5 # Add 5 to initial array so that all the "empty points" are dumped on far edges of k-space
dcf_save = np.zeros((N_bins, k,  np.shape(dcf)[1]), dtype="complex")

for gate_number in range(N_bins):
    subset = resp_gated[int(gate_number)]

    # Select only a subset of trajectories and data
    ksp_subset = ksp[:, subset, :]
    seed_value = 111
    np.random.seed(seed_value)
    # random_k = np.random.choice(ksp_subset.shape[1], k, replace=False)
    random_k = np.random.choice(ksp_subset.shape[1], k, replace=True)
    print("WARNING: np.random.choice(..., replace=True): REPLACING VALUES = TRUE.")
    ksp_subset = ksp_subset[:, random_k, :]
    ksp_save[gate_number, :, :, :] = ksp_subset
    coord_subset = coord[subset, ...]
    coord_subset = coord_subset[random_k, ...]
    coord_save[gate_number, ...] = coord_subset
    dcf_subset = dcf[subset, ...]
    dcf_subset = dcf_subset[random_k, ...]
    dcf_save[gate_number, ...] = dcf_subset

print("Saving data using with the following dimensions...")
np.save(folder + "bksp.npy", ksp_save)
print('bksp: ' + str(np.shape(ksp_save)))
np.save(folder + "bcoord.npy", coord_save)
print('bcoord: ' + str(np.shape(coord_save)))
np.save(folder + "bdcf.npy", dcf_save)
print('bdcf: ' + str(np.shape(dcf_save)))
print("...completed.")

