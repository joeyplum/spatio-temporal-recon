import argparse
import scipy
import sigpy as sp
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("dark_background")
matplotlib.use('TkAgg')

if __name__ == '__main__':

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

    # %% Generate binned data

    # folder = "data/floret-neonatal/"
    # folder = "data/floret-803H-023/"
    # folder = "data/floret-740H-034/"
    # folder = "data/floret-186H-479/"
    # folder = "data/floret-upper-airway/"

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
    # waveform_filt = scipy.signal.medfilt(waveform,15) # median filter

    # Visualize
    if show_plot == 1:
        fig = plt.figure(figsize=(15, 4), dpi=100)
        plt.plot(sp.to_device(waveform[:np.shape(waveform_filt)[0]//10], -1), 'm', label="Unfiltered")
        plt.plot(sp.to_device(
            waveform_filt[:np.shape(waveform_filt)[0]//10], -1), color='gold', linewidth=0.8, label="Filtered")
        plt.xlabel('Excitation number')
        plt.ylabel('Respiratory bellows amplitude')
        plt.title('Filtered motion according to respiratory bellows amplitude')
        plt.legend()
        fig.savefig(folder + 'resp_bellows_wf.png', dpi=100)
        plt.show()

    # Find the difference waveform (i.e. the first derivative of the waveform)
    waveform_filt_diff = np.diff(waveform_filt)


    # Make binning function

    def quantile_bins(array, num_bins=10):
        """quantile_bins

        Args:
            array (Array): Input array (1xN).
            num_bins (int): Number of bins (must be divisible by two).

        Returns:
            out: (Array) (N_bins x N) Indices of output array
        """
        if num_bins % 2:
            raise ValueError(
                f"Number of bins should be even: Current value: {num_bins}!")

        # Find the difference vector to see if increasing/decreasing
        array_diff = np.diff(array)
        array_diff = np.append(array_diff, 0)

        # Calculate the quantile values
        num_bins_halved = num_bins//2
        quantiles = np.linspace(0, 1, num_bins_halved + 1)
        bin_values = np.quantile(array, quantiles)

        # Allocate array elements to bins
        bins = np.zeros_like(array, dtype=int)
        for i in range(num_bins_halved):
            mask = np.logical_and(
                array >= bin_values[i], array <= bin_values[i+1])
            mask_decreasing = np.array(mask)
            mask_decreasing[array_diff < 0] = 0
            mask_increasing = np.array(mask)
            mask_increasing[array_diff >= 0] = 0
            bins[mask_decreasing] = i
            bins[mask_increasing] = num_bins - i - 1

        if show_plot == 1:
            fig = plt.figure(figsize=(15, 4), dpi=100)
            colors = plt.cm.rainbow(np.linspace(0, 1, num_bins))
            plt.gca().set_prop_cycle(color=colors)
            # resp_sub = array
            # bins_sub = bins
            resp_sub = array[:len(array)//10]
            bins_sub = bins[:len(bins)//10]
            for b in range(num_bins):
                resp_array = np.ma.masked_where(bins_sub != b, resp_sub)
                plt.plot(np.arange(resp_sub.size),
                         resp_array, label=f"Bin {b}")
            resp_array = np.ma.masked_where(bins_sub != num_bins, resp_sub)
            plt.plot(np.arange(resp_sub.size), resp_array,
                     label=f"Excluded", color="g")
            plt.legend()
            plt.title(
                "Binned filtered motion according to respiratory bellows amplitude")
            plt.xlabel('Excitation number')
            plt.ylabel('Respiratory bellows amplitude')
            fig.savefig(folder + 'resp_bellows_wf_binned.png', dpi=100)
            plt.show()

        # Assign output data
        out = []
        # print(bins.shape)
        for b in range(num_bins):
            idx = bins == b
            # resp_gated.append(resp_in[idx])
            out.append(idx)

        return out

    # Bin data
    resp_gated = quantile_bins(waveform_filt, num_bins=N_bins)
    print("Number of projections per respiratory bin:")
    print(np.sum(resp_gated, axis=1))

    # Optional: Merge bins 1 and 2 - DOES NOT WORK
    # resp_gated_merged = np.zeros((2, np.shape(resp_gated)[1]))
    # resp_gated_merged[0, :] = np.array(resp_gated[0]) + np.array(resp_gated[1])
    # resp_gated_merged[1, :] = np.array(resp_gated[2]) + np.array(resp_gated[3])
    # resp_gated = resp_gated_merged.astype(int)

    # Estimate "goodness of breathing"
    range_bins = np.ptp(np.sum(resp_gated, axis=1))
    range_norm = range_bins/np.max(np.sum(resp_gated, axis=1))
    print("Normalized variability of projections in each bin: " +
          str(np.round(range_norm, 3)))
    print("(normalized to max number of projections per bin)")
    print("(0 = incredible)")
    print("(1 = awful)")

    # Subset value to have same number proj in each insp exp
    k = np.min(np.sum(resp_gated, axis=1))
    print("Number of points per bin selected for use: " + str(k))

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

    # Look at k0 pts
    if show_plot == 1:
        fig = plt.figure(figsize=(15, 4), dpi=100)
        plt.plot(sp.to_device(
            abs(ksp[0, :3000, 0]), -1), color='y')
        plt.xlabel('Excitation number')
        plt.ylabel('k0 amplitude')
        plt.title('k0 amplitude of 0th channel')
        fig.savefig(folder + 'k0_amplitude.png', dpi=100)
        plt.show()

    # Subset
    ksp_save = np.zeros(
        (N_bins, np.shape(ksp)[0], k, np.shape(ksp)[2]), dtype="complex")
    coord_save = np.zeros((N_bins, k, np.shape(coord)[1], np.shape(coord)[2]))
    dcf_save = np.zeros((N_bins, k,  np.shape(dcf)[1]), dtype="complex")

    for gate_number in range(N_bins):
        subset = resp_gated[int(gate_number)]

        # Select only a subset of trajectories and data
        ksp_subset = ksp[:, subset, :]
        seed_value = 111
        np.random.seed(seed_value)
        random_k = np.random.choice(ksp_subset.shape[1], k, replace=False)
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
