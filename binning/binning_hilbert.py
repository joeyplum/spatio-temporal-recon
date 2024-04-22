import argparse
import scipy
import sigpy as sp
import numpy as np
import copy
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("dark_background")
matplotlib.use('TkAgg')
from functions.HilbertBinning import HilbertBinning as hb
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
    binner = hb(waveform_filt)
    binner.sort_fixed_bin(N_bins, 2)
    if show_plot == 1:
        binner.plot_fixed_bin(N_bins)
        plt.suptitle("Respiratory Binning")
        plt.savefig(folder + 'resp_bellows_wf_binned.png', dpi=100)
        plt.show()
    resp_gated = (binner.bin_hot).T
    resp_gated = np.array(resp_gated, dtype=bool)
    print(np.sum(resp_gated, axis=1))

    # Exclude first bin
    resp_gated = (resp_gated[1:, :]).tolist()

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

    # k = np.max(np.sum(resp_gated, axis=1))
    # print("WARNING: USING THE MAX of points per bin selected for use: " + str(k))

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
    random_k = np.random.choice(ksp_subset.shape[1], k, replace=False)
    # random_k = np.random.choice(ksp_subset.shape[1], k, replace=True)
    # print("WARNING: np.random.choice(..., replace=True): REPLACING VALUES = TRUE.")
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


