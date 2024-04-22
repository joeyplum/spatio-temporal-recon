from scipy.signal import find_peaks
from functions.HilbertBinning import HilbertBinning as hb
import argparse
import scipy
import sigpy as sp
import numpy as np
import os
import copy
import csv
import matplotlib
import matplotlib.pyplot as plt
plt.style.use("dark_background")
# plt.style.use("default")
matplotlib.use('TkAgg')

try:
    import ReadPhilips.readphilips as rp
    from ReadPhilips.readphilips.file_io import io
    import csv
    automate_FOV = True
except:
    print("Could not load ReadPhilips script.")
    automate_FOV = False

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
    parser.add_argument('--nprojections', type=int, default=10000,
                        help='number of projections to include in each bin.')
    parser.add_argument('--exc_start', type=int, default=None,
                        help='enter index of first excitation (if you want to subset). Default == None.')
    parser.add_argument('--exc_end', type=int, default=None,
                        help='enter index of final excitation (if you want to subset). Default == None.')
    parser.add_argument('--reorder', type=int, default=0,
                        help='reorder bins to start from min mean waveform value or max k0? Default == 0, no, 1==waveform, 2==k0')

    args = parser.parse_args()

    N_bins = args.nbins
    folder = args.fname
    show_plot = args.plot
    N_projections = args.nprojections
    start_excitation = args.exc_start
    end_excitation = args.exc_end
    reorder = args.reorder

    # Check whether a specified save data path exists
    results_exist = os.path.exists(folder + "/results")

    # Create a new directory because the results path does not exist
    if not results_exist:
        os.makedirs(folder + "/results")
        print("A new directory inside: " + folder +
              " called 'results' has been created.")

    # Load motion
    motion_load = np.array(np.load(folder + "motion.npy"))
    motion_load = np.squeeze(motion_load)
    if np.size(np.shape(motion_load)) != 2:
        print('Unexpected motion data dimensions.')
    waveform = np.reshape(motion_load, (np.shape(motion_load)[
        0]*np.shape(motion_load)[1]))

    if start_excitation is not None:
        start_excitation = start_excitation
    else:
        start_excitation = 0

    if end_excitation is not None:
        end_excitation = end_excitation
    else:
        end_excitation = np.shape(motion_load)[0]*np.shape(motion_load)[1]

    excitation_range = np.arange(start_excitation, end_excitation)

    # Subset
    waveform = waveform[excitation_range]

    # Optional, normalize waveform
    def normalize_data(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    waveform_normalized = normalize_data(waveform)

    # Smooth motion waveform
    # Bandpass
    # Default - works well most the time
    # sos = scipy.signal.iirfilter(1, Wn=[0.001, 1], fs=500/3, btype="bandpass",
    #                              ftype="butter", output="sos")
    # Uncomment for rapid or noisy waveform
    # sos = scipy.signal.iirfilter(9, Wn=[0.001, 1], fs=500/3, btype="bandpass",
    #                              ftype="butter", output="sos")
    # waveform_filt = scipy.signal.sosfiltfilt(sos, waveform)

    # Apply phase offset (optional if waveform looks offset from filter)
    # waveform_filt_backward = scipy.signal.sosfilt(sos, np.flip(waveform_filt))
    # waveform_filt = np.flip(waveform_filt_backward)

    # Median filter
    # waveform_filt = scipy.signal.medfilt(waveform,31) # median filter

    # Moving average
    window_size_ma = 91  # Moving average window size, larger val = more smoothing
    waveform_filt = np.convolve(waveform, np.ones(
        window_size_ma) / window_size_ma, mode='same')

    if show_plot == 1:
        fig = plt.figure(figsize=(15, 4), dpi=100)
        plt.plot(sp.to_device(
            waveform_filt[:np.shape(waveform_filt)[0]], -1), color='c')
        plt.plot(sp.to_device(
            waveform[:np.shape(waveform_filt)[0]], -1), 'm.', markersize=0.2)
        plt.xlabel('Excitation number')
        plt.ylabel('Respiratory bellows amplitude')
        plt.title('Filtered motion according to respiratory bellows amplitude')
        fig.savefig(folder + 'resp_bellows_wf.png', dpi=100)
        plt.show()

    def find_sin_files(directory):
        sin_files = []

        # Walk through the directory and its subdirectories
        for foldername, subfolders, filenames in os.walk(folder):
            for filename in filenames:
                # Check if the file has a .sin extension

                if filename.endswith(".sin"):
                    # Get the full path of the file and add it to the list
                    sin_files.append(os.path.join(folder, filename))

                    for sin_file in sin_files:
                        print("*.sin file located: ")
                        print(sin_file)
        return sin_files

    rls_file = find_sin_files(folder)[0]
    rls = rp.PhilipsData(rls_file)
    rls.readParamOnly = True
    rls.raw_corr = False
    rls.compute()
    TR = 1e-3 * float(rls.header.get('sin').get('repetition_times')[0][0])

    # Start binning
    # binner = hb(waveform_filt, smoothing=window_size_ma)
    binner = hb(waveform_filt)

    # Work out breathing parameters
    avg_breath_length = binner.breathing_rate(TR=TR)
    segment_length = avg_breath_length / N_bins
    midpoints = []
    for i in range(N_bins):
        # Calculate the midpoint of the segment
        midpoint = i * segment_length + segment_length / 2
        midpoints.append(midpoint)

    csv_filename = folder + "results/binning_times.csv"
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['avg_breath_length_seconds', avg_breath_length])
        writer.writerow(['respiratory_bin', 'mid_time_seconds'])
        for i, midpoint in enumerate(midpoints):
            writer.writerow([i, midpoint])

    binner.sort_dynamic_bin(N_bins, N_projections, stdev=2)  # stdev=0.5-4 ish
    if show_plot == 1:
        binner.plot_dynamic_bin(N_bins)
        plt.suptitle("Respiratory binning for N = " +
                     str(N_projections) + " excitations per bin.")
        plt.savefig(folder + '/results/resp_bellows_wf_binned_' + str(N_bins) + "x" +
                    str(N_projections) + '.png', dpi=100)
        plt.show()
    resp_gated = (binner.bin_hot).T
    resp_gated = np.array(resp_gated, dtype=bool)
    bin_title = "/results/motion_binned_" + str(N_bins) + "x" + \
        str(N_projections) + ".npy"
    np.save(folder + bin_title, resp_gated)
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
        1]*np.shape(ksp)[2], np.shape(ksp)[3]))[:, excitation_range, :]
    print(np.shape(ksp))
    coord = np.load(folder + "coord.npy")
    coord = coord.reshape(
        (np.shape(coord)[0]*np.shape(coord)[1], np.shape(coord)[2], np.shape(coord)[3]))[excitation_range, ...]
    dcf = np.load(folder + "dcf.npy")
    dcf = dcf.reshape(
        (np.shape(dcf)[0] * np.shape(dcf)[1], np.shape(dcf)[2]))[excitation_range, ...]


# Subset
ksp_save = np.zeros(
    (N_bins, np.shape(ksp)[0], k, np.shape(ksp)[2]), dtype="complex")
coord_save = np.zeros((N_bins, k, np.shape(coord)[1], np.shape(coord)[2]))
# coord_save += 5 # Add 5 to initial array so that all the "empty points" are dumped on far edges of k-space
dcf_save = np.zeros((N_bins, k,  np.shape(dcf)[1]), dtype="complex")

# Initialize strorage for the mean waveform value for each bin
wf_mean = np.zeros(N_bins)
k0_mean = np.zeros(N_bins)
for gate_number in range(N_bins):
    subset = resp_gated[int(gate_number)]
    # Estimate mean resp_waveform value for each bin (to work out where max-insp/exp is located)
    wf_mean[gate_number] = np.mean(waveform_filt[subset])
    k0_mean[gate_number] = np.mean(abs(ksp[:, subset, 0]))


print("Mean waveform value for each bin (before circshifting): ")
print(wf_mean)
print("Mean k0 value for each bin (before circshifting): ")
print(k0_mean)

if reorder == 1:
    indices = np.arange(N_bins)
    max_index = np.argmin(wf_mean)
    circshifted_indices = np.roll(indices, -max_index)
    print("New order of bin indices if first bin is the max expiration according to mean waveform: ")
    print(circshifted_indices)
elif reorder == 2:
    indices = np.arange(N_bins)
    max_index = np.argmax(k0_mean)
    circshifted_indices = np.roll(indices, -max_index)
    print("New order of bin indices if first bin is the max expiration according to mean k0: ")
    print(circshifted_indices)
          
else:
    circshifted_indices = np.arange(N_bins)

for ii in range(N_bins):
    gate_number = circshifted_indices[ii]
    subset = resp_gated[int(gate_number)]

    # Select only a subset of trajectories and data
    ksp_subset = ksp[:, subset, :]
    seed_value = 111
    np.random.seed(seed_value)
    random_k = np.random.choice(ksp_subset.shape[1], k, replace=False)
    # random_k = np.random.choice(ksp_subset.shape[1], k, replace=True)
    # print("WARNING: np.random.choice(..., replace=True): REPLACING VALUES = TRUE.")
    ksp_subset = ksp_subset[:, random_k, :]
    ksp_save[ii, :, :, :] = ksp_subset
    coord_subset = coord[subset, ...]
    coord_subset = coord_subset[random_k, ...]
    coord_save[ii, ...] = coord_subset
    dcf_subset = dcf[subset, ...]
    dcf_subset = dcf_subset[random_k, ...]
    dcf_save[ii, ...] = dcf_subset


print("Saving data using with the following dimensions...")
np.save(folder + "bksp.npy", ksp_save)
print('bksp: ' + str(np.shape(ksp_save)))
np.save(folder + "bcoord.npy", coord_save)
print('bcoord: ' + str(np.shape(coord_save)))
np.save(folder + "bdcf.npy", dcf_save)
print('bdcf: ' + str(np.shape(dcf_save)))
print("...completed.")
