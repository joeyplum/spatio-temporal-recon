## spatio-temporal-recon

Reconstruction code for handling 4D MRI data.

Created by Joseph Plummer. 

Please post an issue on the repository
page if there is a problem.

When making edits, please make a new development branch.

**Notice:**

This is a work in progress repository. 


## Installation.

Run the following commands in sequence to run the experiments (note that we use Mamba instead of Conda, as it is faster).

1. `mamba update -n base -c defaults mamba`
2. `make mamba`
3. `mamba activate fb-recon`
4. `make pip`

## Data.

Contact Joseph.Plummer@nih.gov/

Requirements:

1. `bksp` = *.npy file with complex (binned) k-space data in form: [Nbins, Ncoils, Nexc, Nread]
2. `bcoord` = *.npy file with real (binned) k-space coordinates in shape: [Nbins, Nexc, Nread, 3]

**Troubleshooting**:

1. This repository was tested on an NVIDIA GPU. If running on a system without
   the same, please remove the following packages from `environment.yaml`:
   - `cudnn`
   - `nccl`
   - `cupy`
2. Additionally, if not using an NVIDIA GPU, please set `devnum = -1` for each
   of the `*.py` files.


**Troubleshooting**:

1. If not using an NVIDIA GPU, please set `devnum = -1` for each
   of the `.py` files.

## Running the scripts. 

I run the shell script `run_recons.sh` to run the binning and reconstruction algorithms. Settings for each algorithm can be applied in this script.


## Uninstall.

To uninstall, run the following commands:

1. `mamba activate`
2. `make clean`


## DOI.

Manuscript writing in progress.
