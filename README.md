## spatio-temporal-recon

Reconstruction code for handling 4D MRI data.

Created by Joseph Plummer. 

Please post an issue on the repository
page if there is a problem.

When making edits, please make a new development branch.

**Notice:**

This is a work in progress repository. 

Contact me for collaborations. 


## Installation.

Run the following commands in sequence to run the experiments.

1. `conda update -n base -c defaults conda`
2. `make conda`
3. `conda activate spatio-temporal-recon`
4. `make pip`

**Troubleshooting**:

1. This repository was tested on an NVIDIA GPU. If running on a system without
   the same, please remove the following packages from `environment.yaml`:
   - `cudnn`
   - `nccl`
   - `cupy`
2. Additionally, if not using an NVIDIA GPU, please set `devnum = -1` for each
   of the `sim_*.py` files.


**Troubleshooting**:

1. If not using an NVIDIA GPU, please set `devnum = -1` for each
   of the `.py` files.

## Running the scripts. 

I personally run all scripts using the `Run Current File in Interactive Window' tool in VScode.


## Uninstall.

To uninstall, run the following commands:

1. `conda activate`
2. `make clean`


## DOI.

Manuscript writing in progress.