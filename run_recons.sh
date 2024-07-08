#!/bin/bash

# Replace 'python' with the appropriate command if 'python' doesn't work in your environment.
PYTHON_EXECUTABLE=python

# Function to run the binning_quantile.py script with given arguments
# run_binning_quantile() {
#     $PYTHON_EXECUTABLE binning_hilbert_dynamic.py --fname /storage/Joey/MoCoLoR/data/floret-803H-043/ --nbins 12 --plot 1 --nprojections 12000 --reorder 2 
# }


# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py /home/plummerjw/fastaccessdata/20240702-healthy-volunteer/ --vent_flag 1 --rho 1 --method 'cg' --gamma 0 --jsense 2 --use_dcf 3 --lambda_lr 0.01 --recon_res 160 --scan_res 160 --res_scale 0.75 --iner_iter 5 --outer_iter 3 --init_iter 10 --device 3 --n_ref 2
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_nufft() {
    $PYTHON_EXECUTABLE recon_dcf_nufft_npy.py /home/plummerjw/fastaccessdata/20240702-healthy-volunteer/ --vent_flag 0 --recon_res 128 --scan_res 128 --device 3
}

echo "Running binning_quantile.py ..."
# run_binning_quantile
echo "Finished binning_quantile.py"

echo "Running recon_dcf_nufft_npy.py ..."
# run_recon_nufft
echo "Finished recon_dcf_nufft_npy.py"

echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor
echo "Finished recon_mocolor_npy.py"

# You need to make the shell script executable. Open your terminal and navigate to the directory containing the run_recons.sh file, then run the following command:
# chmod +x run_recons.sh
# Followed by:
# ./run_recons.sh






