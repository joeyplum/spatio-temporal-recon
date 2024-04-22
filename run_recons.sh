#!/bin/bash

# Replace 'python' with the appropriate command if 'python' doesn't work in your environment.
PYTHON_EXECUTABLE=python

# Function to run the binning_quantile.py script with given arguments
run_binning_quantile() {
    $PYTHON_EXECUTABLE binning_hilbert_dynamic.py --fname /storage/Joey/MoCoLoR/data/floret-803H-043/ --nbins 6 --plot 1 --nprojections 5000 --reorder 2 
}


# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py /storage/Joey/MoCoLoR/data/floret-803H-043/ --vent_flag 0 --rho 1 --method 'cg' --gamma 1 --jsense 2 --use_dcf 3 --lambda_lr 0.0075 --recon_res 117 --scan_res 220 --res_scale 0.75 --iner_iter 5 --outer_iter 3 --init_iter 10
}

# Default MoCoLoR settings
run_recon_mocolor_default() {
    $PYTHON_EXECUTABLE recon_lrmoco_vent_npy.py /storage/Joey/MoCoLoR/data/floret-803H-043/ --rho 1 --method 'gm' --gamma 0 --jsense 1 --use_dcf 2 --lambda_lr 0.025 --recon_res 117 --scan_res 220 --res_scale 0.75 --iner_iter 5 --outer_iter 3 --init_iter 0
}

# Function to run the recon_xdgrasp_npy.py script with given arguments
run_recon_nufft() {
    $PYTHON_EXECUTABLE recon_dcf_nufft_npy.py /storage/Joey/MoCoLoR/data/floret-803H-043/ --vent_flag 0 --recon_res 117 --scan_res 220
}

# Generate a mask
run_segmentation() {
    $PYTHON_EXECUTABLE segmentation/segmentation_ute.py --fname /storage/Joey/MoCoLoR/data/floret-803H-043/results/ --filename img_mocolor_cg_24_bin_480mm_FOV_3mm_recon_resolution.nii --plot 0 --mask 1
}

echo "Running binning_quantile.py ..."
run_binning_quantile
echo "Finished binning_quantile.py"

echo "Running recon_dcf_nufft_npy.py ..."
# run_recon_nufft
echo "Finished recon_dcf_nufft_npy.py"

echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor
echo "Finished recon_mocolor_npy.py"

echo "Running segmentation_ute.py..."
# run_segmentation
echo "Finished segmentation_ute.py"

echo "Running recon_mocolor_npy.py ..."
# run_recon_mocolor_default
echo "Finished recon_mocolor_npy.py"


# You need to make the shell script executable. Open your terminal and navigate to the directory containing the run_recons.sh file, then run the following command:
# chmod +x run_recons.sh
# Followed by:
# ./run_recons.sh






