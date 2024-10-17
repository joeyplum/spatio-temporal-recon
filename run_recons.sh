#!/bin/bash

# Replace 'python' with the appropriate command if 'python' doesn't work in your environment.
PYTHON_EXECUTABLE=python

# You need to make the shell script executable. Open your terminal and navigate to the directory containing the run_recons.sh file, then run the following command:
# chmod +x run_recons.sh
# Followed by:
# ./run_recons.sh

# Recon commands (copy in your requests here to run)

# Function to run the recon_mocolor_npy.py script with given arguments
run_recon_mocolor() {
    $PYTHON_EXECUTABLE mocolor.py data/ --binned_csm 1 --vent_flag 0 --rho 1 --method 'cg' --gamma 1 --use_dcf 3 --lambda_lr 0.005 --fov_x 220 --fov_y 220 --fov_z 220 --res_scale 1.0 --crop_x 160 --crop_y 160 --crop_z 160 --device 0 
}
echo "Running recon_mocolor_npy.py ..."
run_recon_mocolor
echo "Finished recon_mocolor_npy.py"



