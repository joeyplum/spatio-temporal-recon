"""
Created by Joseph Plummer on Monday 04 March 2024.

Contact:
joseph.plummer@cchmc.org

GitHub:
joeyplum
"""

# %% Import packages

import sigpy as sp
import scipy.ndimage as ndimage_c
import numpy as np

import sys
# sys.path.append("./sigpy_mc/")
import sigpy_e.cfl as cfl
import sigpy_e.ext as ext

import sigpy_e.prox as prox
import sigpy_e.reg as reg
from sigpy_e.linop_e import NFTs, Diags, DLD, Vstacks
import sigpy.mri as mr
import os
import nibabel as nib
import time

import matplotlib.pyplot as plt


# %% LowRankRecon function, with option for motion compensation

def LowRankRecon(use_dcf=1, gamma=0, res_scale=None, scan_resolution=100, recon_resolution=100,
                 fov=480, n_ref=0, reg_flag=1, vent_flag=0,
                 mr_cflag=1, lambda_lr=1e-2, iner_iter=5, outer_iter=3,
                 sup_iter=3, device=0, fname=None):
    """_summary_

    Data:
        Supplied data inside folder called fname must have:
        K-space data, called bksp.npy, with complex array (N_bin, N_coil, N_proj, N_samp)
        K-space coords, called bcoord.npy, with array (N_bin, N_proj, N_samp, N_dim)
        Sample density compensation, called bdcf.npy, with array (N_bin, N_proj, N_samp)

    Args:
        use_dcf (int, optional): _description_. Defaults to 1.
        gamma (int, optional): _description_. Defaults to 0.
        res_scale (_type_, optional): _description_. Defaults to None.
        scan_res (int, optional): _description_. Defaults to 100.
        recon_res (int, optional): _description_. Defaults to 100.
        fov (int, optional): _description_. Defaults to 480.
        n_ref (int, optional): _description_. Defaults to 0.
        reg_flag (int, optional): _description_. Defaults to 1.
        vent_flag (int, optional): _description_. Defaults to 0.
        mr_cflag (int, optional): _description_. Defaults to 1.
        lambda_lr (_type_, optional): _description_. Defaults to 1e-2.
        iner_iter (int, optional): _description_. Defaults to 5.
        outer_iter (int, optional): _description_. Defaults to 3.
        sup_iter (int, optional): _description_. Defaults to 3.
        device (int, optional): _description_. Defaults to 0.
        fname (_type_, optional): _description_. Defaults to None.
    """
    print("Use DCF:", use_dcf)
    print("Gamma:", gamma)
    print("Res Scale:", res_scale)
    print("Scan Resolution:", scan_resolution)
    print("Recon Resolution:", recon_resolution)
    print("FOV:", fov)
    print("Reference Frame:", n_ref)
    print("Reg Flag:", reg_flag)
    print("Vent Flag:", vent_flag)
    print("MR CFlag:", mr_cflag)
    print("Lambda LR:", lambda_lr)
    print("Iner Iter:", iner_iter)
    print("Outer Iter:", outer_iter)
    print("Sup Iter:", sup_iter)
    print("Device:", device)
    print("Filename:", fname)

    print('Reconstruction started...')
    tic_total = time.perf_counter()

    if res_scale is None:
        # OPTIONAL: override res_scale
        res_scale = (recon_resolution/scan_resolution)+0.05
        print("WARNING: res_scale has been overridden. res_scale == " + str(res_scale))

    # Load data
    data = np.load(os.path.join(fname, 'bksp.npy'))
    traj = np.real(np.load(os.path.join(fname, 'bcoord.npy')))
    try:
        dcf = np.sqrt(np.load(os.path.join(fname, 'bdcf.npy')))
    except:
        print("DCF could not be loaded. Will estimate a DCF automatically.")

    # Rescale coordinates
    # Can manually edit if desired
    scale = (scan_resolution, scan_resolution, scan_resolution)
    traj[..., 0] = traj[..., 0]*scale[0]
    traj[..., 1] = traj[..., 1]*scale[1]
    traj[..., 2] = traj[..., 2]*scale[2]

    # Optional: undersample along freq encoding using res_scale
    nf_scale = res_scale
    nf_arr = np.sqrt(np.sum(traj[0, 0, :, :]**2, axis=1))
    nf_e = np.sum(nf_arr < np.max(nf_arr)*nf_scale)
    print("Number of frequency encodes before trimming: " +
          str(data.shape[-1]))
    traj = traj[..., :nf_e, :]
    data = data[..., :nf_e]
    dcf = dcf[..., :nf_e]
    nphase, nCoil, npe, nfe = data.shape
    print('Number of phases used in this reconstruction: ' + str(nphase))
    print('Number of coils: ' + str(nCoil))
    print('Number of phase encodes: ' + str(npe))
    print('Number of frequency encodes (after trimming): ' + str(nfe))

    # Define image shape
    tshape = (int(recon_resolution), int(
        recon_resolution), int(recon_resolution))

    print('Density compensation...')

    print('Calibration...')


LowRankRecon()
# %%
# TODO, make gridded recon but use same inputs as low rank recon, except using my own inverse nufft methods
