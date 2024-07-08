#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:31:08 2022

@author: ftan1
"""

import argparse

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

try:
    from ReadPhilips import readphilips as rp
    from ReadPhilips.readphilips.file_io import io
    import csv
    automate_FOV = True
except:
    print("Could not load ReadPhilips script.")
    automate_FOV = False

# Usage
# python recon_lrmoco_vent_npy.py data/floret-740H-053/ --lambda_lr 0.05 --vent_flag 1 --recon_res 220 --scan_res 220 --mr_cflag 1

if __name__ == '__main__':

    # IO parameters
    parser = argparse.ArgumentParser(
        description='motion compensated low rank constrained recon.')

    parser.add_argument('--use_dcf', type=float, default=1,
                        help='use DCF on objective function, yes == 1')
    parser.add_argument('--jsense', type=float, default=1,
                        help='jsense algorithm choice, default == 1, updated == 2')
    parser.add_argument('--gamma', type=float, default=0,
                        help='T2* weighting in Fourier encoding operator. Default == 0, full weighting == 1.')

    parser.add_argument('--res_scale', type=float, default=1,
                        help='scale of resolution, full res == 1')
    parser.add_argument('--scan_res', type=float, default=300,
                        help='scan matrix size')
    parser.add_argument('--recon_res', type=float, default=160,
                        help='recon matrix size')

    parser.add_argument('--fov_x', type=float, default=1,
                        help='scale of FOV x, full res == 1')
    parser.add_argument('--fov_y', type=float, default=1,
                        help='scale of FOV y, full res == 1')
    parser.add_argument('--fov_z', type=float, default=1,
                        help='scale of FOV z, full res == 1')

    parser.add_argument('--n_ref', type=int, default=0,
                        help='reference frame')
    parser.add_argument('--reg_flag', type=int, default=1,
                        help='derive motion field from registration')
    parser.add_argument('--vent_flag', type=int, default=0,
                        help='output jacobian determinant and specific ventilation')
    parser.add_argument('--mr_cflag', type=int, default=1,
                        help='Resp motion compensation')

    parser.add_argument('--rho', type=float, default=1,
                        help='ADMM rho parameter, default is ')
    parser.add_argument('--lambda_lr', type=float, default=1e-2,
                        help='low rank regularization, default is 0.01')
    parser.add_argument('--init_iter', type=int, default=10,
                        help='Num of initialization iterations (with no moco component).')
    parser.add_argument('--iner_iter', type=int, default=5,
                        help='Num of inner iterations.')
    parser.add_argument('--outer_iter', type=int, default=3,
                        help='Num of outer iterations.')
    parser.add_argument('--sup_iter', type=int, default=3,
                        help='Num of superior iterations.')
    parser.add_argument('--method', type=str,
                        help='Iterative method for inner loop (cg or gm, default = gm).')

    parser.add_argument('--device', type=int, default=0,
                        help='Computing device.')
    parser.add_argument('fname', type=str, default='gm',
                        help='Prefix of raw data and output(_mocolor).')
    # a set of CFL files, including(kspace, trajectory, and density_compensation_weighting)
    args = parser.parse_args()

    #
    use_dcf = args.use_dcf
    gamma = args.gamma
    jsense = args.jsense
    res_scale = args.res_scale
    scan_resolution = args.scan_res
    recon_resolution = args.recon_res
    fname = args.fname
    lambda_lr = args.lambda_lr
    device = args.device
    rho = args.rho
    init_iter = args.init_iter
    outer_iter = args.outer_iter
    iner_iter = args.iner_iter
    sup_iter = args.sup_iter
    method = args.method
    fov_scale = (args.fov_x, args.fov_y, args.fov_z)
    n_ref = args.n_ref
    reg_flag = args.reg_flag
    mr_cflag = args.mr_cflag
    vent_flag = args.vent_flag

    print('Reconstruction started...')
    tic_total = time.perf_counter()

    def find_sin_files(directory):
        sin_files = []

        # Walk through the directory and its subdirectories
        for foldername, subfolders, filenames in os.walk(fname):
            for filename in filenames:
                # Check if the file has a .sin extension
                
                if filename.endswith(".sin"):
                    # Get the full path of the file and add it to the list
                    sin_files.append(os.path.join(fname, filename))
                    
                    for sin_file in sin_files:
                        print("*.sin file located: ")
                        print(sin_file)
        return sin_files                        

    try:
        rls_file = find_sin_files(fname)[0]
    except:
        print("Could not locate *.sin file.")

    if automate_FOV:
        try:
            rls = rp.PhilipsData(rls_file)
            rls.readParamOnly = True
            rls.raw_corr = False
            rls.compute()
            # scan_resolution = int(rls.header.get(
            # 'sin').get('scan_resolutions')[0][0])
            # scan_resolution = 300 # Adult
            # scan_resolution = 200 # neonatal
            print("Automated scan_resolution = " + str(scan_resolution))
            slice_thickness = float(rls.header.get('sin').get('slice_thickness')[0][0])
            field_of_view = int(slice_thickness * scan_resolution)
            # field_of_view = 480 # Adult
            # field_of_view = 200 # neonatal
            TR = float(rls.header.get('sin').get('repetition_times')[0][0]) 
            TE = float(rls.header.get('sin').get('echo_times')[0][0]) 
            flip_angle_applied = float(rls.header.get('sin').get('flip_angles')[0][0]) 

            print("WARNING: forcefully overwriting recon_resolution:")
            recon_voxel_size = field_of_view // recon_resolution
            # recon_voxel_size = 3 # mm
            # recon_voxel_size = 1.2 # mm # neonatal
            # recon_resolution = field_of_view / recon_voxel_size
            print("recon_resolution set to: " + str(recon_voxel_size))

            try:
                print("Exporting important parameters...")

                important_data = {"aqcuisition_matrix": scan_resolution,
                                  "acquisition_voxel_size_mm": slice_thickness,
                                  "field_of_view_mm": field_of_view,
                                  "recon_matrix": recon_resolution,
                                  "recon_voxel_size_mm": field_of_view/recon_resolution,
                                  "repetition_time_ms": TR,
                                  "echo_time_ms": TE,
                                  "flip_angle_deg": flip_angle_applied}
                csv_filename = fname + "results/imaging_parameters.csv"
                with open(csv_filename, mode='w', newline='') as file:
                    writer = csv.writer(file)
    
                    # Write the header
                    writer.writerow(["Parameter", "Value"])

                    # Write the data
                    for key, value in important_data.items():
                        writer.writerow([key, value])   

                print("Important parameters exported successfully.")
            except:
                print("Could not export important parameters from *.sin file.")
            del(rls)
            print("Raw-Lab-Sin cleared from memory.")
        except:
            print("raw-lab-sin reading failed. User-defined scan resolution used instead.")


    # OPTIONAL: override res_scale
    res_scale = (recon_resolution/scan_resolution)+0.05
    print("WARNING: res_scale has been overridden. res_scale == " + str(res_scale))
        
        
    # data loading
    data = np.load(os.path.join(fname, 'bksp.npy'))
    traj = np.real(np.load(os.path.join(fname, 'bcoord.npy')))
    try:
        dcf = np.sqrt(np.load(os.path.join(fname, 'bdcf_pipemenon.npy')))
        print("Philips DCF used.")
    except:
        dcf = np.zeros((data.shape[0], data.shape[2], data.shape[3]), dtype=complex)
        print("No dcf located.")

    nf_scale = res_scale
    nf_arr = np.sqrt(np.sum(traj[0, 0, :, :]**2, axis=1))
    nf_e = np.sum(nf_arr < np.max(nf_arr)*nf_scale)
    # ANISOTROPIC
    scale = (scan_resolution, scan_resolution, 80)  
    print("FORCEFULLY OVERWRITTEN FOVz")
    # scale = fov_scale
    traj[..., 0] = traj[..., 0]*scale[0]
    traj[..., 1] = traj[..., 1]*scale[1]
    traj[..., 2] = traj[..., 2]*scale[2]

    # Optional: undersample along freq encoding - JWP 20230815
    print("Number of frequency encodes before trimming: " + str(data.shape[-1]))
    traj = traj[..., :nf_e, :]
    data = data[..., :nf_e]
    dcf = dcf[..., :nf_e]
    
    nphase, nCoil, npe, nfe = data.shape
    tshape = (int(np.max(traj[..., 0])-np.min(traj[..., 0])), int(np.max(
        traj[..., 1])-np.min(traj[..., 1])), int(np.max(traj[..., 2])-np.min(traj[..., 2])))
    # Or use manual input settings
    tshape = (int(recon_resolution), int(
        recon_resolution), 80)

    print('Number of phases used in this reconstruction: ' + str(nphase))
    print('Number of coils: ' + str(nCoil))
    print('Number of phase encodes: ' + str(npe))
    print('Number of frequency encodes (after trimming): ' + str(nfe))
    
    # Check whether a specified save data path exists
    results_exist = os.path.exists(fname + "/results")

    # Create a new directory because the results path does not exist
    if not results_exist:
        os.makedirs(fname + "/results")
        print("A new directory inside: " + fname +
              " called 'results' has been created.")

    # Save images as Nifti files
    # Build an array using matrix multiplication
    scaling_affine = np.array([[-1, 0, 0, 0],
                               [0, -1, 0, 0],
                               [0, 0, -1, 0],
                               [0, 0, 0, 1]])

    # Rotate gamma radians about axis i
    cos_gamma = np.cos(0)
    sin_gamma = np.sin(0)
    rotation_affine_1 = np.array([[1, 0, 0, 0],
                                  [0, cos_gamma, -sin_gamma,  0],
                                  [0, sin_gamma, cos_gamma, 0],
                                  [0, 0, 0, 1]])
    cos_gamma = np.cos(0)
    sin_gamma = np.sin(0)
    rotation_affine_2 = np.array([[cos_gamma, 0, sin_gamma, 0],
                                  [0, 1, 0, 0],
                                  [-sin_gamma, 0, cos_gamma, 0],
                                  [0, 0, 0, 1]])
    cos_gamma = np.cos(0)
    sin_gamma = np.sin(0)
    rotation_affine_3 = np.array([[cos_gamma, -sin_gamma, 0, 0],
                                  [sin_gamma, cos_gamma, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]])
    rotation_affine = rotation_affine_1.dot(
        rotation_affine_2.dot(rotation_affine_3))

    # Apply translation
    translation_affine = np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 0, 0, 1]])

    # Multiply matrices together
    aff = translation_affine.dot(rotation_affine.dot(scaling_affine))
    print("Affine transformation matrix used for saving this data: ")
    print(str(aff))

    # data loading for sens map
    try:
        print("Calculating sensitivity map from raw (unbinned) data...")
        ksp = np.load(fname + "ksp.npy")
        print("ksp.shape = " + str(np.shape(ksp)))
        coord = np.load(fname + "coord.npy")
        coord[..., 0] = coord[..., 0]*scale[0]
        coord[..., 1] = coord[..., 1]*scale[1]
        coord[..., 2] = coord[..., 2]*scale[2] 
        mps = mr.app.JsenseRecon(y=ksp[..., :nf_e], coord=coord[:, :nf_e, :], device=sp.Device(
                device), img_shape=tshape, mps_ker_width=18, ksp_calib_width=24, lamda=1e-4, 
                                                max_inner_iter=10, max_iter=10).run()
    
        del(ksp, coord)
        S = sp.linop.Multiply(tshape, mps)
        print("Success.")
    except:
        # calibration
        print("Failed.")
        print('Calculating sensitivity map from binned data instead...')
        ksp = np.reshape(np.transpose(data, (1, 0, 2, 3)),
                        (nCoil, nphase*npe, nfe))
        dcf2 = np.reshape(dcf**2, (nphase*npe, nfe))
        dcf_jsense = dcf2  # Must use DCF for older SIGPY JSENSE as it solves a Cartesian problem :(
        
        # Default
        # mps = ext.jsens_calib(ksp, coord, dcf2, device=sp.Device(
        #     device), ishape=tshape, mps_ker_width=12, ksp_calib_width=24)
        # Modified by JWP 20230828
        coord = np.reshape(traj, (nphase*npe, nfe, 3))
        mps = ext.jsens_calib(ksp[..., :nf_e], coord[:, :nf_e, :], dcf_jsense[..., :nf_e], device=sp.Device(
            device), ishape=tshape, mps_ker_width=8, ksp_calib_width=16)
        del(dcf_jsense, dcf2, coord, ksp)
        S = sp.linop.Multiply(tshape, mps)
        # S = sp.linop.Multiply(tshape, np.ones((1,)+tshape)) # ONES
        
    # Visualize estimated sensitivity maps    
    # try:
    #     import sigpy.plot as pl
    #     pl.ImagePlot(mps, x=1, y=2, z=0,
    #                 title="Sensitivity maps estimated with J-SENSE")
    #     plt.show()
    # except:
    #     print("Could not show the sensitivity maps.")


    print('Density compensation...')
    if use_dcf == 0:
        dcf2 = np.ones_like(dcf2)
        dcf = np.ones_like(dcf)
        print("DCF will not be used to precondition the objective function.")
    elif use_dcf == 2:
        dcf = np.zeros_like(dcf)
        print(
            "A new DCF will be calculated based on the coordinate trajectories and image shape. ")
        for i in range(nphase):
            dcf[i, ...] = sp.to_device(ext.pipe_menon_dcf(
                traj[i, ...], device=sp.Device(0), img_shape=tshape), -1)
        dcf /= np.max(dcf)
        np.save(fname + "bdcf_pipemenon.npy", dcf)
        dcf = dcf**0.5
    elif use_dcf == 3:
        dcf = np.zeros_like(dcf)
        print("Attempting k-space preconditoner calculation, as per Ong, et. al.,...")
        for i in range(nphase):
            # Approximate using a single channel precond for now (helps for speed)
            ones = np.ones_like(mps)
            ones /= len(mps)**0.5
            p = sp.to_device(mr.kspace_precond(
                ones,
                coord=sp.to_device(traj[i, ...], device),
                device=sp.Device(device), lamda=1e-3), -1)
            dcf[i, ...] = p[0, ...] # Use only first channel for preconditioner, all will be same in this case
            del (p)
        dcf = dcf**0.5
    elif use_dcf == 4:
        dcf = np.zeros_like(data) 
        print("Attempting k-space preconditoner calculation, as per Ong, et. al.,...")
        for i in range(nphase):
            # mps_tmp = mr.app.JsenseRecon_UPDATED(y=data[i, ..., :nf_e], coord=traj[i, :, :nf_e, :],device=sp.Device(
            #                                     device), img_shape=tshape, 
            #                                      mps_ker_width=14, ksp_calib_width=24, lamda=1e-4).run()
            p = sp.to_device(mr.kspace_precond(
                mps, # mps_tmp,
                coord=sp.to_device(traj[i, ...], device),
                device=sp.Device(device), lamda=1e-3), -1)
            dcf[i, ...] = p # Use all channels here
            del (p)
        dcf = dcf**0.5
    else:
        print("The provided DCF is being used to precondition the objective function.")


    # registration
    print('Motion Field Initialization...')
    # M_fields = []
    # iM_fields = []
    # if reg_flag == 1:
    #     imgL = np.load(os.path.join(fname, 'prL.npy'))
    #     imgL = np.abs(np.squeeze(imgL))
    #     imgL = imgL/np.max(imgL)
    #     for i in range(nphase):
    #         M_field, iM_field = reg.ANTsReg(imgL[n_ref], imgL[i])
    #         M_fields.append(M_field)
    #         iM_fields.append(iM_field)
    #     M_fields = np.asarray(M_fields)
    #     iM_fields = np.asarray(iM_fields)
    #     np.save(os.path.join(fname, '_M_mr.npy'),M_fields)
    #     np.save(os.path.join(fname, '_iM_mr.npy'),iM_fields)
    # else:
    #     M_fields = np.load(os.path.join(fname, '_M_mr.npy'))
    #     iM_fields = np.load(os.path.join(fname, '_iM_mr.npy'))

    # iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
    # M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

    # ######## TODO scale M_field
    # print('Motion Field scaling...')
    # M_fields = [reg.M_scale(M,tshape) for M in M_fields]
    # iM_fields = [reg.M_scale(M,tshape) for M in iM_fields]

    M_fields = np.zeros((nphase,) + tshape + (len(tshape),))
    
    # low rank
    print('Low rank prep...')
    PFTSs = []
    for i in range(nphase):
        FTs = NFTs((nCoil,)+tshape, traj[i, ...], device=sp.Device(device))
        if use_dcf == 4:
            W = sp.linop.Multiply((nCoil, npe, nfe,), dcf[i, :, :, :])
        else:
            # TODO test hypothesis that it is faster to have multiply.Linops with reduced point dimensions
            W = sp.linop.Multiply((nCoil, npe, nfe,), dcf[i, :, :])
                
        if gamma == 0:	        
            FTSs = W*FTs*S	
        else:	
            # Estimate T2* decay	
            t2_star = 1.2  # ms	
            readout = 1.2*res_scale  # ms	
            dwell_time = readout/nfe	
            relaxation = np.zeros((nfe,))	
            for i in range(nfe):	
                relaxation[i] = np.exp(-(i*dwell_time)/t2_star)	
            k = np.reshape(relaxation, [1, 1, nfe])	

            K = sp.linop.Multiply(W.oshape, k**gamma)	
            FTSs = W*K*FTs*S	
            del(K)
        
        
        PFTSs.append(FTSs)
    PFTSs = Diags(PFTSs, oshape=(nphase, nCoil, npe, nfe,),
                  ishape=(nphase,)+tshape)

    if mr_cflag == 1:
        print('With moco...')
        sp.linop.Identity((nphase,)+tshape)
        Ms = []
        # M0s = []
        for i in range(nphase):
            # M = reg.interp_op(tshape,iM_fields[i],M_fields[i])
            M = reg.interp_op(tshape, M_fields[i])
            # M0 = reg.interp_op(tshape,np.zeros(tshape+(3,)))
            M = DLD(M, device=sp.Device(device))
            # M0 = DLD(M0,device=sp.Device(device))
            Ms.append(M)
            # M0s.append(M0)
        Ms = Diags(Ms, oshape=(nphase,)+tshape, ishape=(nphase,)+tshape)
        # M0s = Diags(M0s,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)
        # LRM = GLRA((nphase,)+tshape,lambda_lr,A=Ms)
    else:
        print('Without moco...')
        Ms = sp.linop.Identity((nphase,)+tshape)
        # M0s = sp.linop.Identity((nphase,)+tshape)
    LR = prox.GLRA((nphase,)+tshape, lambda_lr)

    # precondition
    print('Preconditioner calculation...')
    tmp = FTSs.H*FTSs*np.complex64(np.ones(tshape))
    L = np.mean(np.abs(tmp))
    print("Preconditioner: L, using mean of A^N of np.ones(): " + str(L))
    L = sp.app.MaxEig(FTSs.H*FTSs, dtype=np.complex64,
                      device=sp.Device(-1)).run() * 1.01
    # data /= np.linalg.norm(data)
    print("Preconditioner: L, using MaxEig: " + str(L))
    
    # TODO condition number calc
    tmp = np.zeros(tshape)
    tmp[0, 0, 0] = 1.0
    tmp = np.fft.fftshift(tmp)
    tmp = FTSs.H*FTSs*np.complex64(tmp)
    tmp = np.fft.ifftshift(tmp)
    
    print("Data preparation...")
    if use_dcf == 4:
        wdata = data*dcf
    else:
        # TODO: make dcf 4D in all scenarios for robustness
        wdata = data*dcf[:, np.newaxis, :, :]
    del(dcf, data, tmp, traj) # clear from memory to help speed up

    # ADMM
    print('Recon...')
    # Ms is merge
    qt = np.zeros((nphase,)+tshape, dtype=np.complex64)
    u0 = np.zeros_like(qt)
    z0 = np.zeros_like(qt)

    b0 = 1/L*PFTSs.H*wdata
    res_list = []

    del(S, W, FTs)       
    
    # optional - initialize starting Ms # TODO: get a better initial estimate of Ms, before doing the full MoCoLoR
    try:
        b = b0 
        def grad(x): return 1/L*PFTSs.H*PFTSs*x  - b 
        GD_step = sp.alg.GradientMethod(
            grad, qt, .1, accelerate=False, tol=5e-7)  # default false
        for j in range(init_iter):
            tic = time.perf_counter()
            GD_step.update()
            res_norm = GD_step.resid/np.linalg.norm(qt)*GD_step.alpha
            toc = time.perf_counter()
            print('initializing iter:{}, res:{}, {}sec'.format(
                j, res_norm, int(toc - tic)))
            if res_norm < 5e-8:
                break
            res_list.append(res_norm)
            # Save tmp version of recon to view while running
            ni_img = nib.Nifti1Image(abs(np.moveaxis(qt, 0, -1)), affine=aff)
            nib.save(ni_img, fname + '/results/tmp_img_mocolor')
        
    except:
        print("Could not initialize qt.")
    
    # Define ATA(x) for use in conjugate gradient descent
    def ATA(x): return 1/L*PFTSs.H*PFTSs*x + Ms.H*Ms*x
    
    # View convergence
    count = 0
    total_iter = sup_iter * outer_iter * iner_iter
    # img_convergence = np.zeros(
    #     (total_iter, int(recon_resolution), int(recon_resolution), int(recon_resolution)), dtype=float)


    for k in range(sup_iter):
        
        # update motion field
        # print('Registration...')
        M_fields = []
        # iM_fields = []
        if reg_flag == 1:
            imgL = qt
            imgL = np.abs(np.squeeze(imgL))
            imgL = imgL/np.max(imgL)
            for i in range(nphase):
                M_field, iM_field = reg.ANTsReg(imgL[n_ref], imgL[i])
                M_fields.append(M_field)
                # iM_fields.append(iM_field)
            M_fields = np.asarray(M_fields)
            # iM_fields = np.asarray(iM_fields)
            # np.save(os.path.join(fname, '_M_mr.npy'), M_fields) # JWP do not save
            # np.save(os.path.join(fname, '_iM_mr.npy'),iM_fields)
        else:
            M_fields = np.load(os.path.join(fname, '_M_mr.npy'))
            # iM_fields = np.load(os.path.join(fname, '_iM_mr.npy'))

        # iM_fields = [iM_fields[i] for i in range(iM_fields.shape[0])]
        M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

        # TODO scale M_field to np array
        # print('Motion Field scaling...')
        M_fields = [reg.M_scale(M, tshape) for M in M_fields]
        # iM_fields = [reg.M_scale(M,tshape) for M in iM_fields]

        # print('With moco...')
        sp.linop.Identity((nphase,)+tshape)
        Ms = []
        # M0s = []
        for i in range(nphase):
            # M = reg.interp_op(tshape,iM_fields[i],M_fields[i])
            M = reg.interp_op(tshape, M_fields[i])
            # M0 = reg.interp_op(tshape,np.zeros(tshape+(3,)))
            M = DLD(M, device=sp.Device(device))
            # M0 = DLD(M0,device=sp.Device(device))
            Ms.append(M)
            # M0s.append(M0)
        Ms = Diags(Ms, oshape=(nphase,)+tshape, ishape=(nphase,)+tshape)
        # M0s = Diags(M0s,oshape=(nphase,)+tshape,ishape=(nphase,)+tshape)
        
        try:
            del (M, M_fields, imgL, iM_field, M_field)
        except:
            print("Could not delete motion variables to save space. Reconstruction speed may be impacted.")

        
        for i in range(outer_iter): 
            b = b0 + rho*Ms.H*(z0 - u0) 
            if method == "cg":
                CG_step = sp.alg.ConjugateGradient(
                    ATA, b, qt, max_iter=iner_iter, tol=1e-7)
                for j in range(iner_iter):
                    tic = time.perf_counter()
                    CG_step.update()
                    res_norm = CG_step.resid/np.linalg.norm(qt)*CG_step.alpha
                    toc = time.perf_counter()
                    print('superior iter:{}, outer iter:{}, inner iter:{}, res:{}, {}sec'.format(
                        k, i, j, res_norm, int(toc - tic)))
                    if res_norm < 5e-8:
                        break
                    res_list.append(res_norm)

                    # img_convergence[count, ...] = np.abs(
                    #     np.squeeze(qt))[nphase//2, :, :, :]  # Middle resp phase only
                    # count += 1
                    
                    # Save tmp version of recon to view while running
                    ni_img = nib.Nifti1Image(abs(np.moveaxis(qt, 0, -1)), affine=aff)
                    nib.save(ni_img, fname + '/results/tmp_img_mocolor')
                    
            elif method == "gm":
                # def grad(x): 1/L*PFTSs.H*PFTSs*x + rho*Ms.H*Ms*x - b # TODO: fix this bug!!
                def grad(x): return 1/L*PFTSs.H*PFTSs*x + rho*x - b 
                GD_step = sp.alg.GradientMethod(
                    grad, qt, .1, accelerate=False, tol=5e-7)  # default false
                for j in range(iner_iter):
                    tic = time.perf_counter()
                    GD_step.update()
                    # qt = qt - 0.2*(1/L*PFTSs.H*(PFTSs*qt - wdata) + Ms.H*(Ms*qt - z0 + u0)) # Manual method
                    res_norm = GD_step.resid/np.linalg.norm(qt)*GD_step.alpha
                    toc = time.perf_counter()
                    print('superior iter:{}, outer iter:{}, inner iter:{}, res:{}, {}sec'.format(
                        k, i, j, res_norm, int(toc - tic)))
                    if res_norm < 5e-8:
                        break
                    res_list.append(res_norm)

                    # img_convergence[count, ...] = np.abs(
                    #     np.squeeze(qt))[nphase//2, :, :, :]  # Middle resp phase only
                    # count += 1
                    
                    # Save tmp version of recon to view while running
                    ni_img = nib.Nifti1Image(abs(np.moveaxis(qt, 0, -1)), affine=aff)
                    nib.save(ni_img, fname + '/results/tmp_img_mocolor')
            else:
                 print("Improper convex optimization method selected.")
                    
            z0 = np.complex64(LR(1, Ms*qt + u0))
            u0 = u0 + (Ms*qt - z0)
            
        # np.save(os.path.join(fname, 'mocolor_vent.npy'), qt)
        # np.save(os.path.join(fname, 'mocolor_vent_residual.npy'),
        #         np.asarray(res_list))

    # qt = np.load(os.path.join(fname, 'mocolor_vent.npy'))

    # ni_img = nib.Nifti1Image(
    #     abs(np.moveaxis(img_convergence, 0, -1)), affine=aff)
    # nib.save(ni_img, fname + '/results/img_convergence_' + str(nphase) +
    #          '_bin_' + str(int(recon_resolution)) + '_resolution')
    
    # Remove temporary image file for storage purposes
    try:
        os.remove(fname + '/results/tmp_img_mocolor.nii')
    except:
        print("Could not remove tmp image file.")

    try: 
        nifti_filename = str(method) + '_' + str(nphase) + '_bin_' + str(field_of_view) + 'mm_FOV_' + str(int(recon_voxel_size)) + 'mm_recon_resolution'
    except:
        nifti_filename = str(nphase) + '_bin_' + str(int(recon_resolution)) + '_recon_matrix_size'

    ni_img = nib.Nifti1Image(abs(np.moveaxis(qt, 0, -1)), affine=aff)
    nib.save(ni_img, fname + '/results/img_mocolor_' + nifti_filename)

    # nphase = 6
    # jacobian determinant & specific ventilation
    if vent_flag == 1:
        tic = time.perf_counter()
        print('Jacobian Determinant and Specific Ventilation...')
        jacs = []
        svs = []
        qt = np.abs(np.squeeze(qt))
        qt = qt/np.max(qt)
        for i in range(nphase):
            jac, sv = reg.ANTsJac(np.abs(qt[n_ref]), np.abs(qt[i]))
            jacs.append(jac)
            svs.append(sv)
            print('ANTsJac computation completed for phase: ' + str(i))
        jacs = np.asarray(jacs)
        svs = np.asarray(svs)
        # np.save(os.path.join(fname, 'jac_mocolor_vent.npy'), jacs)
        # np.save(os.path.join(fname, 'sv_mocolor_vent.npy'), svs)
        toc = time.perf_counter()
        print('time elapsed for ventilation metrics: {}sec'.format(int(toc - tic)))

        ni_img = nib.Nifti1Image(np.moveaxis(svs, 0, -1), affine=aff)
        nib.save(ni_img, fname + '/results/sv_mocolor_' + nifti_filename)

        ni_img = nib.Nifti1Image(np.moveaxis(jacs, 0, -1), affine=aff)
        nib.save(ni_img, fname + '/results/jacs_mocolor_' + nifti_filename)

    toc_total = time.perf_counter()
    print('total time elapsed: {}mins'.format(int(toc_total - tic_total)/60))
