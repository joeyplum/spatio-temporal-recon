#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: plummerjw
"""

import argparse

import sigpy as sp
import scipy.ndimage as ndimage_c
import numpy as np

import sys
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
from functions import util

if __name__ == '__main__':

    # IO parameters
    parser = argparse.ArgumentParser(
        description='motion compensated low rank constrained recon.')

    parser.add_argument('--use_dcf', type=float, default=1,
                        help='use DCF on objective function, yes == 1')
    parser.add_argument('--binned_csm', type=float, default=1,
                        help='calculate a sensitivity map for each bin, yes == 1')
    parser.add_argument('--gamma', type=float, default=0,
                        help='T2* weighting in Fourier encoding operator. Default == 0, full weighting == 1.')

    parser.add_argument('--res_scale', type=float, default=1,
                        help='scale of resolution, full res == 1')

    parser.add_argument('--fov_x', type=float, default=160,
                        help='scale of FOV x, full res == 160')
    parser.add_argument('--fov_y', type=float, default=160,
                        help='scale of FOV y, full res == 160')
    parser.add_argument('--fov_z', type=float, default=160,
                        help='scale of FOV z, full res == 160')
    parser.add_argument('--crop_x', type=int, default=160,
                        help='x matrix size to be cropped to, default == 160')
    parser.add_argument('--crop_y', type=int, default=160,
                        help='y matrix size to be cropped to, default == 160')
    parser.add_argument('--crop_z', type=int, default=160,
                        help='z matrix size to be cropped to, default == 160')

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
    parser.add_argument('--init_iter', type=int, default=5,
                        help='Num of initialization iterations (with no moco component).')
    parser.add_argument('--iner_iter', type=int, default=5,
                        help='Num of inner iterations.')
    parser.add_argument('--outer_iter', type=int, default=3,
                        help='Num of outer iterations.')
    parser.add_argument('--sup_iter', type=int, default=3,
                        help='Num of superior iterations.')
    parser.add_argument('--method', type=str, default='gm',
                        help='Iterative method for inner loop (cg or gm, default = gm).')

    parser.add_argument('--device', type=int, default=0,
                        help='Computing device.')   
    parser.add_argument('fname', type=str, 
                        help='Prefix of raw data and output(_mocolor).')
 
    # a set of CFL files, including(kspace, trajectory, and density_compensation_weighting)
    args = parser.parse_args()

    #
    use_dcf = args.use_dcf
    binned_csm = args.binned_csm
    gamma = args.gamma
    res_scale = args.res_scale
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
    crop_x = int(args.crop_x * res_scale)
    crop_y = int(args.crop_y * res_scale)
    crop_z = int(args.crop_z * res_scale)
    n_ref = args.n_ref
    reg_flag = args.reg_flag
    mr_cflag = args.mr_cflag
    vent_flag = args.vent_flag

    print('Reconstruction started...')
    tic_total = time.perf_counter()   
        
    # data loading
    print("Loading data from " + fname)
    data = np.load(os.path.join(fname, 'bksp.npy'))
    traj = np.real(np.load(os.path.join(fname, 'bcoord.npy')))
    try:
        dcf = np.sqrt(np.load(os.path.join(fname, 'bdcf_pipemenon.npy')))
        print("Philips DCF used.")
    except:
        dcf = np.zeros((data.shape[0], data.shape[2], data.shape[3]), dtype=complex)
        print("No dcf located.")

    # Robust for 3D isotropic
    nf_scale = res_scale
    nf_arr = np.sqrt(np.sum(traj[0, 0, :, :]**2, axis=1)) 
    nf_e = np.sum(nf_arr < np.max(nf_arr)*nf_scale)
    print(f"Subsetting first {100*res_scale:.1f}% nf_e") # Note, this is not robust for VDS where the edges are more sampled than center
    traj[..., 0] = traj[..., 0]*fov_scale[0]
    traj[..., 1] = traj[..., 1]*fov_scale[1]
    traj[..., 2] = traj[..., 2]*fov_scale[2]

    # Optional: undersample along freq encoding - JWP 20230815
    print("Number of frequency encodes before trimming: " + str(data.shape[-1]))
    traj = traj[..., :nf_e, :]
    data = data[..., :nf_e]
    dcf = dcf[..., :nf_e]
    
    nphase, nCoil, npe, nfe = data.shape
    tshape = (int(np.max(traj[..., 0])-np.min(traj[..., 0])), int(np.max(
        traj[..., 1])-np.min(traj[..., 1])), int(np.max(traj[..., 2])-np.min(traj[..., 2])))
    # Or use manual input settings
    tshape = (int(fov_scale[0] * res_scale),
              int(fov_scale[1] * res_scale),
              int(fov_scale[2] * res_scale))  
    acq_vol = (crop_x, crop_y, crop_z)  # Assume that the cropped matrix is the acq. matrix. This is used for DCF, etc.

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
    # Custom affine: this will depend on if data is measured coronally, axially, etc
    aff = np.array([[0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1]])
    print("Affine transformation matrix used for saving this data: ")
    print(str(aff))


    print('Density compensation...')
    if use_dcf == 0:
        dcf = np.ones_like(dcf)
        print("DCF will not be used to precondition the objective function.")
    elif use_dcf == 2:
        print(
            "A new DCF will be calculated based on the coordinate trajectories and image shape. ")
        for i in range(nphase):
            dcf[i, ...] = sp.to_device(mr.pipe_menon_dcf(traj[i, ...], img_shape=acq_vol,
                                                         device=sp.Device(0)), -1)
        dcf /= np.max(dcf)
        # np.save(fname + "bdcf_pipemenon.npy", dcf)
        dcf = dcf**0.5
    elif use_dcf == 3:
        dcf = np.zeros_like(dcf)
        print("Attempting k-space preconditoner calculation, as per Ong, et. al.,...")
        for i in range(nphase):
            # Approximate using a single channel precond for now (helps for speed)
            ones = np.ones(((nCoil,) + acq_vol))
            ones /= nCoil**0.5
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
        # TODO: this recalculates multiple sense maps, inefficient!!
        mps_binned = np.zeros((nphase, nCoil, ) + (tshape), dtype=complex)
        for i in range(nphase):
            mps_binned[i, ...] = mr.app.JsenseRecon(y=data[i, ..., :nf_e], coord=traj[i, :, :nf_e, :], device=sp.Device(
                device), img_shape=tshape, mps_ker_width=8, ksp_calib_width=24, lamda=1e-4, 
                                                max_inner_iter=10, max_iter=10).run()
            dcf[i, ...] = sp.to_device(mr.kspace_precond(
                mps_binned[i, ...],
                coord=sp.to_device(traj[i, ...], device),
                device=sp.Device(device), lamda=1e-3), -1) # Use all channels here
        dcf = dcf**0.5
    else:
        print("The provided DCF is being used to precondition the objective function.")
        
    # data loading for sens map
    if binned_csm==1:
        if use_dcf != 4:
            mps_binned = np.zeros((nphase, nCoil, ) + (tshape), dtype=complex)
            print('Calculating a binned sensitivity map from binned data')
            for i in range(nphase):
                mps_binned[i, ...] = mr.app.JsenseRecon(y=data[i, ..., :nf_e], coord=traj[i, :, :nf_e, :], device=sp.Device(
                    device), img_shape=tshape, mps_ker_width=8, ksp_calib_width=24, lamda=1e-4, 
                                                    max_inner_iter=20, max_iter=20).run()
        else:
            print("Binned sensitivity maps already calculated for the use_dcf == 4 preconditioner.")
            
    else:
        try:
            print("Calculating sensitivity map from raw (unbinned) data...")
            ksp = np.load(fname + "ksp.npy")
            print("ksp.shape = " + str(np.shape(ksp)))
            coord = np.load(fname + "coord.npy")
            coord[..., 0] = coord[..., 0]*fov_scale[0]
            coord[..., 1] = coord[..., 1]*fov_scale[1]
            coord[..., 2] = coord[..., 2]*fov_scale[2] 
            mps = mr.app.JsenseRecon(y=ksp[..., :nf_e], coord=coord[:, :nf_e, :], device=sp.Device(
                    device), img_shape=tshape, mps_ker_width=18, ksp_calib_width=24, lamda=1e-4, 
                                                    max_inner_iter=10, max_iter=10).run()
        
            del(ksp, coord)
            S = sp.linop.Multiply(tshape, mps)
            print("Success.")
        except:
            # calibration
            print("Failed.")
            print('Calculating sensitivity map from binned data, after merging into one bin, instead...')
            ksp = np.reshape(np.transpose(data, (1, 0, 2, 3)),
                            (nCoil, nphase*npe, nfe))
            dcf2 = np.reshape(dcf**2, (nphase*npe, nfe))
            dcf_jsense = dcf2  
            
            # Default
            coord = np.reshape(traj, (nphase*npe, nfe, 3))
            mps = ext.jsens_calib(ksp[..., :nf_e], coord[:, :nf_e, :], dcf_jsense[..., :nf_e], device=sp.Device(
                device), ishape=tshape, mps_ker_width=8, ksp_calib_width=16)
            del(dcf_jsense, dcf2, coord, ksp)
        
    # Visualize estimated sensitivity maps    
    # try:
    #     import sigpy.plot as pl
    #     pl.ImagePlot(mps_binned[-1,...], x=1, y=2, z=0,
    #                 title="Estimated sensitivity maps", save_basename=fname + '/results/csm_espirit.png')
    # except:
    #     print("Could not show the sensitivity maps.")
    
    
    print('Motion Field Initialization...')

    M_fields = np.zeros((nphase,) + tshape + (len(tshape),))
    
    def variable_te(kz, kmax, TE_min, TE_max):
        m = (TE_max - TE_min) / kmax
        b = TE_min
        return m * kz + b
    
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
        
        if binned_csm==1 or use_dcf==4:
            S = sp.linop.Multiply(tshape, mps_binned[i, ...])
        else:
            S = sp.linop.Multiply(tshape, mps)
                
        if gamma == 0:	        
            FTSs = W*FTs*S	
        else:	
            # Estimate T2* decay	# TODO: test and optimize (future paper?)
            t2_star = 9  # ms	
            readout = 5*res_scale  # ms	
            dwell_time = readout/nfe	
            relaxation = np.zeros((nfe,))	
            for jj in range(nfe):	
                relaxation[jj] = np.exp(-(jj*dwell_time)/t2_star)	
            k = np.reshape(relaxation, [1, 1, nfe])	
            K = sp.linop.Multiply(W.oshape, k**gamma)
            FTSs = W*K*FTs*S	
            del(K)
        
        
        PFTSs.append(FTSs)
    PFTSs = Diags(PFTSs, oshape=(nphase, nCoil, npe, nfe,),
                  ishape=(nphase,)+tshape)
    
    if use_dcf == 4 or binned_csm==1:
        tic = time.perf_counter()
        del(mps_binned)
        toc = time.perf_counter()
        print(f'Time to delete mps_binned: {int(toc-tic)}sec')

    if mr_cflag == 1:
        print('With moco...')
        sp.linop.Identity((nphase,)+tshape)
        Ms = []
        # M0s = []
        for i in range(nphase):
            M = reg.interp_op(tshape, M_fields[i])
            M = DLD(M, device=sp.Device(device))
            Ms.append(M)
        Ms = Diags(Ms, oshape=(nphase,)+tshape, ishape=(nphase,)+tshape)
    else:
        print('Without moco...')
        Ms = sp.linop.Identity((nphase,)+tshape)
    LR = prox.GLRA((nphase,)+tshape, lambda_lr)

    # precondition
    print('Preconditioner calculation...')
    tmp = FTSs.H*FTSs*np.complex64(np.ones(tshape))
    L = np.mean(np.abs(tmp))
    print("Preconditioner: L, using mean of A^N of np.ones(): " + str(L))
    L = sp.app.MaxEig(FTSs.H*FTSs, dtype=np.complex64,
                      device=sp.Device(-1)).run() * 1.01
    del (FTSs, tmp)
    data /= np.linalg.norm(data)
    print("Preconditioner: L, using MaxEig: " + str(L))
    
    print("Data preparation...")
    if use_dcf == 4:
        wdata = data*dcf
    else:
        # TODO: make dcf 4D in all scenarios for robustness
        wdata = data*dcf[:, np.newaxis, :, :]
    del(dcf, data, traj) # clear from memory to help speed up

    # ADMM
    print('Recon...')
    # Ms is merge
    qt = np.zeros((nphase,)+tshape, dtype=np.complex64)
    u0 = np.zeros_like(qt)
    z0 = np.zeros_like(qt)

    b0 = 1/L*PFTSs.H*wdata
    del (wdata)
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
            ni_img = nib.Nifti1Image(abs(np.moveaxis(util.crop(qt, crop_xy=crop_x, crop_z=crop_z), 0, -1)), affine=aff)
            nib.save(ni_img, fname + '/results/tmp_img_mocolor')
            del (ni_img)
        
    except:
        print("Could not initialize qt.")
    
    # Define ATA(x) for use in conjugate gradient descent
    def ATA(x): return 1/L*PFTSs.H*PFTSs*x + Ms.H*Ms*x
    
    # View convergence
    count = 0
    total_iter = sup_iter * outer_iter * iner_iter
    img_convergence = np.zeros(
        (total_iter, )+tshape, dtype=float)

    # Set up a registration mask (optional)
    use_reg_mask = False
    if use_reg_mask:
        inner_mask = np.ones(tshape, dtype=float)
        pad_x = int((fov_scale[0] - crop_x)/2)
        pad_y = int((fov_scale[1] - crop_y)/2)
        pad_z = int((fov_scale[2] - crop_z)/2)
        reg_mask = np.pad(inner_mask, pad_width=((pad_x, pad_x), (pad_y, pad_y), (pad_z, pad_z)), mode='constant', constant_values=0)

    for k in range(sup_iter):
        tic = time.perf_counter()

        # update motion field
        print('Registration...')
        M_fields = []
        if reg_flag == 1:
            imgL = qt
            imgL = np.abs(np.squeeze(imgL))
            imgL = imgL/np.max(imgL)
            for i in range(nphase):
                if use_reg_mask:
                    M_field, iM_field = reg.ANTsReg(imgL[n_ref], imgL[i], mask=reg_mask)
                else:
                    M_field, iM_field = reg.ANTsReg(imgL[n_ref], imgL[i])
                M_fields.append(M_field)
                del (M_field, iM_field)
            del (imgL)
            M_fields = np.asarray(M_fields)
            np.save(os.path.join(fname, '_M_mr.npy'), M_fields) 
        else:
            M_fields = np.load(os.path.join(fname, '_M_mr.npy'))

        M_fields = [M_fields[i] for i in range(M_fields.shape[0])]

        print('Motion Field scaling...')
        M_fields = [reg.M_scale(M, tshape) for M in M_fields]
        
        # Save motion fields to view while running
        ni_img = nib.Nifti1Image(abs(np.moveaxis(M_fields, 0, -1)), affine=aff)
        nib.save(ni_img, fname + '/results/M_fields')

        # print('With moco...')
        sp.linop.Identity((nphase,)+tshape)
        Ms = []
        for i in range(nphase):
            M = reg.interp_op(tshape, M_fields[i])
            M = DLD(M, device=sp.Device(device))
            Ms.append(M)
        Ms = Diags(Ms, oshape=(nphase,)+tshape, ishape=(nphase,)+tshape)
        
        try:
            del (M, M_fields)
        except:
            print("Could not delete motion variables to save space. Reconstruction speed may be impacted.")
        toc = time.perf_counter()
        print(f'Total motion field measurement time: {int(toc-tic)}sec')
        
        # Save tmp version of recon to view while running
        ni_img = nib.Nifti1Image(abs(np.moveaxis(util.crop(Ms*qt, crop_xy=crop_x, crop_z=crop_z), 0, -1)), affine=aff)
        nib.save(ni_img, fname + '/results/Ms_qt_img_mocolor')
        del (ni_img)

        
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

                    img_convergence[count, ...] = np.abs(
                        np.squeeze(qt))[nphase//2, :, :, :]  # Middle resp phase only
                    count += 1
                    
                    # Save tmp version of recon to view while running
                    ni_img = nib.Nifti1Image(abs(np.moveaxis(util.crop(qt, crop_xy=crop_x, crop_z=crop_z), 0, -1)), affine=aff)
                    nib.save(ni_img, fname + '/results/tmp_img_mocolor')
                    del (ni_img)
                    
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

                    img_convergence[count, ...] = np.abs(
                        np.squeeze(qt))[nphase//2, :, :, :]  # Middle resp phase only
                    count += 1
                    
                    # Save tmp version of recon to view while running
                    ni_img = nib.Nifti1Image(abs(np.moveaxis(util.crop(qt, crop_xy=crop_x, crop_z=crop_z), 0, -1)), affine=aff)
                    nib.save(ni_img, fname + '/results/tmp_img_mocolor')
                    del (ni_img)
            else:
                 print("Improper convex optimization method selected.")
                    
            z0 = np.complex64(LR(1, Ms*qt + u0))
            u0 = u0 + (Ms*qt - z0)
            


    ni_img = nib.Nifti1Image(
        abs(np.moveaxis(img_convergence, 0, -1)), affine=aff)
    nib.save(ni_img, fname + '/results/img_convergence_' + str(nphase) +
             '_bin')
    
    # Remove temporary image file for storage purposes
    try:
        os.remove(fname + '/results/tmp_img_mocolor.nii')
    except:
        print("Could not remove tmp image file.")

    nifti_filename = str(nphase) + '_bin_' + str(int(crop_x)) + '_recon_matrix_size'

    ni_img = nib.Nifti1Image(abs(np.moveaxis(util.crop(qt, crop_xy=crop_x, crop_z=crop_z), 0, -1)), affine=aff)
    nib.save(ni_img, fname + '/results/img_mocolor_' + nifti_filename)
    del (ni_img)

    # jacobian determinant & specific ventilation
    if vent_flag == 1:
        tic = time.perf_counter()
        print('Jacobian Determinant and Specific Ventilation...')
        jacs = []
        svs = []
        qt = util.crop(qt, crop_xy=crop_x, crop_z=crop_z)
        qt = np.abs(np.squeeze(qt))
        qt = qt/np.max(qt)
        for i in range(nphase):
            jac, sv = reg.ANTsJac(np.abs(qt[n_ref]), np.abs(qt[i]))
            jacs.append(jac)
            svs.append(sv)
            print('ANTsJac computation completed for phase: ' + str(i))
        jacs = np.asarray(jacs)
        svs = np.asarray(svs)
        toc = time.perf_counter()
        print('time elapsed for ventilation metrics: {}sec'.format(int(toc - tic)))

        ni_img = nib.Nifti1Image(np.moveaxis(svs, 0, -1), affine=aff)
        nib.save(ni_img, fname + '/results/sv_mocolor_' + nifti_filename)

        ni_img = nib.Nifti1Image(np.moveaxis(jacs, 0, -1), affine=aff)
        nib.save(ni_img, fname + '/results/jacs_mocolor_' + nifti_filename)

    toc_total = time.perf_counter()
    print('total time elapsed: {}mins'.format(int(toc_total - tic_total)/60))
