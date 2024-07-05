from sigpy_e.linop_e import NFTs, Diags, DLD, Vstacks
import sigpy_e.cfl as cfl
import nibabel as nib
import logging
import os
import sigpy.mri as mr
import sigpy_e.reg as reg
import sigpy_e.ext as ext
import argparse
import sigpy as sp
import scipy.ndimage as ndimage_c
import numpy as np
import time


try:
    import ReadPhilips.readphilips as rp
    from ReadPhilips.readphilips.file_io import io
    import csv
    automate_FOV = True
except:
    print("Could not load ReadPhilips script.")
    automate_FOV = False

import sys
sys.path.append("./sigpy_e/")

# IO parameters
parser = argparse.ArgumentParser(description='NUFFT recon.')

parser.add_argument('--res_scale', type=float, default=.75,
                    help='scale of resolution, full res == .75')
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

parser.add_argument('--lambda_TV', type=float, default=0,
                    help='TV regularization, default is 0.05')
parser.add_argument('--outer_iter', type=int, default=1,
                    help='Num of Iterations.')

parser.add_argument('--vent_flag', type=int, default=0,
                    help='output jacobian determinant and specific ventilation')
parser.add_argument('--n_ref_vent', type=int, default=0,
                    help='reference frame for ventilation')

parser.add_argument('--device', type=int, default=0,
                    help='Computing device.')

parser.add_argument('fname', type=str,
                    help='Prefix of raw data and output(_mrL).')
args = parser.parse_args()

#
res_scale = args.res_scale
scan_resolution = args.scan_res
recon_resolution = args.recon_res
fname = args.fname
lambda_TV = args.lambda_TV
device = args.device
outer_iter = args.outer_iter
fov_scale = (args.fov_x, args.fov_y, args.fov_z)
vent_flag = args.vent_flag
n_ref_vent = args.n_ref_vent

print('Reconstruction started.')
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
    dcf = np.zeros((data.shape[0], data.shape[2], data.shape[3]))
    print("No dcf located.")

nf_scale = res_scale
nf_arr = np.sqrt(np.sum(traj[0, 0, :, :]**2, axis=1))
nf_e = np.sum(nf_arr < np.max(nf_arr)*nf_scale)
scale = (scan_resolution, scan_resolution, scan_resolution)  # Added JWP
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
    recon_resolution), int(recon_resolution))

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

# data loading for sens map
try:
    print("Calculating sensitivity map from raw (unbinned) data...")
    ksp = np.load(fname + "ksp.npy")
    ksp = np.reshape(ksp, (np.shape(ksp)[0], np.shape(ksp)[
    1]*np.shape(ksp)[2], np.shape(ksp)[3]))[..., :nf_e]
    print(np.shape(ksp))
    coord = np.load(fname + "coord.npy")*scale[0]
    coord = coord.reshape(
    (np.shape(coord)[0]*np.shape(coord)[1], np.shape(coord)[2], np.shape(coord)[3]))[:, :nf_e, :]
    dcf_jsense = np.load(fname + "dcf.npy")
    dcf_jsense = dcf_jsense.reshape((np.shape(dcf_jsense)[0] * np.shape(dcf_jsense)[1], np.shape(dcf_jsense)[2]))[..., :nfe]
    mps = ext.jsens_calib(ksp[..., :nf_e], coord[:, :nf_e, :], dcf_jsense[..., :nf_e], device=sp.Device(
        device), ishape=tshape, mps_ker_width=8, ksp_calib_width=16)
    # mps = mr.app.JsenseRecon_UPDATED(y=ksp[..., :nf_e], coord=coord[:, :nf_e, :], device=sp.Device(
    #     device), img_shape=tshape, mps_ker_width=14, ksp_calib_width=24, lamda=1e-4, 
    #                                     max_inner_iter=20, max_iter=20).run()
    del(dcf_jsense, ksp, coord)
    S = sp.linop.Multiply(tshape, mps)
    print("Success.")
except:
    # calibration
    print('Calculating sensitivity map from binned data...')
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
    # TODO: see if you get improved results without dcf (may not be possible on old sigpy)
    # mps = mr.app.JsenseRecon_UPDATED(y=ksp[..., :nf_e], coord=coord[:, :nf_e, :], device=sp.Device(
    #     device), img_shape=tshape, mps_ker_width=14, ksp_calib_width=24, lamda=1e-4).run()
    del(dcf_jsense, dcf2, coord, ksp)
    S = sp.linop.Multiply(tshape, mps)
    # S = sp.linop.Multiply(tshape, np.ones((1,)+tshape)) # ONES


# recon
q2 = np.zeros((nphase,)+tshape, dtype=np.complex64)
    
def mvd(x): return sp.to_device(x, device)
def mvc(x): return sp.to_device(x, sp.cpu_device)

for i in range(nphase):
    # Move to GPU
    traj_tmp = mvd(traj[i,...])
    data_tmp = mvd(data[i,...])
    dcf_tmp = mvd(dcf[i,...]) # Input data is already in sqrt form
    mps = mvd(mps)

    # Compute linear operators
    S = sp.linop.Multiply(tshape, mps)
    F = sp.linop.NUFFT(mps.shape,
                    coord=traj_tmp,
                    oversamp=1.25,
                    width=3)
    D = sp.linop.Multiply(F.oshape, dcf_tmp)
    
    # Compute a single x = A.H b operation (i.e. inverse NUFFT)
    A_dcf = D * F * S
    b_dcf = data_tmp * dcf_tmp / np.linalg.norm(data_tmp * dcf_tmp)
    q2[i,...] = mvc(abs(A_dcf.H * b_dcf))

q2 = np.abs(np.squeeze(q2))
q2 = q2/np.max(q2)

try: 
    nifti_filename = str(nphase) + '_bin_' + str(field_of_view) + 'mm_FOV_' + str(int(recon_voxel_size)) + 'mm_recon_resolution'
except:
    nifti_filename = str(nphase) + '_bin_' + str(int(recon_resolution)) + '_recon_matrix_size'

ni_img = nib.Nifti1Image(abs(np.moveaxis(q2, 0, -1)), affine=aff)
nib.save(ni_img, fname + '/results/img_nufft_' + nifti_filename)

# jacobian determinant & specific ventilation
if vent_flag == 1:
    tic = time.perf_counter()
    print('Jacobian Determinant and Specific Ventilation...')
    jacs = []
    svs = []
    
    for i in range(nphase):
        jac, sv = reg.ANTsJac(np.abs(q2[n_ref_vent]), np.abs(q2[i]))
        jacs.append(jac)
        svs.append(sv)
        print('ANTsJac computation completed for phase: ' + str(i))
    jacs = np.asarray(jacs)
    svs = np.asarray(svs)
    # np.save(os.path.join(fname, 'jac_nufft.npy'), jacs)
    # np.save(os.path.join(fname, 'sv_nufft.npy'), svs)
    toc = time.perf_counter()
    print('time elapsed for ventilation metrics: {}sec'.format(int(toc - tic)))


if vent_flag == 1:
    ni_img = nib.Nifti1Image(np.moveaxis(svs, 0, -1), affine=aff)
    nib.save(ni_img, fname + '/results/sv_nufft_' +
                str(nphase) + '_bin_' + str(int(recon_resolution)) + '_resolution')

    ni_img = nib.Nifti1Image(np.moveaxis(jacs, 0, -1), affine=aff)
    nib.save(ni_img, fname + '/results/jacs_nufft_' + str(nphase) +
                '_bin_' + str(int(recon_resolution)) + '_resolution')

toc_total = time.perf_counter()
print('total time elapsed: {}mins'.format(int(toc_total - tic_total)/60))
