import sigpy as sp
import numpy as np
import os
import glob
import nibabel
from sigpy.linop import Linop
from sigpy import backend
import scipy.ndimage as ndimage
from scipy.io import loadmat
from scipy import linalg
import ants
import matplotlib.pyplot as plt

__all__ = ['interp_op', 'interp', 'ANTsReg', 'ANTsAff', 'interp_affine_op']


def M_scale2(M, oshape, scale=1):
    Mscale = [oshape[i]/M.shape[i+1] for i in range(M.shape[0])]
    Mo = np.zeros((M.shape[0],)+oshape)
    for i in range(M.shape[0]):
        M[i] = M[i]*(Mscale[i]*scale)
        Mo[i] = ndimage.zoom(M[i], zoom=tuple(Mscale), order=1)

    return Mo


def M_scale(M, oshape, scale=1):
    Mscale = [oshape[i]/M.shape[i] for i in range(M.shape[-1])]
    Mo = np.zeros(oshape+(M.shape[-1],))
    for i in range(M.shape[-1]):
        M[..., i] = M[..., i]*(Mscale[i]*scale)
        Mo[..., i] = ndimage.zoom(M[..., i], zoom=tuple(Mscale), order=2)

    return Mo


def ANTsAff(If, Im, vox_res=[1, 1, 1], reg_level=[8, 4, 2], gauss_filt=[2, 2, 1]):
    # transfer to nifti
    Ifnft = nibabel.Nifti1Image(If, affine=np.diag(vox_res+[1]))
    Imnft = nibabel.Nifti1Image(Im, affine=np.diag(vox_res+[1]))

    nibabel.save(Ifnft, './tmp_If.nii')
    nibabel.save(Imnft, './tmp_Im.nii')

    reg_level_s = 'x'.join([str(t) for t in reg_level])
    gauss_filt_s = 'x'.join([str(t) for t in gauss_filt])

    ants_cmd = 'antsRegistration -d 3 -m MI[ {}, {}, 1, 50 ] -t Rigid[0.1] \
    -c [ 100x100x40, 1e-6, 10 ] -s {}vox -f {} --winsorize-image-intensities [0.1,1]\
    -l 1 -u 1 -z 1 -v -o tmp_'.format('tmp_Im.nii', 'tmp_If.nii', gauss_filt_s, reg_level_s)
    os.system(ants_cmd)
    x = loadmat('./tmp_0GenericAffine.mat')
    T = x['AffineTransform_double_3_3'].reshape([4, 3])

    # ANTs orientation
    M_rot = [[1, -1, 1], [-1, 1, 1], [1, 1, 1], [1, 1, -1]]
    T = T*M_rot
    T[3, ...] = T[3, ...].dot(linalg.inv(T[:3]))

    return T


class interp_affine_op(Linop):
    def __init__(self, ishape, T):
        assert list(T.shape) == [4, 3], "Tmatrix Dimension mismatch!"
        oshape = ishape
        self.T = T
        super().__init__(oshape, ishape)

    def _apply(self, input):
        return interp_affine(input, self.T)

    def _adjoint_linop(self):
        T = self._aff_inversion(self.T)

        return interp_affine_op(self.ishape, T)

    def _aff_inversion(self, T):
        T_inv = np.zeros_like(T)
        T_inv[:3, :] = np.linalg.inv(T[:3, :])
        T_inv[3, :] = -T[3, :].dot(T[:3, :].transpose())
        return T_inv


def interp_affine(I, T, aff_order=1):
    # T should be [4,3], [:3,3] rotation, [3,:] shift
    shift_before_rot = T[3, :]
    shift_after_rot = shift_before_rot.dot(T[:3, :].transpose())
    shift_after_rot = -T[3, :]
    def AT(x): return ndimage.affine_transform(
        x, T[:3, :], offset=-shift_after_rot, order=aff_order)
    if np.iscomplexobj(I) is True:
        I_aff = AT(np.real(I)) + 1j * AT(np.imag(I))
    else:
        I_aff = AT(I)

    return I_aff


def ANTsReg4(Is, ref=0):
    M_fields = []
    iM_fields = []
    nphase = len(Is)
    for i in range(nphase):
        M_field, iM_field = ANTsReg(np.abs(Is[2]), np.abs(Is[i]))

        M_fields.append(M_field)
        iM_fields.append(iM_field)
    # change
    np.save('./M_field.npy', np.asarray(M_fields))
    np.save('./iM_field.npy', np.asarray(iM_fields))


def ANTsReg(If, Im, vox_res=[1, 1, 1], reg_level=[8, 4, 2], gauss_filt=[2, 2, 1]):

    fixed = ants.from_numpy(If) # TODO: check, are these the wrong way round??
    moving = ants.from_numpy(Im)

    tmp_dir = 'tmp{}_'.format(np.random.randint(0, 1e4))

    # default
    registration_params =   {'type_of_transform':'SyNOnly',
                            'syn_metric':'demons', 
                            'syn_sampling':4,
                            'grad_step':0.1,
                            'flow_sigma':5, 
                            'total_sigma':3,
                            'reg_iterations':(40, 20, 10),
                            'verbose':False, 
                            'w':[0.1,1]
                                 }

    # custom
    # registration_params = {
    #         'type_of_transform': 'SyNAggro',
    #         'reg_iterations': (70, 50, 40),  # Increased iterations
    #         'syn_metric': 'CC',  # Cross-correlation metric
    #         'syn_metric_weight': 1,
    #         'syn_metric_radius': 4,
    #         'syn_sampling': 4,  # Increased sampling rate
    #         'grad_step': 1.0,  # Increased gradient step
    #         'flow_sigma': 1.5,
    #         'total_sigma': 0,
    #         # 'verbose': True,
    #         'winsorize_lower_quantile': 0.005,  # Winsorize lower quantile
    #         'winsorize_upper_quantile': 0.995,  # Winsorize upper quantile
    #         'histogram_matching': True,  # Histogram matching
    #         'regularization': 'bspline',  # B-spline regularization
    #         'regularization_param': (4, 40, 0.2),  # B-spline parameters
    #         'shrink_factors': [6, 4, 2, 1],  # Shrink factors
    #         'smoothing_sigmas': [3, 2, 1, 0],  # Smoothing sigmas
    #         'synaggro_param': (2, 0.8, 1)  # SyNAggro parameters
    #     }
    
    # TODO: try registering the inverse or log scaled images
    reg_dict = ants.registration(fixed, moving, outprefix=tmp_dir, **registration_params)     

    # -s -f -l not matched
    # Original (i.e. if If = Im, and Im = If) -- this was how the original code was
    M_field = nibabel.load(reg_dict['fwdtransforms'][0])
    iM_field = nibabel.load(reg_dict['invtransforms'][-1])
    # Testing? (i.e. if If = If, and Im = Im)
    # M_field = nibabel.load(reg_dict['invtransforms'][-1])
    # iM_field = nibabel.load(reg_dict['fwdtransforms'][0])

    # Flip the z direction (this is required, and can be verified by matching with Optical Flow)
    Mt = -M_field.get_fdata()
    print(f'Shape of Mt field: {Mt.shape}')
    iMt = -iM_field.get_fdata()
    Mt[..., :2] = -Mt[..., :2]
    iMt[..., :2] = -iMt[..., :2]

    # Squeeze and rescale
    Mt = np.squeeze(Mt)
    print(f'Shape of Mt field after squeezing: {Mt.shape}')
    iMt = np.squeeze(iMt)
    Mt = M_scale(Mt, If.shape, 1/reg_level[-1])
    iMt = M_scale(iMt, If.shape, 1/reg_level[-1])

    fileList = glob.glob(tmp_dir + '*')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            continue

    return Mt, iMt

# get Jacobian, Specific Ventilation


def ANTsJac(If, Im, vox_res=[1, 1, 1], reg_level=[8, 4, 2], gauss_filt=[2, 2, 1]):
    # to antsimage
    fixed = ants.from_numpy(If)
    moving = ants.from_numpy(Im)

    tmp_dir = 'tmp{}_'.format(np.random.randint(0, 1e4))

    # default
    # registration_params =   {'type_of_transform':'SyNOnly',
    #                         'syn_metric':'demons', 
    #                         'syn_sampling':4,
    #                         'grad_step':0.1,
    #                         'flow_sigma':5, 
    #                         'total_sigma':3,
    #                         'reg_iterations':(40, 20, 10),
    #                         'verbose':False, 
    #                         'outprefix':tmp_dir,
    #                         'w':[0.1,1]
    #                              }

    # custom
    registration_params = {
            'type_of_transform': 'SyNAggro',
            'reg_iterations': (70, 50, 70),  # Increased iterations
            'syn_metric': 'CC',  # Cross-correlation metric
            'syn_metric_weight': 1,
            'syn_metric_radius': 4,
            'syn_sampling': 4,  # Increased sampling rate
            'grad_step': 1.0,  # Increased gradient step
            'flow_sigma': 1.5,
            'total_sigma': 0,
            # 'verbose': True,
            'winsorize_lower_quantile': 0.005,  # Winsorize lower quantile
            'winsorize_upper_quantile': 0.995,  # Winsorize upper quantile
            'histogram_matching': True,  # Histogram matching
            'regularization': 'bspline',  # B-spline regularization
            'regularization_param': (4, 40, 0.2),  # B-spline parameters
            'shrink_factors': [6, 4, 2, 1],  # Shrink factors
            'smoothing_sigmas': [3, 2, 1, 0],  # Smoothing sigmas
            'synaggro_param': (2, 0.8, 1)  # SyNAggro parameters
        }
    
    # TODO: try registering the inverse or log scaled images
    # fixed_i = ants.from_numpy(Im)
    # moving_i = ants.from_numpy(If)
    # reg_dict = ants.registration(fixed_i, moving_i, outprefix=tmp_dir, **registration_params)
    reg_dict = ants.registration(fixed, moving, outprefix=tmp_dir, **registration_params)



    # Jacobian
    # TODO: try registering the inverse or log scaled images
    jac_ants = ants.create_jacobian_determinant_image(
        fixed, reg_dict['invtransforms'][-1])
    # jac_ants = ants.create_jacobian_determinant_image(
    #     fixed_i, reg_dict['invtransforms'][-1])
    jac = jac_ants.numpy()

    # calculate specific ventilation
    reg_ants = reg_dict['warpedmovout'] # Caution, change this to 'warpedfixout' if switching Im and If
    reg = reg_ants.numpy()

    # TODO: optimize this if resolution of sv/jacs is too high
    # reg = ndimage.filters.gaussian_filter(
    #     reg, (2, 2, 2), mode='reflect', truncate=1)
    # If = ndimage.filters.gaussian_filter(
    #     If, (2, 2, 2), mode='reflect', truncate=1)

    sv = (If - reg) / (reg + np.finfo(float).eps) # Note, reg and If were switched from the original code as If and Im were switched as function inputs

    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob(tmp_dir + '*')
    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            continue

    return jac, sv

# Demons registration


def imgrad3d(I):
    gx = I-sp.circshift(I, (-1,), axes=(0,))
    gx[-1, :, :] = 0
    gy = I-sp.circshift(I, (-1,), axes=(1,))
    gy[:, -1, :] = 0
    gz = I-sp.circshift(I, (-1,), axes=(2,))
    gz[:, :, -1] = 0

    return gx, gy, gz


def lap3d(I):
    gxx = sp.circshift(I, (-1,), axes=(0,)) + \
        sp.circshift(I, (1,), axes=(0,)) - 2*I
    gyy = sp.circshift(I, (-1,), axes=(1,)) + \
        sp.circshift(I, (1,), axes=(1,)) - 2*I
    gzz = sp.circshift(I, (-1,), axes=(2,)) + \
        sp.circshift(I, (1,), axes=(2,)) - 2*I
    lapI = gxx + gyy + gzz
    return lapI


def pmask(I, sigma):
    # TODO: optimize
    I = np.abs(I)
    mask = np.abs(I) > sigma
    mask = ndimage.morphology.binary_fill_holes(mask)
    mask = ndimage.morphology.binary_opening(
        mask, structure=np.ones((5, 5, 5)))
    return mask


def DemonsReg4(Is, ref=0, level=3, device=-1):
    M_fields = []
    iM_fields = []
    nphase = len(Is)
    print('4D Demons registration:')
    for i in range(nphase):
        print('Ref/Mov:{}/{}'.format(i, ref))
        M_field = Demons(np.abs(Is[ref]), np.abs(
            Is[i]), level=level, device=device)
        M_fields.append(M_field)

    return np.asarray(M_fields)


def Demons(If, Im, level, device=-1, rho=0.7,
           sigmas_f=[2, 2, 2, 3], sigmas_e=[2, 2, 2, 2], sigmas_s=[.5, .5, 1, 1], iters=[40, 40, 40, 20, 20]):
    # normalization??
    Im = np.abs(Im)
    m_scale = np.max(Im)
    Im = Im/m_scale
    If = np.abs(If)
    If = If/m_scale

    # registration
    M = np.zeros(Im.shape+(3,))
    Mt = np.zeros(Im.shape+(3,))
    for k in range(level):
        print('Demons Level:{}'.format(k))
        # hyperparameter assignment
        scale = 2**(level-k-1)
        sigma_f = sigmas_f[k]
        sigma_e = sigmas_e[k]
        sigma_s = sigmas_s[k]
        iter_each_level = iters[k]

        ###
        Ift = ndimage.zoom(If, zoom=1/scale, order=2)
        Ift = ndimage.gaussian_filter(Ift, sigma=sigma_s, truncate=2.0)
        Imt = ndimage.zoom(Im, zoom=1/scale, order=2)
        Imt = ndimage.gaussian_filter(Imt, sigma=sigma_s, truncate=2.0)
        Imask = pmask(Imt+Ift, 1e-2)

        Isizet = Ift.shape
        Mt = M_scale(Mt, Isizet)
        uo = np.zeros_like(Mt)
        for i in range(iter_each_level):

            Imm = interp(Imt, Mt, device=sp.Device(device), k_id=1)
            Ifm = interp(Ift, -Mt, device=sp.Device(device), k_id=1)
            dI = Ifm-Imm
            Is = (Ifm+Imm)/2
            # Is = ndimage.gaussian_filter((Ifm+Imm)/2,sigma=sigma_s,truncate=2.0)

            gIx, gIy, gIz = imgrad3d(Is)
            gI = np.sqrt(np.abs(gIx**2+gIy**2+gIz**2)+1e-6)
            discriminator = gI**2 + np.abs(dI)**2
            dI = dI * 3.0
            ux = -dI*gIx/discriminator
            uy = -dI*gIy/discriminator
            uz = -dI*gIz/discriminator

            mask = (gI < 1e-4) | (~Imask)
            ux[np.isnan(ux) | mask] = 0
            uy[np.isnan(uy) | mask] = 0
            uz[np.isnan(uz) | mask] = 0

            ux = np.maximum(np.minimum(ux, 1), -1)
            uy = np.maximum(np.minimum(uy, 1), -1)
            uz = np.maximum(np.minimum(uz, 1), -1)
            ux = ndimage.gaussian_filter(ux, sigma=sigma_f)
            uy = ndimage.gaussian_filter(uy, sigma=sigma_f)
            uz = ndimage.gaussian_filter(uz, sigma=sigma_f)

            Mt[..., 0] = Mt[..., 0] + rho * ux + (1-rho)*uo[..., 0]
            Mt[..., 1] = Mt[..., 1] + rho * uy + (1-rho)*uo[..., 1]
            Mt[..., 2] = Mt[..., 2] + rho * uz + (1-rho)*uo[..., 2]
            uo[..., 0] = ux
            uo[..., 1] = uy
            uo[..., 2] = uz

            Mt[..., 0] = ndimage.gaussian_filter(Mt[..., 0], sigma=sigma_e)
            Mt[..., 1] = ndimage.gaussian_filter(Mt[..., 1], sigma=sigma_e)
            Mt[..., 2] = ndimage.gaussian_filter(Mt[..., 2], sigma=sigma_e)

    # TODO inverse combination (right now just double)
    M = M_scale(Mt*2, Im.shape)
    return M


# interpolation operator
class interp_op(Linop):
    def __init__(self, ishape, M_field, iM_field=None):
        ndim = M_field.shape[-1]
        assert list(ishape) == list(M_field.shape[:-1]), "Dimension mismatch!"
        oshape = ishape
        self.M_field = M_field
        self.iM_field = iM_field
        super().__init__(oshape, ishape)

    def _apply(self, input):
        device = backend.get_device(input)

        with device:
            return interp(input, self.M_field, device)

    def _adjoint_linop(self):
        device = backend.get_device(input)
        if self.iM_field is None:
            iM_field = -self.M_field
            M_field = None
        else:
            iM_field = self.iM_field
            M_field = self.M_field

        return interp_op(self.ishape, iM_field, M_field)


def interp(I, M_field, device=sp.Device(-1), k_id=1, deblur=True):
    # b spline interpolation
    N = 64
    if k_id == 0:
        kernel = [(3*(x/N)**3-6*(x/N)**2+4)/6 for x in range(0, N)] + \
            [(2-x/N)**3/6 for x in range(N, 2*N)]
        dkernel = np.array([-.2, 1.4, -.2])

        k_wid = 4
    else:
        kernel = [1-x/(2*N) for x in range(0, 2*N)]
        dkernel = np.array([0, 1, 0])
        deblur = False
        k_wid = 2
    kernel = np.asarray(kernel)

    c_device = sp.get_device(I)
    ndim = M_field.shape[-1]

    # 2d/3d
    if ndim == 3:
        dkernel = dkernel[:, None, None] * \
            dkernel[None, :, None]*dkernel[None, None, :]
        Nx, Ny, Nz = I.shape
        my, mx, mz = np.meshgrid(np.arange(Ny), np.arange(Nx), np.arange(Nz))
        m = np.stack((mx, my, mz), axis=-1)
        M_field = M_field + m
    else:
        dkernel = dkernel[:, None]*dkernel[None, :]
        Nx, Ny = I.shape
        my, mx = np.meshgrid(np.arange(Ny), np.arange(Nx))
        m = np.stack((mx, my, mz), axis=-1)
        M_field = M_field + m
    # TODO remove out of range values

    # image warp

    g_device = device
    I = sp.to_device(input=I, device=g_device)
    # I = sp.interp.interpolate(I,k_wid,kernel,M_field.astype(np.float64))
    # Use newer interp function for newer versions of SigPy
    I = sp.interp.interpolate(
        I, width=k_wid, kernel='spline', coord=sp.to_device(M_field.astype(np.float64), device))
    # deconv
    if deblur is True:
        sp.conv.convolve(I, dkernel)
    I = sp.to_device(input=I, device=c_device)

    return I
