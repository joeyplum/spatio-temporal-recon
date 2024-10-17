import sigpy as sp
import numpy as np
import sigpy.mri as mr
import sigpy_e.nft as nft
from tqdm.auto import tqdm


def jsens_calib(ksp, coord, dcf, ishape, device=sp.Device(-1),
                mps_ker_width=12,
                ksp_calib_width=32,
                lamda=0,):
    img_s = nft.nufft_adj([ksp], [coord], [dcf],
                          device=device, ishape=ishape, id_channel=True)
    ksp = sp.fft(input=np.asarray(img_s[0]), axes=(1, 2, 3))
    mps = mr.app.JsenseRecon(ksp,
                             mps_ker_width=mps_ker_width,
                             ksp_calib_width=ksp_calib_width,
                             lamda=lamda,
                             device=device,
                             comm=sp.Communicator(),
                             max_iter=20,
                             max_inner_iter=20).run()
    return mps


def espirit_calib(ksp, coord, dcf, ishape, device = sp.Device(-1)):
    img_s = nft.nufft_adj([ksp],[coord],[dcf],device = device,ishape = ishape,id_channel =True)
    # print(len(img_s))
    # import sigpy.plot as pl
    # pl.ImagePlot(img_s[0], x=1, y=2, z=0,
    #                         title="img_s")
    ksp = sp.fft(input=np.asarray(img_s[0]),axes=(1,2,3))
    mps = mr.app.EspiritCalib(ksp,
                             kernel_width=6,
                             calib_width=24,
                             device=device,
                             crop=0.75,
                             output_eigenvalue=False,
                             max_iter=100).run()
    return mps


def FD(ishape, axes=None):
    """Linear operator that computes finite difference gradient.
    Args:
       ishape (tuple of ints): Input shape.
    """
    I = sp.linop.Identity(ishape)
    axes = sp.util._normalize_axes(axes, len(ishape))
    ndim = len(ishape)
    linops = []
    for i in axes:
        D = I - sp.linop.Circshift(ishape,
                                   [0] * i + [1] + [0] * (ndim - i - 1))
        R = sp.linop.Reshape([1] + list(ishape), ishape)
        linops.append(R * D)

    G = sp.linop.Vstack(linops, axis=0)

    return G


def TVt_prox(X, lamda, iter_max=10):
    scale = np.max(np.abs(X))
    X = X/scale
    TVt = FD(X.shape, axes=(0,))
    TVs = FD(X.shape, axes=(1, 2, 3))  # JWP
    X_b = X
    Y = TVt*X

    # Instead, apply to spatial gradient of X - JWP
    # Y = TVt*np.linalg.norm(TVs*X, axis=0)

    Y = Y/(np.abs(Y)+1e-9)*np.minimum(np.abs(Y)+1e-9, 1)
    for _ in range(iter_max):
        X_b = X_b - ((X_b-X)+lamda*TVt.H*Y)
        Y = Y + lamda*TVt*X_b
        Y = Y/(np.abs(Y)+1e-9)*np.minimum(np.abs(Y)+1e-9, 1)

    X_b = X_b * scale
    return X_b


def pipe_menon_dcf(coord, img_shape=None, device=sp.cpu_device, max_iter=30,
                   n=128, beta=8, width=4, show_pbar=True):
    r"""Compute Pipe Menon density compensation factor.

    Perform the following iteration:

    .. math::

        w = \frac{w}{|G^H G w|}

    with :math:`G` as the gridding operator.

    Args:
        coord (array): k-space coordinates.
        device (Device): computing device.
        max_iter (int): number of iterations.
        n (int): Kaiser-Bessel sampling numbers for gridding operator.
        beta (float): Kaiser-Bessel kernel parameter.
        width (float): Kaiser-Bessel kernel width.
        show_pbar (bool): show progress bar.

    Returns:
        array: density compensation factor.

    References:
        Pipe, James G., and Padmanabhan Menon.
        Sampling Density Compensation in MRI:
        Rationale and an Iterative Numerical Solution.
        Magnetic Resonance in Medicine 41, no. 1 (1999): 179–86.


    """
    device = sp.Device(device)
    xp = device.xp

    with device:
        w = xp.ones(coord.shape[:-1], dtype=coord.dtype)
        if img_shape is None:
            img_shape = sp.estimate_shape(coord)

        # Get kernel
        x = xp.arange(n, dtype=coord.dtype) / n
        kernel = xp.i0(beta * (1 - x**2)**0.5).astype(coord.dtype)
        kernel /= kernel.max()

        G = sp.linop.Gridding(img_shape, coord, width, kernel)
        with tqdm(total=max_iter, desc="PipeMenonDCF",
                  disable=not show_pbar) as pbar:
            for it in range(max_iter):
                GHGw = G.H * G * w
                w /= xp.abs(GHGw)
                resid = xp.abs(GHGw - 1).max().item()

                pbar.set_postfix(resid='{0:.2E}'.format(resid))
                pbar.update()

    return w
