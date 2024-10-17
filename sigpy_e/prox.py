import numpy as np
from sigpy import backend, util, thresh, linop
from sigpy import prox
import matplotlib.pyplot as plt

def GLRA(shape,lamda,A = None,sind_1 = 1):
    
    u_len = 1
    for i in range(sind_1):
        u_len = u_len * shape[i]
        
    v_len = 1
    for i in range(len(list(shape))-sind_1):
        v_len = v_len * shape[-i-1]    
    
    ishape = (u_len,v_len)
    
    GPR_prox = GLR(ishape, lamda)
    R = linop.Reshape(oshape=ishape,ishape=shape)
    if A is None:
        RA = R
    else:
        RA = R*A
    GLRA_prox = prox.UnitaryTransform(GPR_prox,RA)
    return GLRA_prox

class GLR(prox.Prox):
    def __init__(self, shape, lamda):
        self.lamda = lamda
        super().__init__(shape)

    def _prox(self, alpha, input):
        u,s,vh = np.linalg.svd(input,full_matrices=False)
        # print('SVD outputs: u.shape:{}, s.shape:{}, vh.shape:{}'.format(
        #     u.shape, s.shape, vh.shape))
        try:
            plot = False
            if plot:
                import sigpy.plot as pl
                pl.ImagePlot(vh.reshape((vh.shape[0], 110, 110, 40)),
                                    x=2, y=1, z=0, title="Spatial bases before soft thresholding", save_basename="pre")
                plt.close()
        except:
            print("Plotting failed.")
        s_max = np.max(s)
        #print('Eigen Value:{}'.format(np.diag(s)))
        s_t = thresh.soft_thresh(self.lamda * alpha*s_max, s)
        
        try:
            if plot:
                tmp = np.matmul(u, s_t[..., None]*vh)
                pl.ImagePlot(tmp.reshape((vh.shape[0], 110, 110, 40)),
                                    x=2, y=1, z=0, title="Spatial bases after soft thresholding", save_basename="post")
                plt.close()
        except:
            print("Plotting failed.")
        return np.matmul(u, s_t[...,None]*vh)
    