import numpy as np
import torch
import numpy as np
from scipy import linalg
from scipy.linalg import qr
from utils.sampling import srft
import math
from time import time


def lev_score_point_selection(uk_mat,c_int,beta=0.7,scale=True):
    u_scores = np.linalg.norm(uk_mat, axis=1) ** 2
    n_int, k_int = uk_mat.shape
    # print(uk_mat.shape)
    lev_score = beta * u_scores/k_int + (1-beta)/n_int# length the same as training data instance
    indices = np.random.choice(np.arange(n_int), c_int, replace=True, p= lev_score)
    pd = {i: lev_score[i] for i in range(n_int)} #  2889: 0.00043129640768956235
    # print('n_int,c_int',n_int,c_int)
    s1_mat = np.zeros((n_int, c_int))
    d = np.empty(c_int)
    for i, idx in enumerate(indices): # i is the i-th trial, idx is the index of the column
        s1_mat[idx][i] = 1
        for j in range(c_int):
            d[j] = 1/np.sqrt(c_int * pd[idx])
    
    if scale:
        D = np.diag(d)
        s1_mat = s1_mat @ D
    return s1_mat

def rdeim(xmat,k_int,beta=0.5,eps = 0.1,delta = 0.1):
    u_mat,sig_mat,vt_mat = np.linalg.svd(xmat,full_matrices=False)
    c_int = int(2*k_int*math.log(k_int))
    uk_mat = u_mat[:,:k_int]
    s1u_mat = lev_score_point_selection(uk_mat,c_int,beta,scale=True)
    _, _, prow = qr(np.dot(uk_mat.T,s1u_mat), pivoting=True)

    s = prow[0:k_int]
    Id = np.identity(c_int)
    s2u_mat = Id[:,s]
    su_mat = np.dot(s1u_mat,s2u_mat)

    s = []
    for i in range(len(su_mat)):
        if np.linalg.norm(su_mat[i]) > 0:
            s.append(i)
    vk_mat = vt_mat.T[:,:k_int]
    s1v_mat = lev_score_point_selection(vk_mat,c_int,beta,scale=True)
    # u, _, vt = np.linalg.svd(A, False)
    _, _, pcol = qr(np.dot(vk_mat.T,s1v_mat), pivoting=True)
    p = pcol[0:k_int]
    s2v_mat = Id[:,p]
    sv_mat = np.dot(s1v_mat,s2v_mat)

    p = []
    for i in range(len(sv_mat)):
        if np.linalg.norm(sv_mat[i]) > 0:
            p.append(i)
    return s,p

def hybrid_deim(u_mat,sig_mat,vt_mat,a_mat,k_int,q=4,beta=0.7,eps = 10,delta =10):
    B = (u_mat * sig_mat**(2 * q)).dot(u_mat.T) 
    r_int = 2 * k_int
    C = srft(a_mat, r_int)
    # svd of B = (A*A^T)^q A*Pi
    ub_mat, _, vbt_mat = np.linalg.svd(B @ C, False)

    s,p = rdeim(ub_mat,vbt_mat,k_int,beta,eps,delta)
    return s,p

def deim(xmat, k_int):
    u_mat,sig_mat,vt_mat = np.linalg.svd(xmat,full_matrices=False)
    irow = np.zeros([k_int, ], dtype=int)
    icol = np.zeros([k_int, ], dtype=int)

    uk_mat = u_mat[:,:k_int]
    vk_mat = vt_mat.T[:,:k_int]

    for j in range(k_int):
        irow[j] = np.argmax(np.absolute(uk_mat[:, j]))
        icol[j] = np.argmax(np.absolute(vk_mat[:, j]))
        if j < (k_int - 1):
            x = np.linalg.pinv(uk_mat[irow[:j + 1], :j + 1]) @ uk_mat[irow[:j + 1], j + 1]
            y = np.linalg.pinv(vk_mat[icol[:j + 1], :j + 1]) @ vk_mat[icol[:j + 1], j + 1]
            uk_mat[:, j + 1] = uk_mat[:, j + 1] - uk_mat[:, :j + 1] @ x
            vk_mat[:, j + 1] = vk_mat[:, j + 1] - vk_mat[:, :j + 1] @ y
    s = irow
    p = icol
    return s, p

def qdeim(xmat, k_int):
    u_mat,sig_mat,vt_mat = np.linalg.svd(xmat,full_matrices=False)
    uk_mat = u_mat[:,:k_int]
    vk_mat = vt_mat.T[:,:k_int]

    _, _, prow = qr(uk_mat.T, pivoting=True)
    _, _, pcol = qr(vk_mat.T, pivoting=True)

    s = prow[0:k_int]
    p = pcol[0:k_int]
    return s, p

def qdeimplus(xmat,  k_int):
    u_mat,sig_mat,vt_mat = np.linalg.svd(xmat,full_matrices=False)
    uk_mat = u_mat[:, :k_int]
    vk_mat = vt_mat.T[:, :k_int]
    irow = np.zeros([k_int, ], dtype=int)
    icol = np.zeros([k_int, ], dtype=int)

    for j in range(k_int):
        irow[j] = np.argmax(np.absolute(uk_mat[:, j]))
        icol[j] = np.argmax(np.absolute(vk_mat[:, j]))
        if j < (k_int - 1):
            x = np.linalg.pinv(uk_mat[irow[:j + 1], :j + 1]) @ uk_mat[irow[:j + 1], j + 1]
            y = np.linalg.pinv(vk_mat[icol[:j + 1], :j + 1]) @ vk_mat[icol[:j + 1], j + 1]
            uk_mat[:, j + 1] = uk_mat[:, j + 1] - uk_mat[:, :j + 1] @ x
            vk_mat[:, j + 1] = vk_mat[:, j + 1] - vk_mat[:, :j + 1] @ y

    _, _, s = qr(uk_mat.T, pivoting=True)
    _, _, p = qr(vk_mat, pivoting=True)

    indices = np.where(np.in1d(s, irow))[0]  # find dublicates
    s = np.delete(s, indices, None)[0:k_int]  # remove dublicates and select subset

    indices = np.where(np.in1d(p, icol))[0]  # find dublicates
    p = np.delete(p, indices, None)[0:k_int]  # remove dublicates and select subset

    s = np.concatenate([irow, s])
    p = np.concatenate([icol, p])

    return s, p
    
def ldeim(a_mat,u_mat,vt_mat, r_int, k_int, q=4, method='exact'):
    uk_mat = u_mat[:,:k_int]
    vk_mat = vt_mat.T[:,:k_int]
    irow = np.zeros([k_int, ], dtype=int)
    icol = np.zeros([k_int, ], dtype=int)

    for j in range(k_int):
        irow[j] = np.argmax(np.absolute(uk_mat[:, j]))
        icol[j] = np.argmax(np.absolute(vk_mat[:, j]))
        if j < (k_int - 1):
            x = np.linalg.pinv(uk_mat[irow[:j + 1], :j + 1]) @ uk_mat[irow[:j + 1], j + 1]
            y = np.linalg.pinv(vk_mat[icol[:j + 1], :j + 1]) @ vk_mat[icol[:j + 1], j + 1]
            uk_mat[:, j + 1] = uk_mat[:, j + 1] - uk_mat[:, :j + 1] @ x
            vk_mat[:, j + 1] = vk_mat[:, j + 1] - vk_mat[:, :j + 1] @ y

    if method == 'exact':
        # residual singular vectors
        lev_u = np.linalg.norm(uk_mat, axis=1) ** 2
        lev_v = np.linalg.norm(vk_mat, axis=1) ** 2

    elif method == 'fast':
               
        sr = int(np.floor(s[0]**2/np.linalg.norm(a_mat)**2)*1.5)
        b_mat = (u_mat[:, 0:sr]*s[0:sr]**(2*q)).dot(u_mat[:,0:sr].T)
        #b_mat = (u*s**(2*q)).dot(u.T)
        
        c_mat = srft(a_mat, r_int)
        b_mat = b_mat @ c_mat
        u_mat, _, v_mat_t = np.linalg.svd(b_mat, False)
        u_mat = u_mat[:, 0:r_int]
        v_mat_t = v_mat_t[0:r_int, :]
   
        #u_mat, v_mat_t = _rsvd(A, r, q=8) 
        lev_u = np.linalg.norm(u_mat, axis=1) ** 2
        lev_v = np.linalg.norm(v_mat_t, axis=1) ** 2
    s = np.asarray(np.argsort(lev_u)[::-1])
    indices = np.where(np.in1d(s, irow))[0] # find dublicates
    s = np.delete(s, indices, None)[0:k_int] #remove dublicates and select subset
    
    p = np.asarray(np.argsort(lev_v)[::-1])
    indices = np.where(np.in1d(p, icol))[0] # find dublicates
    p = np.delete(p, indices, None)[0:k_int] #remove dublicates and select subset

    s = np.concatenate([irow, s])
    p = np.concatenate([icol, p])
    return s, p

def orthonormalize(A, overwrite_a=True, check_finite=False):
    """orthonormalize the columns of A via QR decomposition"""
    # NOTE: for A(m, n) 'economic' returns Q(m, k), R(k, n) where k is min(m, n)
    # TODO: when does overwrite_a even work? (fortran?)
    Q, _ = linalg.qr(A, overwrite_a=overwrite_a, check_finite=check_finite,
                     mode='economic', pivoting=False)
    return Q

def perform_subspace_iterations(A, Q, n_iter=2, axis=1):
    """perform subspace iterations on Q"""
    # TODO: can we figure out how not to transpose for row wise
    if axis == 0:
        Q = Q.T

    # orthonormalize Y, overwriting
    Q = orthonormalize(Q)

    # perform subspace iterations
    for _ in range(n_iter):
        if axis == 0:
            Z = orthonormalize(A.dot(Q))
            Q = orthonormalize(A.T.dot(Z))
        else:
            Z = orthonormalize(A.T.dot(Q))
            Q = orthonormalize(A.dot(Z))

    if axis == 0:
        return Q.T
    return Q

def srft(a_mat, c_int):
    '''
    Args:
        a_mat: m-by-n dense Numpy matrix
        c_int: sketch size.
    Returns:
        c_mat: m-by-s sketch C = A * S
    '''
    n_int = a_mat.shape[1]
    sign_vec = np.random.choice(2,n_int) * 2 - 1
    idx_vec = np.random.choice(n_int, c_int, replace=False)
    a_mat = a_mat * sign_vec.reshape(1,n_int)
    a_mat = realfft_row(a_mat)
    c_mat = a_mat[:, idx_vec] * np.sqrt(n_int / c_int)
    return c_mat

def realfft_row(a_mat):
    '''
    Real Fast Fourier Transform (FFT) Independently Applied to Each Row of A
    Input
        a_mat: m-by-n dense NumPy matrix.
    Output
        c_mat: m-by-n matrix C = A * F.
        Here F is the n-by-n orthogonal real FFT matrix (not explicitly formed)
    Notice that $C * C^T = A * A^T$;
    however, $C^T * C = A^T * A$ is not true.
    '''
    n_int = a_mat.shape[1]
    fft_mat = np.fft.fft(a_mat, n=None, axis=1) / np.sqrt(n_int)
    if n_int % 2 == 1:
        cutoff_int = int((n_int + 1) / 2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int, n_int))
    else:
        cutoff_int = int(n_int / 2)
        idx_real_vec = list(range(1, cutoff_int))
        idx_imag_vec = list(range(cutoff_int + 1, n_int))
    c_mat = fft_mat.real
    c_mat[:, idx_real_vec] *= np.sqrt(2)
    c_mat[:, idx_imag_vec] = fft_mat[:, idx_imag_vec].imag * np.sqrt(2)
    return c_mat

def get_sensors(X_train, flag, sensor_num=64, datashape=(128, 128), interestshape=None):
    X_train = X_train.T
    n_pix, n_snapshots_train = X_train.shape
    assert n_pix == datashape[0]*datashape[1]
    
    s_time = time()
    if flag == 'deim':
        s, _ = deim(X_train, sensor_num)
    elif flag == 'uniform':
        s = get_uniform_sensors(datashape, sensor_num, interestshape=interestshape)
    elif flag == 'random':
        s = np.random.permutation(n_pix)[:sensor_num]
    e_time = time()
    print(f'------ time used for sampling {e_time-s_time} ------')

    sensors = X_train[s,:].T
    sensors = sensors.reshape(n_snapshots_train, sensor_num) # 100 * 12
    return sensors, s

def get_uniform_sensors(datashape, sensor_num, interestshape=None):
    m, n = datashape
    md, nd = m, n
    if interestshape:
        m, n = interestshape
        assert m <= md and n <= nd
    c1 = np.arange(1, sensor_num/2+1, dtype=int)
    divisible = sensor_num % c1 == 0
    c1 = c1[divisible]
    c2 = sensor_num//c1
    argmin = np.argmin(np.abs(c1-c2))
    c1val, c2val = c1[argmin], c2[argmin]
    if m <= n:
        mval = min(c1val, c2val)
        nval = max(c1val, c2val)
    else:
        mval = max(c1val, c2val)
        nval = min(c1val, c2val)
    mlin, nlin = np.linspace(0, m-1, mval, dtype=int), np.linspace(0, n-1, nval, dtype=int)
    mv, nv = np.meshgrid(mlin, nlin)
    s = (mv*nd + nv).flatten()
    s += nd*(md-m)//2 + (nd-n)//2
    assert np.all(s < md*nd)
    return s