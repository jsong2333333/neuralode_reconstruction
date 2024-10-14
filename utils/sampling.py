import numpy as np

def uniform(A, r):
    m, n = A.shape
    s = np.random.choice(m, r, p = [1/m]*m, replace=False)
    p = np.random.choice(n, r, p = [1/n]*n, replace=False)
    
    C = A[:, p]
    R = A[s, :]
    M = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
    return C, R, M

def lev_fast(A, r, q=4):
    '''

    Args:
        a_mat:
        q_int:

    Returns:

    '''
    m, n = A.shape

    u, s, vt = np.linalg.svd(A, False)
    B = (u * s**(2 * q)).dot(u.T)  
    C = srft(A, r)
    u, _, v = np.linalg.svd(B @ C, False)

    lev_score_u = np.linalg.norm(u, axis=1)**2
    lev_score_v = np.linalg.norm(v, axis=1)**2
    
    s = np.asarray(np.argsort(lev_score_u)[::-1][0:r])
    p = np.asarray(np.argsort(lev_score_v)[::-1][0:r])

    C = A[:, p]
    R = A[s, :]
    M = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
    return C, R, M

def lev_exact(A, r ,scale=False):
    '''

    Args:
        a_mat:
        c_int:
        k:
        scale:

    Returns:

    '''
    u_mat, _, v_mat_t = np.linalg.svd(A, False)
    v_mat = v_mat_t.T
    u1_mat = u_mat #[:, :r]
    v1_mat = v_mat #[:,:r]
    lev_score_u = np.linalg.norm(u1_mat, axis=1)**2
    lev_score_v = np.linalg.norm(v1_mat, axis=1)**2

    s = np.asarray(np.argsort(lev_score_u)[::-1][0:r])
    p = np.asarray(np.argsort(lev_score_v)[::-1][0:r])


    C = A[:, p]
    R = A[s, :]

    M = np.linalg.pinv(C) @ A @ np.linalg.pinv(R)
    return C, R, M

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

