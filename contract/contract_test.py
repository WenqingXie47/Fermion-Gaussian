import numpy as np
import itertools
from numpy import linalg as LA
from scipy.stats import unitary_group


def det_submatrix(matrix, out_index, in_index):
    assert(len(out_index)== len(in_index))
    sub_matrix = matrix[out_index,:][:,in_index]
    return LA.det(sub_matrix)

def occupied_modes_to_index(occupied_modes, n_modes):

    big_endian =  - occupied_modes + n_modes -1
    ind = np.sum(2**big_endian) # e.g. 0100 -> 4
    return ind

def from_mode_unitary(one_body_unitary):
    n_modes = one_body_unitary.shape[0]
    dim_Hilbert = 2**n_modes
    
    U = np.zeros((dim_Hilbert,dim_Hilbert),dtype="complex")
    U[0,0]=1
    
    for particle_number in range(1,n_modes+1): 
        combinations = list(itertools.combinations(range(n_modes), particle_number))
        combinations = np.array(combinations)
        dim_subspace = len(combinations)
        for i in range(dim_subspace):
            modes_out = combinations[i]
            index_out = occupied_modes_to_index(modes_out, n_modes)
            for j in range(dim_subspace):
                modes_in = combinations[j]
                index_in = occupied_modes_to_index(modes_in, n_modes)
                U[index_out,index_in] = det_submatrix(one_body_unitary, modes_out, modes_in)
    return U


def mode_trace(unitary, contracted_ind):
    dim = unitary.shape[0]
    modes_ind = np.arange(dim)
    remain_ind = np.setdiff1d(modes_ind, contracted_ind)
    

    U_AA = unitary[remain_ind,:][:,remain_ind]
    U_BB = unitary[contracted_ind,:][:,contracted_ind]
    U_AB = unitary[remain_ind,:][:,contracted_ind]
    U_BA = unitary[contracted_ind,:][:,remain_ind]
    I_BB = np.eye(U_BB.shape[0]) 
    contracted_u =  U_AA - U_AB @ np.linalg.pinv(I_BB+U_BB) @ U_BA 
    return contracted_u


def many_body_trace(U, contracted_modes_ind):

    contracted_modes_ind = np.array(contracted_modes_ind)
    contracted_modes_ind = np.sort(contracted_modes_ind)

    # calculate number of modes
    dim_Hilbert = U.shape[0]
    n_modes = int(np.log2(dim_Hilbert))
    n_contracted_modes = contracted_modes_ind.shape[0]
    n_remain_modes = n_modes - n_contracted_modes
    
    contracted_modes_ind = np.array(contracted_modes_ind)
    contracted_modes_ind = np.sort(contracted_modes_ind)
    
    
    shape = (2,) * (2 * n_modes)
    U_tensor = U.reshape(shape)

    
    for i in range(contracted_modes_ind.shape[0]):
        mode = contracted_modes_ind[i]
        U_tensor = np.trace(U_tensor, axis1=(mode-i), axis2=(mode+n_modes-2*i))

    
    # reshape into rank-2 unitary matrix
    
    dim_Hilbert_remain = 2**n_remain_modes
    UA = U_tensor.reshape((dim_Hilbert_remain, dim_Hilbert_remain))
    return UA


if __name__ == "__main__":
    dim = 10
    contracted_modes_ind = [6,7]

    u = unitary_group.rvs(dim)
    U = from_mode_unitary(u)
    ua = mode_trace(u, contracted_modes_ind)
    UA1 = from_mode_unitary(ua)

    UA2 = many_body_trace(U,contracted_modes_ind)
    ubb = u[contracted_modes_ind,:][:,contracted_modes_ind]
    norm = LA.det(ubb+np.eye(ubb.shape[0]))
    assert(np.allclose(UA2[0,0],norm))
    UA2 = UA2/UA2[0,0]
    # verify UA2 is really a unitary
    Id = UA2 @ UA2.conj().T
    assert(np.allclose(Id, np.eye(Id.shape[0])))
    # verify UA1 == UA2
    assert(np.allclose(UA1,UA2))