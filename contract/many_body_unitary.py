import numpy as np
from scipy.stats import unitary_group
import itertools
from numpy import linalg as LA

def random_unitary_charge_conserved(n_modes):
    
    dim_Hilbert = 2**n_modes
    
    U = np.zeros((dim_Hilbert,dim_Hilbert),dtype="complex")
    U[0,0]=1
    random_angle = np.random.uniform(0, 2 * np.pi)
    U[dim_Hilbert-1,dim_Hilbert-1] =  np.exp(1j*random_angle)
    for particle_number in range(1,n_modes): 
        combinations = list(itertools.combinations(range(n_modes), particle_number))
        combinations = np.array(combinations)
        dim_subspace = len(combinations)
        u = unitary_group.rvs(dim_subspace)
        for i in range(dim_subspace):
            modes_out = combinations[i]
            index_out = np.sum(2**modes_out)
            for j in range(dim_subspace):    
                modes_in = combinations[j]
                index_in = np.sum(2**modes_in)
                U[index_out,index_in] = u[i,j]
    return U



def det_submatrix(matrix, out_index, in_index):
    assert(len(out_index)== len(in_index))
    sub_matrix = matrix[out_index,:][:,in_index]
    return LA.det(sub_matrix)



def from_mode_unitary_sector(one_body_unitary, particle_number):
    n_modes = one_body_unitary.shape[0]

    if particle_number==0:
        return np.array([[1]],dtype="complex")
    
    combinations = list(itertools.combinations(range(n_modes), particle_number))
    dim_Hilbert = len(combinations)

    U = np.zeros((dim_Hilbert,dim_Hilbert),dtype="complex")
    for i in range(dim_Hilbert):
        for j in range(dim_Hilbert):
            U[i,j] = det_submatrix(one_body_unitary, combinations[i], combinations[j])
    return U


def from_mode_unitary_full(one_body_unitary):
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
            index_out = np.sum(2**modes_out)
            for j in range(dim_subspace):
                modes_in = combinations[j]
                index_in = np.sum(2**modes_in)
                U[index_out,index_in] = det_submatrix(one_body_unitary, modes_out, modes_in)
    return U


if __name__ == "__main__":
    n_modes = 10
    U = random_unitary_charge_conserved(n_modes)

    id = U @ U.conj().T
    assert(np.allclose(id, np.eye(id.shape[0])))