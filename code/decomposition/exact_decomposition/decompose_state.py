import numpy as np
from numpy import linalg as LA



def get_occupied_modes(correlation_matrix):
    eigenvalues, eigenvectors = LA.eigh(correlation_matrix)
    indices = np.isclose(eigenvalues,0)
    occupied_modes = eigenvectors[:,indices]
    return occupied_modes
    


def decompose_state(correlation_matrix, dim_A, tolerance=0.001):
    dim = correlation_matrix.shape[0]
    dim_B = dim-dim_A

    # get occupied modes
    Phi_occ = get_occupied_modes(correlation_matrix)
    n_occupied = Phi_occ.shape[1]

    # express occupied modes in subregions A and B
    Phi_occ_A = Phi_occ[:dim_A]
    Phi_occ_B = Phi_occ[dim_A:]

    # svd for A and B separately
    U_A, Cos_A, Vh_A =  LA.svd(Phi_occ_A)
    U_B, Cos_B, Vh_B =  LA.svd(Phi_occ_B)
    
    k_A = Cos_A.shape[0]
    k_B = Cos_B.shape[0]

    
    # find local occupied modes for A and B
    local_indices_A = (Cos_A >= (1-tolerance))
    local_indices_B = (Cos_B >= (1-tolerance))

    U_A_local = U_A[:,:k_A][:, local_indices_A]
    U_B_local = U_B[:,:k_B][:, local_indices_B]

    # find A&B mixed modes
    mix_index_A = (Cos_A < (1-tolerance))  &   (Cos_A > tolerance)
    mix_index_B = (Cos_B < (1-tolerance))  &   (Cos_B > tolerance)

    U_A_mix = U_A[:,:k_A][:, mix_index_A]
    U_B_mix = U_B[:,:k_B][:, mix_index_B]

    Cos_A_mix = Cos_A[mix_index_A]
    Cos_B_mix = Cos_B[mix_index_B]

    Vh_A_mix = Vh_A[:k_A][mix_index_A]
    Vh_B_mix = Vh_B[:k_B][mix_index_B]

    # reverse the order of states and values for B
    U_B_mix = U_B_mix[:,::-1]
    Vh_B_mix = Vh_B_mix[::-1]
    Cos_B_mix = Cos_B_mix[::-1]


    # SVD is up to a phase, make the phase compatible for A and B
    V_A_mix = Vh_A_mix.conj().T
    phase = Vh_B_mix @ V_A_mix
    U_B_mix = U_B_mix @ phase


   

    assert(np.allclose(np.sqrt(1-Cos_A_mix**2), Cos_B_mix))
    Cos = Cos_A_mix
    # Sin = Cos_B_mix

    return U_A_local, U_B_local, U_A_mix, U_B_mix, Cos


def reconstruct_state(U_A_local, U_B_local, U_A_mix, U_B_mix, Cos):

    # number of modes in subregions A and B
    dim_A = U_A_local.shape[0]
    dim_B = U_B_local.shape[0]
    dim = dim_A + dim_B

    # number of entangled occupied modes
    assert(U_A_mix.shape[1]==U_B_mix.shape[1])
    n_entangled = U_A_mix.shape[1]

    # pad zeros, express modes in full space A+B
    U_A_local_full = np.block([[U_A_local],
                         [np.zeros((dim_B, U_A_local.shape[1]))]])
    U_B_local_full = np.block([[np.zeros((dim_A, U_B_local.shape[1]))],
                         [U_B_local]])


    U_A_mix_full = np.block([[U_A_mix],
                         [np.zeros((dim_B, n_entangled))]])
    U_B_mix_full = np.block([[np.zeros((dim_A, n_entangled))],
                         [U_B_mix]])
    
    

    Sin = np.sqrt(1-Cos**2)
    Phi_occ_entangled = U_A_mix_full @ np.diag(Cos) +  U_B_mix_full @ np.diag(Sin)
    Phi_occ = np.block([U_A_local_full, Phi_occ_entangled,  U_B_local_full])
    Phi_occ_h = Phi_occ.conj().T

    correlation_matrix =  np.identity(dim) - Phi_occ @ Phi_occ_h
    return correlation_matrix





    



    

