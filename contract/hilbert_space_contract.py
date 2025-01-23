import numpy as np





def many_body_contract(U, n_contracted_modes):
    dim_Hilbert = U.shape[0]
    n_modes = int(np.log2(dim_Hilbert))
    n_free_modes = n_modes - n_contracted_modes

    dim_Hilbert_A = 2**n_free_modes
    dim_Hilbert_B = 2**n_contracted_modes
    
    U_tp = U.reshape((dim_Hilbert_B, dim_Hilbert_A, dim_Hilbert_B, dim_Hilbert_A))
    UA = np.trace(U_tp, axis1=0, axis2=2)
    return UA


def many_body_contract2(U, n_contracted_modes):
    dim_Hilbert = U.shape[0]
    n_modes = int(np.log2(dim_Hilbert))
    n_free_modes = n_modes - n_contracted_modes

    dim_Hilbert_A = 2**n_free_modes
    dim_Hilbert_B = 2**n_contracted_modes
    
    U_tp = U.reshape((dim_Hilbert_B, dim_Hilbert_A, dim_Hilbert_B, dim_Hilbert_A))

    for i in range(dim_Hilbert_B):
        binary_str = np.binary_repr(i)
        binary_list = [int(bit) for bit in binary_str]
        # even or odd number of fermion
        n_fermions = np.sum(binary_list)
        parity = n_fermions%2
        U_tp[i,:,i,:] *= (-1)**parity

    UA = np.trace(U_tp, axis1=0, axis2=2)
    return UA