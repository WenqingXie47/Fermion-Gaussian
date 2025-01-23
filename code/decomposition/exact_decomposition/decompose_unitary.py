
import numpy as np
from numpy import linalg as LA



def get_local_isometry(
    U_A, Cos_A, Vh_A,
    tolerance=0.001 
):
    
    local_indices_A = (Cos_A >= (1-tolerance))
    U_A_local = U_A[:, local_indices_A]
    Vh_A_local = Vh_A[local_indices_A]

    M_A_local = U_A_local @ Vh_A_local
    return M_A_local


def get_translation_svd(
    Cos_A, Vh_A, 
    M_BA,
    tolerance=0.001
):
    translate_indices_A = (Cos_A < tolerance)
    Vh_A_trans = Vh_A[translate_indices_A]
    U_B_trans = M_BA @ Vh_A_trans.conj().T

    return U_B_trans, Vh_A_trans


def find_mix_svd(
    U_A, Cos_A, Vh_A,
    tolerance=0.001
):
    mix_index_A = (Cos_A < (1-tolerance))  &   (Cos_A > tolerance)
    U_A_mix = U_A[:, mix_index_A]
    Cos_A_mix = Cos_A[mix_index_A]
    Vh_A_mix = Vh_A[mix_index_A]
    
    return U_A_mix, Cos_A_mix, Vh_A_mix


def get_mix_weight(Cos_A_mix, Cos_B_mix):
    assert np.allclose(Cos_A_mix, Cos_B_mix)
    Cos = Cos_A_mix
    Sin = np.sqrt(1-Cos**2)
    return Cos, Sin
    




def decompose_isometry(matrix, dim_A, tolerance=0.001):

    M = matrix
    # dimension of subregion A and B
    dim = M.shape[0]
    dim_B = dim - dim_A

    M_AA = M[:dim_A,:dim_A]
    M_BB = M[dim_A:,dim_A:]
    M_AB = M[:dim_A,dim_A:]
    M_BA = M[dim_A:,:dim_A]


    # SVD for A and B separately    
    U_A, Cos_A, Vh_A =  LA.svd(M_AA)
    U_B, Cos_B, Vh_B =  LA.svd(M_BB)

    
    # Find local transformations for regions A and B
    M_A_local = get_local_isometry(U_A, Cos_A, Vh_A)
    M_B_local = get_local_isometry(U_B, Cos_B, Vh_B)



    U_B_trans, Vh_A_trans = get_translation_svd(Cos_A, Vh_A, M_BA)
    U_A_trans, Vh_B_trans = get_translation_svd(Cos_B, Vh_B, M_AB)


    # Find transformation that mix A and B
    U_A_mix, Cos_A_mix, Vh_A_mix = find_mix_svd(U_A, Cos_A, Vh_A)
    U_B_mix, Cos_B_mix, Vh_B_mix = find_mix_svd(U_B, Cos_B, Vh_B)
    
    # Check vec A and vec B transform in pairs
    Cos, Sin = get_mix_weight(Cos_A_mix, Cos_B_mix)
    
    # SVD is up to a phase, make the phase compatible for A and B
    V_A = Vh_A_mix.conj().T
    Uh_B = U_B_mix.conj().T

    phase = Uh_B @ M_BA @ V_A /Sin
    U_B_mix = U_B_mix @ phase
    Vh_B_mix = phase.conj().T @ Vh_B_mix

    return (M_A_local, M_B_local, 
        U_A_trans, Vh_B_trans, U_B_trans, Vh_A_trans,
        U_A_mix, Vh_A_mix, U_B_mix, Vh_B_mix, Cos)





def reconstruct_unitary(
    M_A_local, M_B_local, 
    U_A_trans, Vh_B_trans, 
    U_B_trans, Vh_A_trans,
    U_A_mix, Vh_A_mix, 
    U_B_mix, Vh_B_mix, 
    Cos
):
    dim_A = M_A_local.shape[0]
    dim_B = M_B_local.shape[0]

    # mix transformation
    U_mix = np.block(
        [[U_A_mix, np.zeros_like(U_A_mix)],
        [np.zeros_like(U_B_mix), U_B_mix]]
    )

    Vh_mix = np.block(
        [[Vh_A_mix, np.zeros_like(Vh_B_mix)],
        [np.zeros_like(Vh_A_mix), Vh_B_mix]]
    )

    Sin = np.sqrt(1-Cos**2)
    O_mix = np.block(
        [[np.diag(Cos), -np.diag(Sin)],
        [np.diag(Sin), np.diag(Cos)]]
    )

    M_mix = U_mix @ O_mix @ Vh_mix

    # local transformation
    M_A_local_full = np.block(
        [[M_A_local, np.zeros((dim_A, dim_B))],
         [np.zeros((dim_B, dim_A)), np.zeros((dim_B, dim_B))]]
    )

    M_B_local_full = np.block(
        [[np.zeros((dim_A, dim_A)), np.zeros((dim_A, dim_B))],
         [np.zeros((dim_B, dim_A)), M_B_local]]
    )
    M_local = M_A_local_full + M_B_local_full

    # translation transformation
    U_A_trans_full = np.block(
        [[U_A_trans],
         [np.zeros((dim_B, U_A_trans.shape[1]))]]
    )
    U_B_trans_full = np.block(
        [[np.zeros((dim_A, U_B_trans.shape[1]))],
        [U_B_trans]]
    )
    Vh_A_trans_full = np.block(
        [Vh_A_trans, np.zeros((Vh_A_trans.shape[0], dim_B))]
    )
    Vh_B_trans_full = np.block(
        [np.zeros((Vh_B_trans.shape[0], dim_A)), Vh_B_trans]
    )

    M_trans_AB =  U_A_trans_full @ Vh_B_trans_full
    M_trans_BA =  U_B_trans_full @ Vh_A_trans_full
    M_trans =  M_trans_AB + M_trans_BA

    M = M_mix + M_local + M_trans
    return M









# def decompose_unitary(U, index_A):
#     '''
#     decompose unitary into A and B 
#     '''
#     dim = U.shape[0]
#     index = np.arange(dim)
#     index_B = np.delete(index, index_A)

#     dim_A = index_A.shape[0]
#     dim_B = dim-dim_A
#     if dim_A < dim/2:
#         index_A, index_B = index_B, index_A
#         dim_A, dim_B = dim_B, dim_A

#     U_AA = U[index_A][index_A]
#     # U = VDW^H
#     v_A, c, w_Ah =  sp.linalg.svd(U_AA)
#     w_A = w_Ah.conj().T
#     s = np.ones(dim_B) - c[dim_A-dim_B:]

#     V_A = np.zeros((dim,dim_A))
#     V_A[index_A] = v_A
#     W_A = np.zeros((dim,dim_A))
#     W_A[index_A] = w_A
    
#     V_B = U @ W_A[:,dim_A-dim_B:] - V_A[:,dim_A-dim_B:] @ np.diag(c)
#     W_B = U.conj().T @ V_B @ np.diag(1/c[dim_A-dim_B:])
#     W_B[index_A,:]=0
    
#     V = np.block([V_A, V_B])
#     W = np.block([W_A, W_B])

#     D = np.zeros((dim,dim))
#     D[np.arange(dim_A),np.arange(dim_A)] = c
#     D[np.arange(dim_A,dim),np.arange(dim_A,dim)] = c[dim_A-dim_B:]
#     D[np.arange(dim_A,dim),np.arange(dim_A-dim_B,dim_A)] = s
#     D[np.arange(dim_A-dim_B,dim_A), np.arange(dim_A,dim)] = -s
    

    

#     return V,D,W.conj().T
