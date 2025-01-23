import numpy as np
from scipy.stats import unitary_group, ortho_group
from code.decomposition.exact_decomposition.decompose_unitary import decompose_isometry, reconstruct_unitary
from numpy.linalg import matrix_rank


dim = 9
dim_A = 2
dim_B = dim-dim_A

U = unitary_group.rvs(dim)

(
    M_A_local, M_B_local, 
    U_A_trans, Vh_B_trans, 
    U_B_trans, Vh_A_trans, 
    U_A_mix, Vh_A_mix, 
    U_B_mix, Vh_B_mix, 
    Cos
) = decompose_isometry(U, dim_A)

U_re = reconstruct_unitary(
    M_A_local, M_B_local, 
    U_A_trans, Vh_B_trans, 
    U_B_trans, Vh_A_trans, 
    U_A_mix, Vh_A_mix, 
    U_B_mix, Vh_B_mix, 
    Cos
)


print("Error :", np.max(np.abs(U-U_re)))