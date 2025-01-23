import numpy as np
from scipy.stats import unitary_group, ortho_group
from code.decomposition.exact_decomposition.decompose_unitary import decompose_isometry, reconstruct_unitary
from code.decomposition.exact_decomposition.index import calculate_index


dim = 10
U = np.zeros((dim, dim))
for i in range(dim-2):
    U[i+2, i] = 1


(
    M_A_local, M_B_local, 
    U_A_trans, Vh_B_trans, 
    U_B_trans, Vh_A_trans, 
    U_A_mix, Vh_A_mix, 
    U_B_mix, Vh_B_mix, 
    Cos
) = decompose_isometry(U, dim_A=5)

U_re = reconstruct_unitary(
    M_A_local, M_B_local, 
    U_A_trans, Vh_B_trans, 
    U_B_trans, Vh_A_trans, 
    U_A_mix, Vh_A_mix, 
    U_B_mix, Vh_B_mix, 
    Cos
)

index = calculate_index(U,dim_A=5)
print("Error :", np.max(np.abs(U-U_re)))
print("index:", index)
