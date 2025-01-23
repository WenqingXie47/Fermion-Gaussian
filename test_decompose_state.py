import numpy as np
from scipy.stats import unitary_group, ortho_group
from code.decomposition.exact_decomposition.decompose_state import decompose_state, reconstruct_state




dim = 6
dim_A = 2
n_occupied = 3
n_empty = dim-n_occupied


# randomize a correlation matrix for pure Gaussian state
eigenvalues = np.concatenate([np.ones(n_empty), np.zeros(n_occupied)])
U = unitary_group.rvs(dim)
Uh = U.conj().T
correlation_matrix = U @ np.diag(eigenvalues) @ Uh

# decompose the correlation matrix
U_A_local, U_B_local, U_A_mix, U_B_mix, Cos = decompose_state(correlation_matrix,dim_A=dim_A)

# test the error
Co_re = reconstruct_state(U_A_local, U_B_local, U_A_mix, U_B_mix, Cos)


print("number of occupied modes :", n_occupied)
print("Error :", np.max(np.abs(correlation_matrix-Co_re)))

U_C_local, U_D_local, U_C_mix, U_D_mix, Cos = decompose_state(correlation_matrix,dim_A=4)

print(U_B_mix[2:])
print(U_C_mix[:2])

