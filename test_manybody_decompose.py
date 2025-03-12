import numpy as np
from scipy.stats import unitary_group, ortho_group
from numpy import linalg as LA


n_modes_A = 2
n_modes_B = 2

dim_Hilbert_A = 2**n_modes_A
dim_Hilbert_B = 2**n_modes_B
dim_Hilbert = dim_Hilbert_A * dim_Hilbert_B

U = unitary_group.rvs(dim_Hilbert)
U_tp = U.reshape((dim_Hilbert_A, dim_Hilbert_B, dim_Hilbert_A, dim_Hilbert_B))
# UAA = np.trace(U_tp, axis1=1, axis2=3)
# UBB = np.trace(U_tp, axis1=0, axis2=2)



UAA = np.sum(U_tp, axis=(1,3))
UBB = np.sum(U_tp, axis=(0,2))


UA, SA, VhA = LA.svd(UAA, full_matrices=True)
UB, SB, VhB = LA.svd(UBB, full_matrices=True)

print(SA)
print(SB)



