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
UA = np.trace(U_tp, axis1=0, axis2=2)

print(UA @ (UA.conj().T))



