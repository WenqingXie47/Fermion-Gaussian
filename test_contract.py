import numpy as np
from scipy.stats import unitary_group, ortho_group
from numpy import linalg as LA


threshold = 0.0001
def geometric_sum(M, n):
    result = np.zeros_like(M)
    for i in range(n):
        result += LA.matrix_power(M,i)
    return result


dim = 5
dim_A = 3
dim_B = dim-dim_A
U = unitary_group.rvs(dim)
U_AA = U[:dim_A,:dim_A]
U_BB = U[dim_A:,dim_A:]
U_AB = U[:dim_A,dim_A:]
U_BA = U[dim_A:,:dim_A]
Id = np.eye(dim)
# projectors
PA =  Id[:,:dim_A] @ Id[:,:dim_A].conj().T 
PB = Id[:,dim_A:] @ Id[:,dim_A:].conj().T 



value, vector = LA.eig(U @ PA)
print(value)

I_AA = np.eye(dim_A)
id = geometric_sum((U_AA.conj().T), 50) @ (I_AA - (U_AA.conj().T @ U_AA) ) @ geometric_sum(U_AA, 50)
id = U_AB.conj().T @ id @ U_AB + U_BB.conj().T @ U_BB
print(id)




