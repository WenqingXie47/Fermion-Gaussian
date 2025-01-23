import numpy as np
from scipy.stats import unitary_group

def even_odd_basis(n_modes):
    even_index = []
    odd_index = []
    
    for i in range(2**n_modes):
        binary_number = np.binary_repr(i)
        bit_sum = sum(int(bit) for bit in binary_number)
        if bit_sum%2 ==0:
            even_index.append(i)
        else:
            odd_index.append(i)
    index = even_index + odd_index
    index  = np.array(index)
    return index


n_modes_A = 3
n_modes_B = 3
n_modes = n_modes_A + n_modes_B

dim_Hilbert_A = 2**n_modes_A
dim_Hilbert_B = 2**n_modes_B
dim_Hilbert = dim_Hilbert_A * dim_Hilbert_B

U_even = unitary_group.rvs(dim_Hilbert//2)
U_odd = unitary_group.rvs(dim_Hilbert//2)
U = np.block(
    [[U_even, np.zeros_like(U_even)],
     [np.zeros_like(U_even), U_odd]]
)
print(U.shape)
index = even_odd_basis(n_modes)
reverse_index = np.argsort(index)
U = U[reverse_index,:][:,reverse_index]

U_tp = U.reshape((dim_Hilbert_A, dim_Hilbert_B, dim_Hilbert_A, dim_Hilbert_B))
UA = np.trace(U_tp, axis1=0, axis2=2)
index_A = even_odd_basis(n_modes_A)
UA= UA[index_A,:][:,index_A]
IA = UA @ (UA.conj().T)
print(IA[:4,:4])




