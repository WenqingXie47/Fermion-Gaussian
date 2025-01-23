from ncon import ncon
import numpy as np
from scipy.stats import unitary_group


def contract_pbc(M, n_copy):
    TensorArray = [M]*n_copy
    IndexArray = []
    for i in range(n_copy):
        index = [-i,i%n_copy,-(i+n_copy),(i+1)%n_copy]
        IndexArray.append(index)
    contracted_tensor = ncon(TensorArray,IndexArray)
    return contracted_tensor


n_modes_phys = 2
n_modes_bond = 2

dim_phys = 2**n_modes_phys
dim_bond = 2**n_modes_bond

dim = dim_phys * dim_bond

U = unitary_group.rvs(dim)
U_tensor = U.reshape((dim_phys, dim_bond, dim_phys, dim_bond))

for n_copy in range(4):
    u = contract_pbc(U_tensor, n_copy)
    print(u.shape)


