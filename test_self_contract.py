import numpy as np
from scipy.stats import unitary_group, ortho_group
from numpy import linalg as LA
from contract.mode_space_contract import contract,contract2


dim = 5
dim_contract=4
u = unitary_group.rvs(dim)
contracted_u= contract(u, dim_contract)


# verify that contraction gives unitary
id = contracted_u@ (contracted_u.conj().T)
assert(np.allclose(id, np.eye(id.shape[0])))


# verify that contraction multiple index = contract index one by one 
u2 = u.copy()
for i in range(dim_contract):
    u1 = contract(u,i+1)
    u2 = contract(u2, 1)
    assert(np.allclose(u1,u2))


u2 = u.copy()
for i in range(dim_contract):
    u1 = contract2(u,i+1)
    u2 = contract2(u2, 1)
    assert(np.allclose(u1,u2))













