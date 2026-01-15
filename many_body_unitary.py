
import numpy as np
from numpy import linalg as LA
from scipy.stats import unitary_group
from contract.mode_space_contract import contract, contract2
from contract.hilbert_space_contract import many_body_contract, many_body_contract2
from contract.many_body_unitary import from_mode_unitary_full




dim = 10
dim_contract = 5

u = unitary_group.rvs(dim)
U = from_mode_unitary_full(u)

# verify that many body U is really a unitary
Id = U @ U.conj().T
assert(np.allclose(Id, np.eye(Id.shape[0])))

# verify that contract multiple index == contract index one by one
U1 = U.copy()
for i in range(dim_contract):
    U1 = many_body_contract(U1,1)
    U2 = many_body_contract(U,i+1)
    
    assert(np.allclose(U1,U2))
    U3 = U1/U1[0,0]
    Id = U3@U3.conj().T
    assert(np.allclose(Id, np.eye(Id.shape[0])))

# U1 = U.copy()
# for i in range(dim_contract):
#     U1 = many_body_contract2(U1,1)
#     U2 = many_body_contract2(U,i+1)
#     assert(np.allclose(U1,U2))
#     U3 = U1/U1[0,0]
#     Id = U3@U3.conj().T
#     assert(np.allclose(Id, np.eye(Id.shape[0])))


ua = contract(u,dim_contract)
UA1 = from_mode_unitary_full(ua)
UA2 = many_body_contract(U,dim_contract)
ubb = u[-dim_contract:,-dim_contract:]
norm = LA.det(ubb+np.eye(ubb.shape[0]))
assert(np.allclose(UA2[0,0],norm))
UA2 = UA2/UA2[0,0]
# verify UA2 is really a unitary
Id = UA2 @ UA2.conj().T
assert(np.allclose(Id, np.eye(Id.shape[0])))
np.savetxt('matrix/unitary.txt',U,fmt="%.3f")
np.savetxt('matrix/contracted_unitary1.txt',UA1,fmt="%.3f")
np.savetxt('matrix/contracted_unitary2.txt',UA2,fmt="%.3f")
# verify UA1 == UA2
assert(np.allclose(UA1,UA2))



ua = contract2(u,dim_contract)
UA1 = from_mode_unitary_full(ua)
UA2 = many_body_contract2(U,dim_contract)
ubb = u[-dim_contract:,-dim_contract:]
norm = LA.det(np.eye(ubb.shape[0])-ubb)
assert(np.allclose(UA2[0,0],norm))
UA2 = UA2/UA2[0,0]
# verify UA2 is really a unitary
Id = UA2 @ UA2.conj().T
assert(np.allclose(Id, np.eye(Id.shape[0])))
np.savetxt('matrix/unitary.txt',U,fmt="%.3f")
np.savetxt('matrix/contracted_unitary1.txt',UA1,fmt="%.3f")
np.savetxt('matrix/contracted_unitary2.txt',UA2,fmt="%.3f")
# verify UA1 == UA2
assert(np.allclose(UA1,UA2))





