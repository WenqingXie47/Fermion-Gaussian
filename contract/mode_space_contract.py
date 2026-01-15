import numpy as np
from scipy.stats import unitary_group

def contract2(unitary, dim_contract):
    dim = unitary.shape[0]
    dim_free = dim-dim_contract

    U_AA = unitary[:dim_free,:dim_free]
    U_BB = unitary[dim_free:,dim_free:]
    U_AB = unitary[:dim_free,dim_free:]
    U_BA = unitary[dim_free:,:dim_free]
    I_BB = np.eye(U_BB.shape[0]) 
    contracted_u =  U_AA + U_AB @ np.linalg.pinv(I_BB-U_BB) @ U_BA 
    return contracted_u


def contract(unitary, dim_contract):
    dim = unitary.shape[0]
    dim_free = dim-dim_contract

    U_AA = unitary[:dim_free,:dim_free]
    U_BB = unitary[dim_free:,dim_free:]
    U_AB = unitary[:dim_free,dim_free:]
    U_BA = unitary[dim_free:,:dim_free]
    I_BB = np.eye(U_BB.shape[0]) 
    contracted_u =  U_AA - U_AB @ np.linalg.pinv(I_BB+U_BB) @ U_BA 
    return contracted_u

if __name__ == "__main__":
    n_modes = 4
    n_contracted_modes = 2
    U = unitary_group.rvs(n_modes)

    UA1 = contract2(U,n_contracted_modes)
    UA2 = contract(U,n_contracted_modes)

    id = UA1 @ UA1.conj().T
    assert(np.allclose(id, np.eye(id.shape[0])))
    id = UA2 @ UA2.conj().T
    assert(np.allclose(id, np.eye(id.shape[0])))


    u2 = U
    for i in range(n_contracted_modes):
        u1 = contract(U,i+1)
        u2 = contract(u2, 1)
        assert(np.allclose(u1,u2))
