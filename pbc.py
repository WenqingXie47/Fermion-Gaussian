from ncon import ncon
import numpy as np
from scipy.stats import unitary_group
from contract.many_body_unitary import from_mode_unitary_full
from contract.state import product_state, random_state, W_state


def contract_pbc(M, n_copy):
    TensorArray = [M]*n_copy
    IndexArray = []
    for i in range(n_copy):
        index = [-(i+1),((i+1)%n_copy)+1,-(i+n_copy+1),(i%n_copy)+1]
        IndexArray.append(index)
    print(IndexArray)
    contracted_tensor = ncon(TensorArray,IndexArray)
    return contracted_tensor
    
def contract_obc(M, n_copy):
    TensorArray = [M]*n_copy
    IndexArray = []
    for i in range(1,n_copy+1):
        index = [-i,i,-(i+n_copy+1),i-1]
        IndexArray.append(index)
    IndexArray[n_copy-1][1] = -(n_copy+1)
    IndexArray[0][3] = -(2*n_copy+2)
    

    print(IndexArray)
    contracted_tensor = ncon(TensorArray,IndexArray)
    return contracted_tensor


def error(A,B):
    diff = A-B
    spectral_norm = np.linalg.norm(diff, ord=2)
    return spectral_norm

def deviation_from_id(A):
    trace = np.trace(A)
    id = np.eye(A.shape[0],dtype="complex") * (trace/A.shape[0])
    spectral_norm = np.linalg.norm(A-id, ord=2)
    return spectral_norm



def test_pbc():
    for n_copy in range(1,10):
        u = contract_pbc(U_tensor, n_copy)
        u = u.reshape((dim_phys**n_copy, dim_phys**n_copy))
        print(np.mean(np.linalg.svd(u)[1]))
        # print(np.linalg.svd(u)[2])
        p_state = product_state(n_modes= n_modes_phys*n_copy).T
        qubits = np.zeros(n_modes_phys*n_copy,dtype="int")
        qubits[0] = 1
        p_state2 = product_state(n_modes= n_modes_phys*n_copy,).T
        w_state = W_state(n_modes=n_modes_phys*n_copy).T
        r_state = random_state(n_modes=n_modes_phys*n_copy).T
        print("norm ",np.linalg.norm(u@p_state),np.linalg.norm(u@w_state),np.linalg.norm(u@r_state))
        print("inner product ", (np.dot(u@p_state, u@p_state2)))
        uh = u.conj().T
        id = u@uh
        err = deviation_from_id(id)
        print(n_copy, err)


def test_obc():
    for n_copy in range(1,6):
        u = contract_obc(U_tensor, n_copy)
        u = u.reshape((dim_phys**n_copy * dim_bond, dim_phys**n_copy * dim_bond))
        uh = u.conj().T
        id = u@uh
        err = error(id, np.eye(u.shape[0],dtype="complex"))
        print(n_copy, err)

n_modes_phys = 1
n_modes_bond = 1
n_modes = n_modes_phys + n_modes_bond



dim_phys = 2**n_modes_phys
dim_bond = 2**n_modes_bond
dim = dim_phys * dim_bond

# u = unitary_group.rvs(n_modes)
# U = from_mode_unitary_full(u)

U = unitary_group.rvs(dim)
U_tensor = U.reshape((dim_phys, dim_bond, dim_phys, dim_bond))

# U_phys = unitary_group.rvs(dim_phys)
# U_bond = unitary_group.rvs(dim_bond)
# U_tensor = np.tensordot(U_phys, U_bond,axes=0)
# # local cases, bond and phys decouple
# U_tensor = np.transpose(U_tensor, (0,2,1,3))
# # the translation case 
# U_tensor = np.transpose(U_tensor, (0,2,3,1))




# for n_copy in range(2,4):
#     u = contract_pbc(U_tensor, n_copy)
#     print(u.shape)




