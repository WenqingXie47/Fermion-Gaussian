from ncon import ncon
import numpy as np
from scipy.stats import unitary_group
from contract.many_body_unitary import from_mode_unitary_full


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


def product_state(n_modes, binary_list=None):
    if binary_list is None:
        binary_list = np.ones(n_modes,dtype="int")
    psi = np.zeros((2**n_modes),dtype="complex")
    index = np.sum(binary_list * 2**np.arange(len(binary_list)))
    psi[index] = 1
    return psi
    
def W_state(n_modes):
    psi = np.zeros((2**n_modes),dtype="complex")
    zero_list = np.zeros(n_modes,dtype="int")
    for i in range(n_modes):
        binary_list = zero_list.copy()
        binary_list[i] = 1
        index = np.sum(binary_list * 2**np.arange(len(binary_list)))
        psi[index] = 1
    psi = psi/np.sqrt(n_modes)
    return psi


def random_state(n_modes):
    real = np.random.rand((2**n_modes))
    imag = np.random.rand((2**n_modes))
    psi = real + 1j* imag
    psi = psi/ np.linalg.norm(psi)
    return psi


def test_pbc(local_U, dim_phys):
    for n_copy in range(1,6):
        u = contract_pbc(local_U, n_copy)
        u = u.reshape((dim_phys**n_copy, dim_phys**n_copy))
        print(np.linalg.svd(u)[1])
        # print(np.linalg.svd(u)[2])
        uh = u.conj().T
        id = u@uh
        err = error(id, np.eye(u.shape[0],dtype="complex"))
        print(n_copy, err)



n_modes_in = 2
n_modes_out = n_modes_in
n_modes_bond = 1
n_modes_mid = n_modes_out - n_modes_bond


dim_in = 2**n_modes_in
dim_out = dim_in
dim_bond = 2**n_modes_bond
dim_mid = 2**n_modes_mid

# u = unitary_group.rvs(n_modes)
# U = from_mode_unitary_full(u)

U1 = unitary_group.rvs(dim_in)
U1_tensor = U1.reshape((dim_mid, dim_bond, dim_in))

U2 = unitary_group.rvs(dim_out)
U2_tensor = U2.reshape((dim_out, dim_bond, dim_mid))

TensorArray = [U2_tensor, U1_tensor]
IndexArray = [[-1,-4,1],[1,-2,-3]]
U = ncon(TensorArray,IndexArray)

test_pbc(U,dim_out)
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




