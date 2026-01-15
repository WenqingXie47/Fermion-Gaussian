from ncon import ncon
import numpy as np
from scipy.stats import unitary_group
from contract.many_body_unitary import from_mode_unitary_full
from MPU.contract import contract_obc, contract_pbc, pbc_from_obc
import matplotlib
matplotlib.use('TkAgg')  # Or 'Agg' for non-interactive (saves plots to files only)
from matplotlib import pyplot as plt

def XY_gate(theta):
    c = np.cos(theta)
    s = np.sin(theta)
    xygate = np.array(
        [[1, 0, 0, 0],
         [0, c, s*1j, 0],
         [0, s*1j, c, 0],
         [0, 0, 0, 1]
        ]
    )
    return xygate

def CNOT_gate():
    cnot = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]
        ]
    )
    return cnot



def error(A,B):
    diff = A-B
    spectral_norm = np.linalg.norm(diff, ord=2)
    return spectral_norm

def deviation_from_id(A):
    trace = np.trace(A)
    id = np.eye(A.shape[0],dtype="complex") * (trace/A.shape[0])
    spectral_norm = np.linalg.norm(A-id, ord=2)
    return spectral_norm

def error_from_id(A):

    id = np.eye(A.shape[0],dtype="complex")
    return np.linalg.norm(A-id, ord=2)





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

U = XY_gate(theta=(np.pi/4))
# U = CNOT_gate()
U_tensor = U.reshape((dim_phys, dim_bond, dim_phys, dim_bond))

# for n_copy in range(1,10):
#     u = contract_obc(U_tensor, n_copy)
#     u = u.reshape((dim_phys**n_copy * dim_bond, dim_phys**n_copy * dim_bond))
#     uh = u.conj().T
#     id = u@uh
#     err = error(id, np.eye(u.shape[0],dtype="complex"))
#     print(n_copy, err)

err_list = []
for n_copy in range(1,12):
    u = contract_pbc(U_tensor, n_copy)
    u = u.reshape((dim_phys**n_copy, dim_phys**n_copy))
    uh = u.conj().T
    id = u@uh
    err = error_from_id(id)
    err_list.append(err)
    print(n_copy, err)


err_list = np.array(err_list)
log_err = np.log(err_list)
fig, ax = plt.subplots()
# ax.plot(err_list)
# fig.savefig("./figures/exponential_decay.png")

ax.plot(log_err)
ax.set_ylabel(r"$\ln \|UU^\dagger - I \|$")
ax.set_xlabel(r"System Size $L$")   
fig.savefig("./figures/log_exponential_decay.png")





