import numpy as np

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