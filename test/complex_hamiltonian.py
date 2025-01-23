import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '..')
import numpy as np


from code.complex_rep.hamiltonian import  Hamiltonian
from code.random_matrix_generator import random_hermitian_matrix

dim = 6
herm1 = random_hermitian_matrix(dim)
herm2 = random_hermitian_matrix(dim)
non_hermitian = herm1+herm2*1j


# random Hamiltonian
ham = Hamiltonian(herm1)
# eigen energy/state in descend order
eig_energy, eig_state = np.linalg.eig(herm1)
perm = np.argsort(eig_energy)[::-1]
eig_energy = eig_energy[perm]
eig_state = eig_state[:,perm]


# correlation matrix eigen value/vector in ascending order
corr_mixed = ham.to_mixed_state_correlation()
eig_value, eig_vector = np.linalg.eig(corr_mixed)
perm = np.argsort(eig_value)
eig_value = eig_value[perm]
eig_vector = eig_vector[:,perm]

assert(np.allclose(1/(1+np.exp(eig_energy)), eig_value))
assert(np.allclose(eig_state,eig_vector))


ham = np.diag([1,-2,3,-4,5,-6])
ham = Hamiltonian(ham)
corr = ham.to_ground_state_correlation()
assert(np.allclose(corr, np.diag([0,1,0,1,0,1])))



