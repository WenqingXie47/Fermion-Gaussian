import numpy as np
from numpy import linalg as LA

def dirac_hamiltonian(A,B):
    pass


def eigen_decomposition(matrix):
    eigenvalues, eigenvectors = LA.eig(matrix)
    diag_matrix = np.diag(eigenvalues)
    return diag_matrix, eigenvectors


class Hamiltonian:

    def __init__(self, value):
        self.value = value

    
    def to_diagonal(self):
        ham_diag, U = eigen_decomposition(self.value)
        return ham_diag, U
    

    # correlation matrix is defined by <aa^dagger>
    def to_correlation(self, beta=1.0):
        # ham_diag is an 1D array containing eigenvalues of hamiltonian
        ham_diag, U = LA.eig(self.value) 
        correlation_diag = 1/(1+np.exp(2*ham_diag*beta))
        correlation = U @ np.diag(correlation_diag) @ U.H
        return correlation


    # correlation matrix is defined by <aa^dagger>
    # zero temperature, ground state case
    def to_correlation_ground_state(self):
        # ham_diag is an 1D array containing eigenvalues of hamiltonian
        ham_diag, U = LA.eig(self.value)
        # 
        correlation_diag = np.where(ham_diag<0, 0, 1)
        correlation = U @ np.diag(correlation_diag) @ U.H
        return correlation
    






