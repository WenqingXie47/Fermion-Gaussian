import numpy as np
from numpy import linalg as LA


def swapping_modes(matrix, order):
    assert matrix.shape[0] == matrix.shape[1]
    m = matrix[order]
    m[:,:] = m[:,order]
    return m


# C_ij = <c^dagger_i c_j>
class CorrelationMatrix:

    def __init__(self, matrix):
        assert(CorrelationMatrix.is_physical(matrix))
        self.value = matrix
        

    @staticmethod
    def is_physical(correlation_matrix):
        eig_values, eig_vectors = LA.eig(correlation_matrix)
        if (np.any(eig_values>1)):
            return False
        if (np.any(eig_values<0)):
            return False
        return True
    
    @staticmethod
    def is_pure(correlation_matrix):
        eig_values, eig_vectors = LA.eig(correlation_matrix)
        return (np.all(eig_values==0 or eig_values==1))

    def to_hamiltonian(self):
        # ham_diag is an 1D array containing eigenvalues of hamiltonian
        corr_diag, U = LA.eig(self.value) 
        ham_diag = np.log((1-corr_diag)/corr_diag)
        ham = U @ np.diag(ham_diag) @ U.H
        return ham
    

    def get_covariance_matrix(self):
        correlation =  self.value
        G = np.identity(like=self.value)
        Ommega = -1j* (2*correlation-G)
        return Ommega





