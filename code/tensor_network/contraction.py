import numpy as np


import numpy as np

def is_isometry(matrix):

    M = matrix
    # Compute the hermitian conjugate of the matrix
    Mh = M.T.conj()
    # Multiply the hermitian conjugate by the original matrix
    product = Mh @ M
    # Create an identity matrix of the same size
    identity = np.eye(M.shape[1])
    # Check if the product is approximately equal to the identity matrix
    return np.allclose(product, identity)


def check_legal_contracted_index(isometry1, isometry2, index1, index2):

    M1 = isometry1
    M2 = isometry2

    # check there are no repeated index
    assert(len(index1) == len(set(index1)))
    assert(len(index2) == len(set(index2)))

    # check index number to be contracted are same for 1 and 2
    assert(len(index1)==len(index2))

    # check index number no lager than shape of unitary
    dim_in1 = M1.shape[1]
    dim_out2 = M2.shape[0]
    assert(len(index1) <= dim_in1)
    assert(len(index2) <= dim_out2)



def get_complement_index(n_dim, index_list):

    # check there are no repeated index
    assert(len(index_list) == len(set(index_list)))

    all_index = np.arange(n_dim)

    complement_mask = ~np.isin(all_index, index_list) 
    complement_index_list = all_index[complement_mask]
    return complement_index_list


def contract(isometry1, isometry2, index1, index2):

    # first check all matrix fullfill requirement
    check_legal_contracted_index(isometry1, isometry2, index1, index2)
    assert(is_isometry(isometry1))
    assert(is_isometry(isometry2))
    
    M1 = isometry1
    M2 = isometry2

    dim_in1 = M1.shape[1]
    dim_out2 = M2.shape[0]

    complement_index1 = get_complement_index(dim_in1, index1)
    complement_index2 = get_complement_index(dim_out2, index2)


    


   



    