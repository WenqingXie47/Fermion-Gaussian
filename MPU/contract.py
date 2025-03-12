import numpy as np
from ncon import ncon

def contract_pbc(M, n_copy):
    TensorArray = [M]*n_copy
    IndexArray = []
    for i in range(n_copy):
        index = [-(i+1),((i+1)%n_copy)+1,(i%n_copy)+1, -(i+n_copy+1)]
        IndexArray.append(index)
    print(IndexArray)
    contracted_tensor = ncon(TensorArray,IndexArray)
    return contracted_tensor
    
def contract_obc(M, n_copy):
    TensorArray = [M]*n_copy
    IndexArray = []
    for i in range(1,n_copy+1):
        index = [-i,i,i-1,-(i+n_copy+2)] # [A_out, B_out, A_in, B_in]
        IndexArray.append(index)
    IndexArray[n_copy-1][1] = -(n_copy+1)
    IndexArray[0][2] = -(n_copy+2)
    
    print(IndexArray)
    contracted_tensor = ncon(TensorArray,IndexArray)
    return contracted_tensor


def pbc_from_obc(M, n_copy):
    return np.trace(M, axis1=n_copy, axis2=n_copy+1)
