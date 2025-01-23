import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../code')
import numpy as np



from matrix import diag_real_skew, generate_random_real_skew_matrix



m = generate_random_real_skew_matrix(dim=4)

b,v = diag_real_skew(m)
assert(np.allclose(m, v @ b @ v.H ))
