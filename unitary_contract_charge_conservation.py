import numpy as np
from contract.many_body_unitary import random_unitary_charge_conserved
from contract.hilbert_space_contract import many_body_contract


n_modes = 5
n_contracted_modes = 2
U = random_unitary_charge_conserved(n_modes)
UA = many_body_contract(U, n_contracted_modes)

id = UA @ UA.conj().T
print(id)
