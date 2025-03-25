import numpy as np
import matplotlib.pyplot as plt

def get_M(N):
    M=np.zeros((2*N,2*N))
    first_diag_above=np.array([2-i%2 for i in range(2*N-1)])
    third_diag_above=np.array([-((i+1)%2) for i in range(2*N-3)])
    M+=np.diag(first_diag_above, k=1)+np.diag(-first_diag_above, k=-1)+np.diag(third_diag_above, k=3)+np.diag(-third_diag_above, k=-3)
    #with periodic boundary conditions
    M[0,2*N-1]=-1
    M[1,2*N-2]=1
    M[2*N-2,1]=-1
    M[2*N-1,0]=1
    return M
