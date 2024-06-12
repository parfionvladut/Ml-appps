import cv2
import numpy as np
import matplotlib.pyplot as plt

def construct_matrix():
    A = np.zeros((10304, 320), 'i2')
    for i in range(1, 41):
        for j in range(1, 9):
            img_path = f'C:\\Users\\jacso\\Desktop\\ORL\\s{i}\\{j}.pgm'
            img = cv2.imread(img_path, 0)
            A[:, (i - 1) * 8 + j - 1: (i - 1) * 8 + j] = np.reshape(img, (-1, 1))
    return A

def Lanczos( A, v, m=100 ):
    n = len(v)
    if m>n: m = n;
    
    V = np.zeros( (m,n) )
    T = np.zeros( (m,m) )
    vo   = np.zeros(n)
    beta = 0
    for j in range( m-1 ):
        w    = np.dot( A, v )
        alfa = np.dot( w, v )
        w    = w - alfa * v - beta * vo
        beta = np.sqrt( np.dot( w, w ) ) 
        vo   = v
        v    = w / beta 
        T[j,j  ] = alfa 
        T[j,j+1] = beta
        T[j+1,j] = beta
        V[j,:]   = v
    w    = np.dot( A,  v )
    alfa = np.dot( w, v )
    w    = w - alfa * v - beta * vo
    T[m-1,m-1] = np.dot( w, v )
    V[m-1]     = w / np.sqrt( np.dot( w, w ) ) 
    return T, V

# generate matrix A
n = 320; m=10

A = construct_matrix()
print(A.shape)

#  full solve for eigenvalues for reference
esA, vsA = np.linalg.eig( A )


#  approximate solution by Lanczos
v0   = np.random.rand( n ) 
v0 /= np.sqrt( np.dot( v0, v0 ) )
print ("v0"); print (v0)
T, V = Lanczos( A, v0, m )
esT, vsT = np.linalg.eig( T )
VV = np.dot( V, np.transpose( V ) ) # check orthogonality


#print ("A : "); print (A)
print ("T : "); print (T)
#print ("V : "); print (V)
print ("VV :"); print (VV)
#print ("esA :"); print (np.sort(esA))
print ("esT : "); print (np.sort(esT))

#plt.plot( esA, np.ones(n)*0.2,  '+' )
plt.plot( esT, np.ones(m)*0.1,  '+' )
plt.ylim(0,1)
plt.show( m )