#!/usr/bin/python
import cv2
import numpy as np
import scipy as sp
import scipy.linalg as la
import numpy.linalg as npl

def construct_matrix():
    A=np.zeros((10304,320),'i2')
    for i in range(1, 41):
        for j in range(1, 9):
            img=cv2.imread(f'C:\\Users\\jacso\\Desktop\\ORL\\s{i}\\{j}.pgm',0)
            A[:, (i-1)*8+j-1 : (i-1)*8+j]=np.reshape(img,(-1,1))
            return(A)

while True:        
     print("Please input the person's valid number (1 to 40):")
     perschoice=int(input())
     if( perschoice<41 and perschoice>0):
        break
while True:        
     print("Please input the picture valid number (9 or 10):")
     imgchoice=int(input())
     if( imgchoice<11 and imgchoice>8):
        break
    
def EFpreproc(A,k):#pre-processing
    m,n=A.shape
    A=A.astype(np.float64)
    media=np.mean(A,axis=0)
    A-=media
    if n>m:
        L=np.dot(A,A.T)
    else:
        L=np.dot(A.T,A)
    ev,eV=np.linalg.eig(L)
    ev=ev.real.astype(np.float64)
    eV=eV.real.astype(np.float64)
    eV=eV[:,np.argsort(ev)]
    if(n>m):
        eV=np.dot(A.T,eV)
    V=eV[:,::-1][:,0:k]
    proiectie=np.dot(A,V)
    return (V,media,proiectie,L)

def EFquery(V, L, media, proiectie, poza):#testing
    n=proiectie.shape[0]
    poza=poza.astype(np.float64)
    poza=arrtovector(poza)
    
    #print(poza.shape)
   # print(media.shape)
    poza-=media  
    poza_pro=np.dot(poza.T,V)
    z=la.norm(proiectie-poza_pro,ord=1,axis=1)
    print(n)
    mz=min(z)#syntax error
    #print(mz)
 
    iPoza=[int(i) for i , x in enumerate(z) if x == mz]
    return L[iPoza]

def arrtovector(poza):
    norm_results = np.zeros((320, ), 'i')
   
    test_pic_1column = poza.reshape((-1,1))
    for i in range(320, 0):
        diff_column = A[:, i - 1:i]- test_pic_1column
        norm_results[0, i - 1] = npl.norm(diff_column, 2)
        print(i)
    norm_results = norm_results.astype('float32')
    
    #print(norm_results.shape)
    #print(norm_results)
    
    return norm_results
 
if __name__ == '__main__':
    A=construct_matrix()
    k = input('input k: ')
    try:
            k = int(k)
    except ValueError:
        print('K is not an integer.')
        exit(1)
    preprocessing = EFpreproc(A, k)
    picture = cv2.imread(f'C:\\Users\\jacso\\Desktop\\ORL\\s{perschoice}\\{imgchoice}.pgm', 0)
   # cv2.imshow('img',picture)
    result = EFquery(preprocessing[0], preprocessing[3], preprocessing[1], preprocessing[2], picture)
    print(f'Person:\n{result}')
        