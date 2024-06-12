import cv2
import numpy as np
import scipy as sp
import scipy.linalg as la
import numpy.linalg as npl
import random
from scipy.spatial import distance
from matplotlib import pyplot as plt
A=np.random.random((500,2))
#A=[[12,15],[23,14],[2,5],[23,9],[12,5],[12,4],[12,8],[12,3],[22,13],[12,43],[22,83],[42,12],[12,11]]
A=A.tolist()
k=3

def centeroidnp(arr):#calculate centroid 
    length = len(arr)
    arr = np.array(arr)
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def ploting(c2,A,m):    
    c2 = np.array(c2)    
    A = np.array(A)
    
    for n in range(len(A)):
        
        if A[n, 2]==1:
                plt.plot(A[n, 0],A[n, 1],'ro')
       
        if A[n, 2]==2:
                plt.plot(A[n, 0],A[n, 1],'yo')
                       
        if A[n, 2]==3:
                  plt.plot(A[n, 0],A[n, 1],'go')
    for n in range(k):
        if m[n]==1:
            plt.plot(c2[n, 0],c2[n, 1],'rs')

        if m[n]==2:
            plt.plot(c2[n, 0],c2[n, 1],'ys')

        if m[n]==3:
            plt.plot(c2[n, 0],c2[n, 1],'gs')
    plt.axis([0, max(A[:, 0]), 0, max(A[:, 1])])
    plt.show()
    
def dist(A,c2):#distance to the centroid
    A = np.array(A)
    
    for n in range (len(A)):
       # print(A[n,0:2])
        if distance.euclidean(A[n,0:2], c2[0])<distance.euclidean(A[n,0:2], c2[1]) and distance.euclidean(A[n,0:2], c2[0])<distance.euclidean(A[n,0:2], c2[2]):
            A[n,2]=1
          #  print("1")
        elif distance.euclidean(A[n,0:2], c2[1])<distance.euclidean(A[n,0:2], c2[0]) and distance.euclidean(A[n,0:2], c2[1])<distance.euclidean(A[n,0:2], c2[2]):
                A[n,2]=2
              #  print("2")
        elif distance.euclidean(A[n,0:2], c2[2])<distance.euclidean(A[n,0:2], c2[0]) and distance.euclidean(A[n,0:2], c2[2])<distance.euclidean(A[n,0:2], c2[1]):
                    A[n,2]=3
                   # print("3")
    #print("sss")
    return A

def arrange(A):#order the points in the array
    c1=[]
    n=1
    for n in range (k+1):
        c1.append([])
        for x in range (len(A)):    
            if A[x][2]==n:
                c1[n-1].append(A[x])
    return c1

def centroid(c1):#compute centroid
    c2=[]   
    for n in range (k):
        c2.append(centeroidnp(c1[n]))
    return c2
       
for x in range (len(A)):#atribute random clusters
    A[x].append(random.randint(1,k))
   # print(A[x])
 #   print(A[x][2])

m=[1,2,3]
z=0
y=0
while z!=1:
    a1=A  
    c1=arrange(A)
    c2=centroid(c1)    
    ploting(c2,A,m)    
    A=dist(A,c2)
    a2=A
    ploting(c2,A,m)
    comparison = a1 == a2
    equal_arrays = comparison.all() 
    print(y) 
    y=y+1
    if equal_arrays==True:        
        z=z+1
   


    