#ver1
#from numpy import asarray
import numpy as np
import cv2
import numpy.linalg as npl
from PIL import Image
#cale=r'C:\Users\jacso\Desktop\ORL\s1\1.pgm'
#img=cv2.imread(cale,0)
#cv2.imshow('img',img)
#data=asarray(img)
#print(data)
#V = np.asarray(data).reshape(-1)
#print(V.shape)

A=np.zeros((10304,320),'i2')
for i in range(1, 41):
    for j in range(1, 9):
        img=cv2.imread(f'C:\\Users\\jacso\\Desktop\\ORL\\s{i}\\{j}.pgm',0)
        A[:, (i-1)*8+j-1 : (i-1)*8+j]=np.reshape(img,(-1,1))
        print(A)

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
picture = cv2.imread(f'C:\\Users\\jacso\\Desktop\\ORL\\s{perschoice}\\{imgchoice}.pgm', 0)
cv2.imshow('img',picture)
test_pic_1column = picture.reshape((-1, 1)) 
norm_results = np.zeros((1, 320), 'i')
for i in range(1, 321):
    diff_column = A[:, i - 1:i]- test_pic_1column
    norm_results[0, i - 1] = npl.norm(diff_column, 2)
    
index = np.argmin(norm_results)
print(f'Index: {index}')
resultPic = A[:, index].reshape((112, 92))
img = Image.fromarray(resultPic)
img.show()


cv2.waitKey(0)
cv2.destroyAllWindows()

