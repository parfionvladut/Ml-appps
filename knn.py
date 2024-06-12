import cv2
import numpy as np
import numpy.linalg as npl
from PIL import Image


def construct_matrix():
    A = np.zeros((10304, 320), 'i2')
    for i in range(1, 41):
        for j in range(1, 9):
            img_path = f'C:\\Users\\jacso\\Desktop\\ORL\\s{i}\\{j}.pgm'
            img = cv2.imread(img_path, 0)
            A[:, (i - 1) * 8 + j - 1: (i - 1) * 8 + j] = np.reshape(img, (-1, 1))
    return A





def test_picture_knn(picture: np.ndarray, k: int):
    test_pic_1column = picture.reshape((-1, 1))  # transform picture into column array
    norm_results = np.zeros((1, 320), 'i')  # initialize vector of norm results
    for i in range(1, 321):  # from 1-320
        # substract the test picture from all other pictures and calculate the norms
        diff_column = pics_matrix[:, i - 1:i] - test_pic_1column
        norm_results[0, i - 1] = npl.norm(diff_column, 2)

    index_matrix = np.zeros((1, 320), 'i')
    for i in range(0, 320):
        index_matrix[0, i] = i + 1
    # add 2nd row with indexes and then sort columns by 1st row
    norm_results = np.concatenate((norm_results, index_matrix))[:, norm_results[0].argsort()]
    print(f'Norm matrix:\n{norm_results}\n')

    classes_array = norm_results[1, 0:k]
    for i, e in enumerate(classes_array):
        if e % 8 == 0:
            classes_array[i] = int(e / 8)
        else:
            classes_array[i] = int(e / 8) + 1
    print(f'classes array: {classes_array}')
    count_array = np.bincount(classes_array)
    print(f'count array:{count_array}')

    maximum_array = np.argmax(count_array)
    print(f'maximum: {maximum_array}')
    return maximum_array


if __name__ == '__main__':
    # ORL path
    path = 'C:\\Users\\jacso\\Desktop\\ORL'
    # get matrix of pictures
    pics_matrix = construct_matrix()
    print(f'Pics matrix:\n{pics_matrix}')
    print(f'Pics matrix shape: {pics_matrix.shape}\n')

    # load testing picture
    while True:        
     print("Please input the person's valid number (1 to 40):")
     test_pic_s=int(input())
     if( test_pic_s<41 and test_pic_s>0):
        break
    while True:        
     print("Please input the picture valid number (9 or 10):")
     test_pic_img=int(input())
     if( test_pic_img<11 and test_pic_img>8):
        break
    test_pic = cv2.imread(f'C:\\Users\\jacso\\Desktop\\ORL\\s{test_pic_s}\\{test_pic_img}.pgm', 0)
    #cv2.imshow('img',test_pic)
    k = input('input k: ')
        
    try:
     k = int(k)
    except ValueError:
            print('K is not an integer.')
            exit(1)
    result = test_picture_knn(test_pic, int(k))
    
    print(f'Person:\n{result}')
    resultPic = pics_matrix[:, result].reshape((112, 92))
    img = Image.fromarray(resultPic)
    img.show()
