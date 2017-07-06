import sys

import numpy as np
import cv2
import operator

im = cv2.imread('C:\\Users\\Dragisa\\Desktop\\PythonWorkspace\\SudokuSolutionTester\\sudoku.png')
im3 = im.copy()

gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

#################      Konture         ###################

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

samples = np.empty((0,100))
responses = []
keys = [i for i in range(48,58)]

counter = 0
for cnt in contours:
    if cv2.contourArea(cnt)> 35 and cv2.contourArea(cnt)< 130:
        [x,y,w,h] = cv2.boundingRect(cnt)
        counter = counter + 1
        print "X koordinata: " + str(x)
        print str(counter)
        if  h>10:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,0,255),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            cv2.imshow('norm',im)
            key = cv2.waitKey(0)

            if key == 27:  # (escape to quit)
                sys.exit()
            elif key in keys:
                responses.append(int(chr(key)))
                sample = roismall.reshape((1,100))
                samples = np.append(samples,sample,0)

responses = np.array(responses,np.float32)
responses = responses.reshape((responses.size,1))
print "trening gotov"

np.savetxt('generalsamples.data',samples)
np.savetxt('generalresponses.data',responses)


samples = np.loadtxt('generalsamples.data',np.float32)
responses = np.loadtxt('generalresponses.data',np.float32)
responses = responses.reshape((responses.size,1))

model = cv2.ml.KNearest_create()
model.train(samples,cv2.ml.ROW_SAMPLE,responses)

############################# Deo za testiranje  #########################


#sudoku.png - Idealan slucaj: 100% tacnosti
#sudokuTest2.png - Samo jedan broj nije popunjen: Izbacuje gresku!
#sudokuGreska.png - Sve je popunjeno, ali sudoku nije tacan!

im = cv2.imread('C:\\Users\\Dragisa\\Desktop\\PythonWorkspace\\SudokuSolutionTester\\sudokuGreska.png')
out = np.zeros(im.shape,np.uint8)
gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)

image, contours, hierarchy = cv2.findContours(thresh,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
counter = 0
numOfArrayElements = 0
sudokuMatrix = []
tempArray = {}
for cnt in contours:
    if cv2.contourArea(cnt)> 35 and cv2.contourArea(cnt)< 130:
        [x,y,w,h] = cv2.boundingRect(cnt)
        counter = counter + 1
        print "X koordinata: " + str(x)
        print str(counter)
        if h>10:
            cv2.rectangle(im,(x,y),(x+w,y+h),(0,255,0),2)
            roi = thresh[y:y+h,x:x+w]
            roismall = cv2.resize(roi,(10,10))
            roismall = roismall.reshape((1,100))
            roismall = np.float32(roismall)
            retval, results, neigh_resp, dists = model.findNearest(roismall, k = 1)
            string = str(int((results[0][0])))
            print "Ono sto vraca je :" + str(string)
            tempArray[x] = int((results[0][0]))

            numOfArrayElements = numOfArrayElements + 1
            cv2.putText(out,string,(x,y+h),0,1,(0,255,0))

            if numOfArrayElements == 9:
                numOfArrayElements = 0
                sortedByKeyTempArray = sorted(tempArray.items(), key=operator.itemgetter(0))
                sudokuMatrix.append(sortedByKeyTempArray)
                tempArray = {}

cv2.imshow('im',im)
cv2.imshow('out',out)
print str(len(sudokuMatrix))
print "Gledaju se redovi od dole ka gore"
for red in sudokuMatrix:
    suma = 0
    for par in red:
        suma += par[1]
    print "Suma reda " + str(red)+ " : " + str(suma)

print "Suma kockica"
indexReda = 0
indexKolone = 0
for j in range (3):
    for i in range (3):
        kockica = [[sudokuMatrix[indexReda][indexKolone][1], sudokuMatrix[indexReda][indexKolone+1][1],  sudokuMatrix[indexReda][indexKolone+2][1]],
                   [sudokuMatrix[indexReda+1][indexKolone][1], sudokuMatrix[indexReda+1][indexKolone+1][1], sudokuMatrix[indexReda+1][indexKolone+2][1]],
                   [sudokuMatrix[indexReda+2][indexKolone][1], sudokuMatrix[indexReda+2][indexKolone+1][1], sudokuMatrix[indexReda+2][indexKolone+2][1]]]
        suma1 = sudokuMatrix[indexReda][indexKolone][1] + sudokuMatrix[indexReda][indexKolone+1][1] + sudokuMatrix[indexReda][indexKolone+2][1]
        suma2 = sudokuMatrix[indexReda+1][indexKolone][1] + sudokuMatrix[indexReda+1][indexKolone+1][1] + sudokuMatrix[indexReda+1][indexKolone+2][1]
        suma3 = sudokuMatrix[indexReda+2][indexKolone][1] + sudokuMatrix[indexReda+2][indexKolone+1][1] + sudokuMatrix[indexReda+2][indexKolone+2][1]
        print "Konacna suma kockice: " + str(kockica) + ": " + str(suma1+suma2+suma3)
        indexKolone = indexKolone + 3
    indexKolone = 0
    indexReda = indexReda + 3
cv2.waitKey(0)