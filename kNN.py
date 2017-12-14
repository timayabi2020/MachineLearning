from numpy import * # Scientific computing package
import operator # package for sorting
from os import listdir
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group, labels

def classify0(inX, dataset, labels, k):
    dataSetSize = dataset.shape[0] # Getting the shape of the array i.e No of rows in the array
    diffMat = tile(inX, (dataSetSize, 1)) - dataset # tile is a numpy array iterator
    # for building a matrix. In this case inX is shaped into a 2 * 4 matrix.
    # Telling it to repeat for the column only once while for the rows repeat only four times as per the group array size
    # Subtraction operation is
    #  done on the matrix by subtracting each element of the gropup matrix from inX matrix
    #Calculating euclidean distance i.e d = sqrt((xA0 - xB0)squared - (xA1 - xB1)squared)
    sqDiffMat = diffMat ** 2
    #get sum of distances by adding row elements
    sqDistances = sqDiffMat.sum(axis=1)

    #Getting the square root of each sum
    distances = sqDistances ** 0.5
    #sort distances in increasing order
    sortedDistIndicies = distances.argsort()
    #Create a dictionary
    classCount = {}
     #k is the number of training examples to be used
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
    key=operator.itemgetter(1), reverse=True)

    return sortedClassCount[0][0]

def file2matrix(filename):
    fr = open(filename)
    numberOfLines = len(fr.readlines())         #get the number of lines in the file
    returnMat = zeros((numberOfLines,3))        #prepare matrix to return
    classLabelVector = []                       #prepare labels return
    fr = open(filename)
    index = 0
    for line in fr.readlines():
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if (listFromLine[-1] == 'largeDoses'):
            classLabelVector.append("largeDoses")
        elif (listFromLine[-1] == 'smallDoses'):
            classLabelVector.append("smallDoses")
        elif (listFromLine[-1] == 'didntLike'):
            classLabelVector.append("didntLike")
        index += 1
    return returnMat,classLabelVector
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))
    return normDataSet, ranges,minVals
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],
                                     datingLabels[numTestVecs:m],3)
        print "The classifier came back with: %s, the real number is: %s " % (classifierResult,datingLabels[i])
        if (classifierResult != datingLabels[i]) : errorCount += 1.0
    print "The total error rate is: %f " % (errorCount/float(numTestVecs))

def classifyPerson():
    resultList = ['not at all','in small doses','in large doses']
    percentTats = float(raw_input("Percentage of time spent playing video games?"))
    ffmiles = float(raw_input("Frequent flier miles earned per year?"))
    icecream = float(raw_input("Litres of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffmiles,percentTats,icecream])
    classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print "You will probably like this person: ", resultList[classifierResult - 1]

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32 * i+j] = int(lineStr[j])
    return returnVect

def handwrittingClassTest():
    hwLabels = []
    trainingFileList = listdir('digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))

    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = img2vector('digits/trainingDigits/%s'% fileNameStr)
    testFileList = listdir('digits/testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = fileStr.split('_')[0]
        vectorUnderTest = img2vector('digits/trainingDigits/%s'% fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)

        print "The classifier came back with: %s, the real answer is %s" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr):
            errorCount+=1.0
    print "\nThe total number of errors is: %d" % errorCount
    print "\nThe total error rate is: %f " % (errorCount/float(mTest))

