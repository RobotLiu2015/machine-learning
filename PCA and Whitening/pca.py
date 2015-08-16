import numpy
import math
import time
import scipy.io
import scipy.optimize
import matplotlib.pyplot
import warnings
warnings.simplefilter("ignore", numpy.ComplexWarning)

def zeroMean(dataMat,row):
    meanVal=numpy.mean(dataMat,axis=1)     #按行求均值，即求各个特征的均值
    meanVal = numpy.reshape(meanVal,(row,1))
    newData=dataMat-meanVal
    return newData

def loadDataset(num_patches, patch_side):#num_patches为列数据，patch_side*patch_side为行数目，一行代表一个维度。

    """ Load images into numpy array """

    images = scipy.io.loadmat('IMAGES_RAW.mat')
    images = images['IMAGESr']

    """ Initialize dataset as array of zeros """

    dataset = numpy.zeros((patch_side*patch_side, num_patches))

    """ Initialize random numbers for random sampling of images
        There are 10 images of size 512 X 512 """

    rand = numpy.random.RandomState(int(time.time()))
    image_indices = rand.randint(512 - patch_side, size = (num_patches, 2))
    image_number  = rand.randint(10, size = num_patches)

    """ Sample 'num_patches' random image patches """

    for i in range(num_patches):

        """ Initialize indices for patch extraction """

        index1 = image_indices[i, 0]
        index2 = image_indices[i, 1]
        index3 = image_number[i]

        """ Extract patch and store it as a column """

        patch = images[index1:index1+patch_side, index2:index2+patch_side, index3]
        patch = patch.flatten()
        dataset[:, i] = patch

    """ Normalize and return the dataset """

    dataset = zeroMean(dataset,patch_side*patch_side)
    return dataset

def visualize(dataset, nums, vis_patch_side):
    figure, axes = matplotlib.pyplot.subplots(nrows = nums, ncols = nums)
    index = 0

    for axis in axes.flat:

        """ Add row of weights as an image to the plot """

        axis.imshow(dataset[:,index].reshape(vis_patch_side, vis_patch_side),
                            cmap = matplotlib.pyplot.gray(), interpolation = 'nearest')
        axis.set_frame_on(False)
        axis.set_axis_off()
        index += 1
    """ Show the obtained plot """
    matplotlib.pyplot.show()

def percentage2n(eigVals,percentage):
    sortArray=numpy.sort(eigVals)   #升序
    sortArray=sortArray[-1::-1]  #逆转，即降序
    arraySum=sum(sortArray)
    tmpSum=0
    num=0
    for i in sortArray:
        tmpSum+=i
        num+=1
        if tmpSum>=arraySum*percentage:
            return num

patch_side = 10
num_patches = 100
# Step 0: Prepare data
# Step 0a: Load data
# Step 0b: Zero mean the data
dataset =  loadDataset(num_patches,patch_side)
visualize(dataset,int(math.sqrt(num_patches)),patch_side)
# Step 1: Implement PCA
# Step 1a: Implement PCA
covMat = numpy.cov(dataset,rowvar=1)
eigVals,eigVects=numpy.linalg.eig(numpy.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
eigValIndice=numpy.argsort(eigVals)            #对特征值从小到大排序
eigValIndice = eigValIndice[::-1]
eigVects=eigVects[:,eigValIndice]        #所有的特征向量重新按照大小排布
Xrot = numpy.dot(eigVects.T,dataset)
# Step 1b: Check covariance
covar = numpy.cov(Xrot,rowvar=1)
#print(covar)#没有找到画图函数，通过看数据可以知道是对角矩阵
# Step 2: Find number of components to retain
k = percentage2n(eigVals,0.9)
eigValIndice=numpy.argsort(eigVals)            #对特征值从小到大排序
k_eigValIndice=eigValIndice[-1:-(k+1):-1]   #最大的n个特征值的下标
# Step 3: PCA with dimension reduction
U = numpy.zeros((patch_side*patch_side,patch_side*patch_side))
U[:,k_eigValIndice]=eigVects[:,k_eigValIndice] #最大的n个特征值对应的特征向量,一列为一个特征向量
Xrot = numpy.dot(U.T,dataset)
Xreco = numpy.dot(U,Xrot)
visualize(Xreco,int(math.sqrt(num_patches)),patch_side)
# Step 4: PCA with whitening and regularization
# Step 4a: Implement PCA with whitening and regularization
Xrot = numpy.dot(eigVects.T,dataset)
print(Xrot)
epsilon = 0.1
eigVals = eigVals + epsilon
eigVals = eigVals.reshape(len(eigVals),1)
Xpcawhite = Xrot/numpy.sqrt(eigVals)
visualize(Xpcawhite,int(math.sqrt(num_patches)),patch_side)
# Step 5: ZCA whitening
Xzcawhite = numpy.dot(eigVects,Xpcawhite)
visualize(Xzcawhite,int(math.sqrt(num_patches)),patch_side)
