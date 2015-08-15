import numpy as np
import matplotlib.pyplot as plt
import math

#数据要求：每列代表一个数据集，每行代表一个维度
def zeroMean(dataMat):
    meanVal=np.mean(dataMat,axis=1)     #按行求均值，即求各个特征的均值
    meanVal = np.reshape(meanVal,(2, 1))
    newData=dataMat-meanVal
    return newData,meanVal

# Step 0: Load data
dataMat = np.loadtxt('pcaData.txt')#读取文件中的数据
plt.scatter(dataMat[0],dataMat[1])
plt.show()
# Step 1a: Finding the PCA basis
newData,meanVal=zeroMean(dataMat)
#print(meanVal)
covMat = np.cov(newData,rowvar=1)
eigVals,eigVects=np.linalg.eig(np.mat(covMat))#求特征值和特征向量,特征向量是按列放的，即一列代表一个特征向量
print(eigVals,eigVects)
plt.plot([0,eigVects[0,0]],[0,eigVects[1,0]],color="blue", linewidth=1.0, linestyle="-")
plt.plot([0,eigVects[0,1]],[0,eigVects[1,1]],color="blue", linewidth=1.0, linestyle="-")
plt.scatter(dataMat[0],dataMat[1])
plt.show()
# Step 1b: Check xRot
eigValIndice=np.argsort(eigVals)            #对特征值从小到大排序
eigValIndice = eigValIndice[::-1]
eigVects=eigVects[:,eigValIndice]#所有的特征向量重新按照大小排布
Xrot = np.dot(eigVects.T,newData)
plt.scatter(Xrot[0],Xrot[1])
plt.show()
# Step 2: Dimension reduce and replot
U = eigVects + 0 #不能够直接赋值，因为直接赋值之后地址的值也会变化
U[:,1] = [0]
Xhat = np.dot(U,Xrot)
plt.scatter(Xhat[0],Xhat[1])
plt.show()
# Step 3: PCA Whitening
epsilon = 0
eigVals = eigVals + epsilon
print(eigValIndice)
eigVals = eigVals.reshape(len(eigVals),1)
eigVals = eigVals[eigValIndice,:]#所有的特征值重新按照大小排布
Xpcawhite = Xrot/np.sqrt(eigVals)
plt.scatter(Xpcawhite[0],Xpcawhite[1])
plt.show()
# Step 4: ZCA Whitening
Xzcawhite = np.dot(eigVects,Xpcawhite)
plt.scatter(Xzcawhite[0],Xzcawhite[1])
plt.show()
