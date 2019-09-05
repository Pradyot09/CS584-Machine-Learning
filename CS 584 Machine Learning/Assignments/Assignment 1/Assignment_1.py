
# coding: utf-8

# In[248]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


data = pd.read_csv("E:\\Local Disk D\\IIT-C\Sem 4\\CS 584 Machine Learning\\Homeworks\\Homework 1\\NormalSample.csv")

df = pd.DataFrame(data)

noOfobservations=df['x']

N=df.x.count()

# print(df)
print(N)


# In[249]:


data.describe()


# In[250]:


# Visualization of the variable x 
plt.hist(noOfobservations)
plt.title("Histogram")
plt.xlabel("Values of X")
plt.ylabel("Frequency")
plt.show()


# In[251]:


#Binwidth According to Izenman (1991) method
median =np.median(noOfobservations)

LowerQuartile=np.percentile(noOfobservations,25)

UpperQuartile=np.percentile(noOfobservations,75)

InterQuartile=UpperQuartile-LowerQuartile

IQR=InterQuartile

binwidth=2*(𝐼𝑄𝑅)*N ** (-1. / 3)

# print("Median ",median)

# print("LowerQuartile ",LowerQuartile)

# print("UpperQuartile ",UpperQuartile)

# print("InterQuartile ",IQR)

print("1(a) Binwidth:", format(binwidth))


# In[252]:


#Min and Max Value of field X

MaxValue=max(noOfobservations)
MinValue=min(noOfobservations)

print("1(b)")
print("MaxValue ",MaxValue)

print("MinValue ",MinValue)


# In[253]:


#Next Maximum and Minimum integer value for X

a=math.floor(MinValue)
b=math.ceil(MaxValue)

print("1(c)")
print("NextMin ",a)

print("NextMax ",b)


# In[254]:


#h = 0.1, minimum = a and maximum = b

# plt.hist(noOfobservations)

print("1(d)")
binwidth_h=0.1

plt.hist(noOfobservations, bins=np.arange(a, b + binwidth_h, binwidth_h))

# plt.hist(noOfobservations, bins=((b-a)/binwidth_h))
plt.title("Histogram")
plt.xlabel("Values of X")
plt.ylabel("Frequency")
plt.show()


# In[255]:


#h = 0.5, minimum = a and maximum = b

# plt.hist(noOfobservations)

print("1(e)")
binwidth_h=0.5

plt.hist(noOfobservations, bins=np.arange(a, b + binwidth_h, binwidth_h))

# plt.hist(noOfobservations, bins=np.arange(a, b, binwidth_h))
plt.title("Histogram")
plt.xlabel("Values of X")
plt.ylabel("Frequency")
plt.show()


# In[256]:


#h = 1, minimum = a and maximum = b

# plt.hist(noOfobservations)

print("1(f)")
binwidth_h=1

plt.hist(noOfobservations, bins=np.arange(a, b + binwidth_h, binwidth_h))
plt.title("Histogram")
plt.xlabel("Values of X")
plt.ylabel("Frequency")
plt.show()


# In[257]:


#h = 2, minimum = a and maximum = b

# plt.hist(noOfobservations)

print("1(g)")
binwidth_h=2

plt.hist(noOfobservations, bins=np.arange(a, b + binwidth_h, binwidth_h))
plt.title("Histogram")
plt.xlabel("Values of X")
plt.ylabel("Frequency")
# plt.grid(axis="x")
# plt.grid(axis="y")
plt.show()


# In[258]:


#Five number summary of the box plot

Median=np.median(noOfobservations)

LowerQuartile=np.percentile(noOfobservations,25)

Q1=LowerQuartile

UpperQuartile=np.percentile(noOfobservations,75)

Q3=UpperQuartile

MaxValue=max(noOfobservations)

MinValue=min(noOfobservations)

InterQuartile=UpperQuartile-LowerQuartile

IQR=InterQuartile


print("2(a)")
print("MinValue ",MinValue)

print("LowerQuartile ",LowerQuartile)

print("Median ",Median)

print("UpperQuartile ",UpperQuartile)

print("MaxValue ",MaxValue)


# In[259]:


# Values of the 1.5 IQR whiskers

Lowerwhisker=Q1-1.5*IQR

Upperwhisker=Q3+1.5*IQR

print("2(a)")

print("InterQuartile ",IQR)
print("Lowerwhisker ",Lowerwhisker)

print("Upperwhisker ",Upperwhisker)



# In[260]:


#Five-number summary of x for category one of the group

isOne=(df.group==1)

# groupOne = data[isOne]['x']

groupOne = df[isOne]['x']
# print(One.head(20))

MinValueOne=min(groupOne)

LowerQuartileOne=np.percentile(groupOne,25)

MedianOne=np.median(groupOne)

UpperQuartileOne=np.percentile(groupOne,75)

MaxValueOne=max(groupOne)

InterQuartileOne=UpperQuartileOne-LowerQuartileOne

LowerwhiskerOne = LowerQuartileOne-1.5*InterQuartileOne

UpperwhiskerOne = UpperQuartileOne+1.5*InterQuartileOne

print("2(b)")
print("Min Value of group One ",MinValueOne)

print("Lower Quartile of group One ",LowerQuartileOne)

print("Median of group One ",Medianone)

print("Upper Quartile of group One ",UpperQuartileOne)

print("Max Value of group One ",MaxValueOne)

print("Lower whisker of group One ",LowerwhiskerOne)

print("Upper whisker of group One ",UpperwhiskerOne)




# In[261]:


groupOne.describe()


# In[262]:


#Five-number summary of x for category Zero of the group

isZero=(df.group!=1)

groupZero = data[isZero]['x']

MinValueZero=min(groupZero)

LowerQuartileZero=np.percentile(groupZero,25)

MedianZero=np.median(groupZero)

UpperQuartileZero=np.percentile(groupZero,75)

MaxValueZero=max(groupZero)

InterQuartileZero=UpperQuartileZero-LowerQuartileZero

LowerwhiskerZero = LowerQuartileZero-1.5*InterQuartileZero

UpperwhiskerZero = UpperQuartileZero+1.5*InterQuartileZero


print("2(b)")
print("Min Value of group Zero ",MinValueZero)

print("Lower Quartile of group Zero ",LowerQuartileZero)

print("Median of group Zero",Medianzero)

print("Upper Quartile of group Zero ",UpperQuartileZero)

print("Max Value of group Zero ",MaxValueZero)

print("Lower whisker of group Zero ",LowerwhiskerZero)

print("Upper whisker of group Zero ",UpperwhiskerZero)


# In[263]:


groupZero.describe()


# In[264]:


# Visualization values of x using the boxplot 

print("2(c)")
plt.boxplot(noOfobservations,vert=False)
plt.title("Boxplot for Values of X")
plt.xlabel("Values of X")
plt.grid(axis="x")
plt.show()


# In[265]:


#Five number summary of x for each category of the group

isOne=(df.group==1)

isZero=(df.group!=1)

One = data[isOne]['x']
Zero = data[isZero]['x']

print("2(d)")
fig = plt.figure()
ax = fig.add_subplot(111)
# ax.boxplot(noOfobservations,vert=False)
ax.boxplot([noOfobservations,Zero,One],labels=['All x','0', '1'],vert=False)
# ax.boxplot([One,Zero], labels=['1', '0'],vert=False)

plt.xlabel("Values of X")
plt.ylabel("Values of groups")
plt.show()



# In[296]:


#Outliers for the entire data

# Lowerwhisker  27.4
# Upperwhisker  35.4

print("2(d)")
outliersBelowLowerwhisker=noOfobservations[noOfobservations<Lowerwhisker]

print(outliersBelowLowerwhisker)

outliersAboveUpperwhisker=noOfobservations[noOfobservations>Upperwhisker]

print(outliersAboveUpperwhisker)


# In[268]:


#Outliers for the group one

#Lower whisker of group One  29.449999999999992
#Upper whisker of group One  34.650000000000006

print("2(d)")
outliersLowerwhiskerone=groupOne[groupOne<LowerwhiskerOne]

print("Outliers of Lower Whisker for group one \n",outliersLowerwhiskerone)

outliersUpperwhiskerone=groupOne[groupOne>UpperwhiskerOne]

print("Outliers of Upper Whisker for group one \n",outliersUpperwhiskerone)


# In[269]:


#Outliers for the group Zero

#Lower whisker of group Zero  27.599999999999994
#Upper whisker of group Zero  32.400000000000006

print("2(d)")
outliersLowerwhiskerzero=groupZero[groupZero<LowerwhiskerZero]

print("Outliers of Lower Whisker for group zero \n",outliersLowerwhiskerzero)

outliersUpperwhiskerzero=groupZero[groupZero>UpperwhiskerZero]

print("Outliers of Upper Whisker for group zero \n",outliersUpperwhiskerzero)


# In[270]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math


fraudData = pd.read_csv("E:\\Local Disk D\\IIT-C\\Sem 4\\CS 584 Machine Learning\\Homeworks\\Homework 1\\Fraud.csv")

df = pd.DataFrame(fraudData)


# print(df.head(10))


# In[271]:


df.head()


# In[272]:


#Percent of the fradulant data 

totalData=df.FRAUD.count()

FraudData=(df.FRAUD == 1).sum()

# fraudData=df[df['FRAUD'] == 1].count()

percentFraud=(FraudData/totalData)*100

print("3(a)")

print("Percentage of fradulant data ",round(percentFraud,4))


# In[273]:


# Visualization of total spend using boxplot  

print("3(b)")

isFradulantData=(df.FRAUD==1)

isOtherwiseData=(df.FRAUD!=1)

FradulantData = fraudData[isFradulantData]

OtherwiseData=fraudData[isOtherwiseData]

FradulantData = fraudData[isFradulantData]['TOTAL_SPEND']
OtherwiseData = fraudData[isOtherwiseData]['TOTAL_SPEND']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([OtherwiseData,FradulantData], labels=['0', '1'],vert=False)

plt.ylabel("Values of Fraud")
plt.xlabel("Total Amount of Claims")
plt.show()



# In[274]:


# Visualization of doctor visits using boxplot  

print("3(b)")
isFradulantData=(df.FRAUD==1)

isOtherwiseData=(df.FRAUD!=1)

FradulantData = fraudData[isFradulantData]
OtherwiseData=fraudData[isOtherwiseData]
FradulantData = fraudData[isFradulantData]['DOCTOR_VISITS']
OtherwiseData = fraudData[isOtherwiseData]['DOCTOR_VISITS']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([OtherwiseData,FradulantData], labels=['0', '1'],vert=False)
plt.ylabel("Values of Fraud")
plt.xlabel("Doctor Visits")
plt.show()


# In[275]:


# Visualization of number of claims using boxplot  

print("3(b)")
isFradulantData=(df.FRAUD==1)

isOtherwiseData=(df.FRAUD!=1)

FradulantData = fraudData[isFradulantData]

OtherwiseData=fraudData[isOtherwiseData]

FradulantData = fraudData[isFradulantData]['NUM_CLAIMS']
OtherwiseData = fraudData[isOtherwiseData]['NUM_CLAIMS']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([OtherwiseData,FradulantData], labels=['0', '1'],vert=False)

plt.ylabel("Values of Fraud")
plt.xlabel("Number of Claims Made")
plt.show()


# In[276]:


# Visualization of membership duration using boxplot  

print("3(b)")
isFradulantData=(df.FRAUD==1)

isOtherwiseData=(df.FRAUD!=1)

FradulantData = fraudData[isFradulantData]
OtherwiseData=fraudData[isOtherwiseData]

FradulantData = fraudData[isFradulantData]['MEMBER_DURATION']
OtherwiseData = fraudData[isOtherwiseData]['MEMBER_DURATION']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([OtherwiseData,FradulantData], labels=['0', '1'],vert=False)
plt.ylabel("Values of Fraud")
plt.xlabel("Membership Duration")
plt.show()


# In[277]:


# Visualization of optical examination using boxplot  

print("3(b)")

isFradulantData=(df.FRAUD==1)

isOtherwiseData=(df.FRAUD!=1)

FradulantData = fraudData[isFradulantData]
OtherwiseData=fraudData[isOtherwiseData]

FradulantData = fraudData[isFradulantData]['OPTOM_PRESC']
OtherwiseData = fraudData[isOtherwiseData]['OPTOM_PRESC']
fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([OtherwiseData,FradulantData], labels=['0', '1'],vert=False)
plt.ylabel("Values of Fraud")
plt.xlabel("Number of Optical Examination")
plt.show()


# In[278]:


# Visualization of number of members using boxplot  

print("3(b)")

isFradulantData=(df.FRAUD==1)

isOtherwiseData=(df.FRAUD!=1)

FradulantData = fraudData[isFradulantData]
OtherwiseData=fraudData[isOtherwiseData]

FradulantData = fraudData[isFradulantData]['NUM_MEMBERS']
OtherwiseData = fraudData[isOtherwiseData]['NUM_MEMBERS']

fig = plt.figure()
ax = fig.add_subplot(111)
ax.boxplot([OtherwiseData,FradulantData], labels=['0', '1'],vert=False)
plt.ylabel("Values of Fraud")
plt.xlabel("Number of Members Covered ")
plt.show()


# In[279]:


#Orthonormalize interval variables 
df.head()


# In[280]:


import scipy as sp

from scipy import linalg as la
from numpy import linalg as la2


intervalMatrix=np.array(fraudData.iloc[:,2:8].values)


orthonormalize=la.orth(intervalMatrix)

print("The orthonormalize matrix = \n", orthonormalize)

Varifiy = orthonormalize.transpose().dot(orthonormalize)
print("Identity Matrix = \n", Varifiy)
print (intervalMatrix.ndim)


# In[281]:


#Orthonormalizing interval variables

import scipy as sp

from scipy import linalg as la

from numpy import linalg as la2

intervalMatrix=np.matrix(fraudData.iloc[:,2:8].values)

print (intervalMatrix)

#Creating Transpose Matrix
transposeMatrix=intervalMatrix.transpose()*intervalMatrix

print("Multiplication of Transpose Matrix and original Matrix \n\n",transposeMatrix)


# In[282]:


#Eigen values and Eigenvectors
evals, evecs = la2.eigh(transposeMatrix)
print("3(c)(i)")
print("Eigenvalues of transposeMatrix = \n\n", evals)
print("Eigenvectors of transposeMatrix = \n\n",evecs)


# In[283]:


#Transformation matrix

print("3(c)(ii)")
transformationMatrix = evecs * la2.inv(np.sqrt(np.diagflat(evals)))
print("Transformation Matrix = \n\n", transformationMatrix)


# In[284]:


transf_im.shape


# In[285]:


intervalMatrix.shape


# In[286]:


transformationMatrix.shape


# In[287]:


#Transformation of  intervalMatrix

transf_im=intervalMatrix*transformationMatrix
print("The Transformed Interval Matrix = \n\n", transf_im)


# In[288]:


# Identity Matrix to prove the the matrix is orthonormalization

print("3(c)(ii)")
xtx = transf_im.transpose()*transf_im

# print(np.shape(xtx))

print("Expect an Identity Matrix = \n\n", xtx)


# In[289]:


# Nearest Neighbors module 

from sklearn.neighbors import KNeighborsClassifier 

#Transform data as traindata

trainData = transf_im


targetData = df['FRAUD']


KNeighbor = KNeighborsClassifier(n_neighbors=5 , algorithm = 'brute', metric = 'euclidean')
nbrs = KNeighbor.fit(trainData, targetData)

print(nbrs)


# In[290]:


score=nbrs.score(trainData,targetData)

print("3(d)(i)")
print(score)


# In[291]:


# Observation of input variables

inputVariables = pd.DataFrame(columns=["TOTAL_SPEND", "DOCTOR_VISITS", "NUM_CLAIMS", "MEMBER_DURATION", "OPTOM_PRESC", "OPTOM_PRESC"], 
                         data=[[7500,15,3,127,2,2]])



print("3(e)")
inputMatrix=np.matrix(inputVariables)

print(inputMatrix)

transInputMatrix = inputMatrix * transformationMatrix;

print(transInputMatrix)


myNeighbors = nbrs.kneighbors(transInputMatrix, return_distance = False)
print("Nearest Neighbors = \n\n", myNeighbors)


# In[292]:


#Values of all the target values
#Since the index starts from 0 so we subtract 1 from each neighbour
print("3(e)")
targetData[[588-1, 2897-1 ,1199-1, 1246-1 , 886-1]]


# In[293]:


#Values of all the input 
#Since the index starts from 0 so we subtract 1 from each neighbour
print("3(e)")
print(intervalMatrix[588-1])
print(intervalMatrix[2897-1])
print(intervalMatrix[1199-1])
print(intervalMatrix[1246-1])
print(intervalMatrix[886-1])


# In[294]:


nbrs.predict(transInputMatrix)

prediction=nbrs.predict(transInputMatrix)

print("3(f)")
print(prediction)


# In[295]:


class_proba=nbrs.predict_proba(inputMatrix)
print("3(f)")
print(class_proba)

