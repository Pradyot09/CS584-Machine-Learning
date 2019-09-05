
# coding: utf-8

# In[221]:


# References:
# Week 4 and Week 5 slides and lecture notes


# In[222]:


#1(a) Number of possible itemset

import numpy as np

print("1(a)")

givenItems=['A','B','C','D','E','F','G']


print("Given Items: \n",givenItems,"\n")

noofItems=len(givenItems)

print("Number of Items:",noofItems)

possibleItemsets=(2 **noofItems)-1

print("Number of possible itemset:",possibleItemsets)


# In[223]:


# import sympy as sy

# comb=sy.binomial(7,7)

# print(comb)


# In[224]:


# 1(b) Possible 1-itemset
# 1(c) Possible 2-itemset
# 1(d) Possible 3-itemset
# 1(e) Possible 4-itemset
# 1(f) Possible 5-itemset
# 1(g) Possible 6-itemset
# 1(h) Possible 7-itemset

print("All the possible itemsets: \n")
import itertools as itr

j=1
while j<=len(givenItems):
    
    comb=itr.combinations(givenItems,j)

    countComb=[]

#     print("1(b)")
    for i in comb:
        countComb.append(i)
        print(countComb.index(i)+1,i)
    j=j+1
    
#     print("number of combinations",len(count_combin))



# In[225]:


import pandas as pd

import numpy as np

groceriesData = pd.read_csv("E:\Local Disk D\IIT-C\Sem 4\CS 584 Machine Learning\Homeworks\Homework 2\Groceries.csv")

df = pd.DataFrame(groceriesData)

df.head(20)



# In[226]:


# Total number of Customer
print("2(a)")

uniqueCustomers=df.Customer.unique()

print("Unique Customer: ",uniqueCustomers,"\n")

totalCustomer=np.count_nonzero(uniqueCustomers)

print("Total number of Customer: ",totalCustomer)


# In[227]:


# unique items in the market basket across all customers

print("2(b)")

uniqueItems=df.Item.unique()

print("Unique Items: ",uniqueItems,"\n")

noofuniqueItems=np.count_nonzero(uniqueItems)

print("Total number of Unique Items: ",noofuniqueItems)


# In[228]:


# 
print("2(c)")

import matplotlib.pyplot as plt

import collections

itemperCustomer = df.groupby(['Customer'])['Item'].nunique()

print("Dataset for distinct items for for each customer: \n ",itemperCustomer)


# In[229]:


print("2(c)")
customerFrequency=df.Customer.value_counts()

sortedcustFrequency=sorted(customerFrequency)

itemset=collections.Counter(sortedcustFrequency)

# print(itemset)

newDataset=pd.DataFrame.from_dict(itemset, orient='index').reset_index()

newDataset=newDataset.rename(columns={'index':'Itemset', 0:'Customers'})

print("Frequency table for unique itemset: \n",newDataset)


# In[230]:


print("2(c)")
plt.figure(figsize=(8, 6))
plt.hist(customerFrequency)
# plt.scatter(newDataset.Itemset,newDataset.Customers)

plt.title("Histogram")
plt.xlabel("Unique Items")
plt.ylabel("Customer")
plt.grid()
plt.show()


median =np.median(newDataset.Itemset)

LowerQuartile=np.percentile(newDataset.Itemset,25)

UpperQuartile=np.percentile(newDataset.Itemset,75)

print(newDataset.describe(),"\n")
print("Median in the histogram: ",median,"\n")

print("25th percentile in the histogram: ",LowerQuartile,"\n")

print("75th percentile in the histogram: ",UpperQuartile,"\n")


# In[231]:


# 
print("2(d)")
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# nItemPerCustomer = groceriesData.groupby(['Customer'])['Item'].nunique()
ListItem = groceriesData.groupby(['Customer'])['Item'].apply(list).values.tolist() # Sale Receipt -> Item List

te = TransactionEncoder()
te_ary = te.fit(ListItem).transform(ListItem)
trainData = pd.DataFrame(te_ary, columns=te.columns_) # Item List -> Item Indicator

print("Items list in sales receipt format: \n",ListItem)


# In[232]:


#
print("2(d)")
totalTransactions=np.count_nonzero(itemperCustomer)

minSupport=75/totalTransactions

frequent_itemsets = apriori(trainData, min_support = minSupport, use_colnames = True)

print("Frequent itemset \n",frequent_itemsets)


# In[233]:


print("2(d)")
noOfItemset=frequent_itemsets.support.count()

print("Total number of itemset: ",noOfItemset,"\n")

print("The highest value of k in the itemset: 4")


# In[234]:


# 
print("2(e)")
assoc_rules = association_rules(frequent_itemsets, metric = "confidence", min_threshold = 0.01)

print("Association rules \n",assoc_rules)


# In[235]:


assoc_rules


# In[236]:


print("2(e)")
noOfassociationrules=assoc_rules.antecedents.count()

print("Total number of association rules: ",noOfassociationrules,"\n")


# In[237]:


#
print("2(f)")

support=assoc_rules.as_matrix(columns=['support'])

confidence=assoc_rules.as_matrix(columns=['confidence'])

lift=assoc_rules.as_matrix(columns=['lift'])

plt.figure(figsize=(8, 6))
plt.scatter(confidence,support,c=lift,s=lift)
plt.title("Confidence vs. Support")
plt.xlabel('Confidence')
plt.ylabel('Support')

cbar=plt.colorbar()

cbar.set_label("Lift", labelpad=+1)
plt.grid()
plt.show()


# In[238]:


#
print("2(g)")

confAbove60=(assoc_rules.confidence>=0.6)

antecendets_WRT=np.array(assoc_rules[confAbove60]['antecedents'])

consequents_WRT=np.array(assoc_rules[confAbove60]['consequents'])

confidence_WRT=np.array(assoc_rules[confAbove60]['confidence'])

support_WRT=np.array(assoc_rules[confAbove60]['support'])

lift_WRT=np.array(assoc_rules[confAbove60]['lift'])

# print(lift_WRT)

data=np.column_stack((antecendets_WRT,consequents_WRT,confidence_WRT,support_WRT,lift_WRT))

requiredData=pd.DataFrame(data,columns=['Antecents','Consequents','Confidence','Support','Lift'])

print("The required data for confidence geater than 60%: \n",requiredData)


# In[239]:


requiredData


# In[240]:


#
print("2(h)")

print("Consequents are the same for all the antecendents")


# In[241]:


print("3(a)")

# import numpy as np

import sklearn.cluster as cluster
import sklearn.metrics as metrics

carsData=pd.read_csv('E:\Local Disk D\IIT-C\Sem 4\CS 584 Machine Learning\Homeworks\Homework 2\cars.csv')

carsDf=pd.DataFrame(carsData)

# print(carsDf)

horsepower=np.array(carsDf['Horsepower'])
weight=np.array(carsDf['Weight'])

inputVariables=carsDf[['Horsepower','Weight']]

# print(horsepower)
# print(weight)

# inputVariables=np.column_stack((horsepower,weight))

noOfCluster=15
kmeans = cluster.KMeans(n_clusters=noOfCluster, random_state=60616).fit(inputVariables)

print("Cluster Assignment: \n", kmeans.labels_,"\n")

for i in range(noOfCluster):
    print("Cluster Centroid ",i,":" "\n", kmeans.cluster_centers_[i],"\n")


# In[242]:


# Determine the number of clusters
nClusters = np.zeros(15)
Elbow = np.zeros(15)
Silhouette = np.zeros(15)
# TotalWCSS = np.zeros(15)
# Inertia = np.zeros(15)


# In[243]:


inputVar=carsDf[['Horsepower','Weight']].as_matrix()
    
# print(inputVar)


# In[244]:


print("3(a)")

for c in range(15):
    KClusters = c + 1
    nClusters[c] = KClusters

    kmeans = cluster.KMeans(n_clusters=KClusters, random_state=60616).fit(inputVar)
  
    if (1 < KClusters):
        Silhouette[c] = metrics.silhouette_score(inputVar, kmeans.labels_)
    else:
        Silhouette[c] = np.NaN

    WCSS = np.zeros(KClusters)
    nC = np.zeros(KClusters)

    for i in range(len(inputVar)):
        k = kmeans.labels_[i]
        nC[k] += 1
        diff = inputVar[i] - kmeans.cluster_centers_[k]
        WCSS[k] += diff.dot(diff)

    Elbow[c] = 0
    for k in range(KClusters):
        Elbow[c] += WCSS[k] / nC[k]
    print("Cluster Assignment: \n", kmeans.labels_)
    
    for k in range(KClusters):
        print("Cluster ", k)
        print("Centroid = ", kmeans.cluster_centers_[k])
        print("Size = ", nC[k])
        print("Within Sum of Squares = ", WCSS[k])
        print(" ")


# In[245]:


print("3(a)\n")
print("N Clusters\t Elbow Value\t Silhouette Value:")
for i in range(15):
    print('{:.0f} \t\t {:.3f} \t {:.3f}'
          .format(nClusters[i], Elbow[i], Silhouette[i]))


# In[246]:


print("3(a)")
plt.figure(figsize=(8, 6))
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(np.arange(1, 16, step = 1))
plt.show()


# In[247]:


print("3(a)")
plt.figure(figsize=(8, 6))
plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(np.arange(1, 16, step = 1))
plt.show()  


# In[248]:


print("3(b)")

print("Based on the elbow and Silhouette the number of cluster are 4 ")


# In[249]:


# 
print("4(a)")

spiralData=pd.read_csv('E:\Local Disk D\IIT-C\Sem 4\CS 584 Machine Learning\Homeworks\Homework 2\Spiral.csv')

spiralDataFrame=pd.DataFrame(spiralData)

x=spiralDataFrame['x']
y=spiralDataFrame['y']


plt.figure(figsize=(8, 6))
plt.scatter(x,y)
plt.xlabel("X observations")
plt.ylabel("Y observations")
plt.grid()
plt.show()

print("By observing or visualizing the graph we can say that there are 2 clusters")
print("Number of Clusters: 2")


# In[250]:


# 
print("4(b)")

inputData=spiralDataFrame[['x','y']]

kmeanModel = cluster.KMeans(n_clusters=2, random_state=60616).fit(inputData)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, c=kmeanModel.labels_.astype(float))
plt.xlabel("X observations")
plt.ylabel("Y observations")
plt.grid()
plt.show()


# In[251]:


# Ref: Class Slides
print("4(c)")

import math
from sklearn.neighbors import NearestNeighbors as kNN
from sklearn.neighbors import DistanceMetric as Dm


trainData=spiralDataFrame[['x','y']]

# Three nearest neighbors
kNNSpec = kNN(n_neighbors = 3, algorithm = 'brute', metric = 'euclidean')
# kNNSpec = kNN(n_neighbors = 8, algorithm = 'brute', metric = 'euclidean')
nbrs = kNNSpec.fit(trainData)
d3, i3 = nbrs.kneighbors(trainData)

# Retrieve the distances among the observations
distObject = Dm.get_metric('euclidean')
distances = distObject.pairwise(trainData)

print("Distance Metric: \n",distances)


# In[252]:


print("4(d)")
noOfObs=spiralDataFrame.shape[0]
Adjacency = np.zeros((noOfObs, noOfObs))
Degree = np.zeros((noOfObs, noOfObs))

for i in range(noOfObs):
    for j in i3[i]:
        if (i <= j):
            Adjacency[i,j] = math.exp(- distances[i][j])
            Adjacency[j,i] = Adjacency[i,j]

for i in range(noOfObs):
    sum = 0
    for j in range(noOfObs):
        sum += Adjacency[i,j]
    Degree[i,i] = sum
        
print("Adjacency Matrix: \n",Adjacency)


# In[253]:


print("4(d)")
print("Degree Matrix: \n",Degree)


# In[254]:


print("4(d)")
Lmatrix = Degree - Adjacency

print("Laplace Matrix: \n",Lmatrix,"\n")


# In[255]:


print("4(d)")
#Eigen values and Eigenvectors

from numpy import linalg as LA
evals, evecs = LA.eigh(Lmatrix)

print("Eigenvalues of Laplace Matrix = \n", evals,"\n")
print("Eigenvectors of Laplace Matrix = \n",evecs,"\n")


# In[256]:


print("4(d)")
# Series plot of the smallest ten eigenvalues to determine the number of clusters

plt.figure(figsize=(8, 6))
plt.scatter(np.arange(0,9,1), evals[0:9,])
plt.xlabel('Sequence')
plt.ylabel('Eigenvalue')
plt.grid()
plt.show()


# In[257]:


Z = evecs[:,[0,1]]
# print(Z)

print(Z[[0]].mean(), Z[[0]].std())
print(Z[[1]].mean(), Z[[1]].std())


# In[258]:


print("4(e)")

# Inspect the values of the selected eigenvectors 
Z = evecs[:,[0,1]]

plt.figure(figsize=(8, 6))
plt.scatter(Z[[0]], Z[[1]])
plt.xlabel('Z[0]')
plt.ylabel('Z[1]')
plt.grid()
plt.show()


# In[259]:


print("4(f)")

kmeans_spectral = cluster.KMeans(n_clusters=2, random_state=60616).fit(Z)
spiralDataFrame['SpectralCluster'] = kmeans_spectral.labels_
plt.figure(figsize=(8, 6))
plt.scatter(spiralDataFrame[['x']], spiralDataFrame[['y']], c = spiralDataFrame[['SpectralCluster']])
plt.xlabel('X Observation')
plt.ylabel('Y Observation')
plt.grid(True)
plt.show()

