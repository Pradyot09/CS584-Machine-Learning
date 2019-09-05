import matplotlib.pyplot as plt
import numpy
import pandas
import sklearn.cluster as cluster
import sklearn.decomposition as decomposition
import sklearn.metrics as metrics
import sklearn.linear_model as linear_model

# The following data for all US Treasury Rate since 1/1/2018

ChicagoDiabetes = pandas.read_csv('D:\\IIT\\Fall18\\Machine Learning\\Assignment4\\ChicagoDiabetes.csv',
                          delimiter=',')


# Feature variables
X = ChicagoDiabetes[['Crude Rate 2000','Crude Rate 2001','Crude Rate 2002','Crude Rate 2003',
            'Crude Rate 2004','Crude Rate 2005','Crude Rate 2006','Crude Rate 2007',
           'Crude Rate 2008','Crude Rate 2009','Crude Rate 2010', 'Crude Rate 2011']]

nObs = X.shape[0]
nVar = X.shape[1]

#part a
print("Number of observations used ={0}\nNumber of variables used={1}".format(nObs,nVar))

X.columns=['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011']

#part b 
pandas.plotting.scatter_matrix(X, figsize=(20,20), c = 'red',
                               diagonal='hist', hist_kwds={'color':['blue']})
plt.suptitle("Crude Rate Scatter Matrix")


# Calculate the Correlations among the variables
XCorrelation = X.corr(method = 'pearson', min_periods = 1)

print('Empirical Correlation: \n', XCorrelation)

# Extract the Principal Components
_thisPCA = decomposition.PCA(n_components = nVar)
_thisPCA.fit(X)

cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)

print('Explained Variance: \n', _thisPCA.explained_variance_)
print('Explained Variance Ratio: \n', _thisPCA.explained_variance_ratio_)
print('Cumulative Explained Variance Ratio: \n', cumsum_variance_ratio)
print('Principal Components: \n', _thisPCA.components_)

#part c Explained Variance and ratio
plt.plot(_thisPCA.explained_variance_ratio_, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.axhline((1/nVar), color = 'r', linestyle = '--')
plt.grid(True)
plt.show()

#part d cusmum_variance_ratio
cumsum_variance_ratio = numpy.cumsum(_thisPCA.explained_variance_ratio_)
plt.plot(cumsum_variance_ratio, marker = 'o')
plt.xlabel('Index')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.xticks(numpy.arange(0,nVar))
plt.grid(True)
plt.show()

#part e 

print("{0:.2f}%  of the total variance is explained by the first two principal components"
      .format(cumsum_variance_ratio[1]*100))


first2PC = _thisPCA.components_[:, [0,1]]
print('Principal COmponent: \n', first2PC)

# Transform the data using the first two principal components
_thisPCA = decomposition.PCA(n_components = 2)
X_transformed = pandas.DataFrame(_thisPCA.fit_transform(X))

# Find clusters from the transformed data
maxNClusters = 15

nClusters = numpy.zeros(maxNClusters-1)
Elbow = numpy.zeros(maxNClusters-1)
Silhouette = numpy.zeros(maxNClusters-1)
TotalWCSS = numpy.zeros(maxNClusters-1)
Inertia = numpy.zeros(maxNClusters-1)

for c in range(maxNClusters-1):
   KClusters = c + 2
   nClusters[c] = KClusters

   kmeans = cluster.KMeans(n_clusters=KClusters, random_state=20181010).fit(X_transformed)

   # The Inertia value is the within cluster sum of squares deviation from the centroid
   Inertia[c] = kmeans.inertia_
   
   if (KClusters > 1):
       Silhouette[c] = metrics.silhouette_score(X_transformed, kmeans.labels_)
   else:
       Silhouette[c] = float('nan')

   WCSS = numpy.zeros(KClusters)
   nC = numpy.zeros(KClusters)

   for i in range(nObs):
      k = kmeans.labels_[i]
      nC[k] += 1
      diff = X_transformed.iloc[i,] - kmeans.cluster_centers_[k]
      WCSS[k] += diff.dot(diff)

   Elbow[c] = 0
   for k in range(KClusters):
      Elbow[c] += (WCSS[k] / nC[k])
      TotalWCSS[c] += WCSS[k]

   print("The", KClusters, "Cluster Solution Done")

print("N Clusters\t Inertia\t Total WCSS\t Elbow Value\t Silhouette Value:")
for c in range(maxNClusters-1):
   print('{:.0f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}'
         .format(nClusters[c], Inertia[c], TotalWCSS[c], Elbow[c], Silhouette[c]))

# Draw the Elbow and the Silhouette charts  
plt.plot(nClusters, Elbow, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Elbow Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

plt.plot(nClusters, Silhouette, linewidth = 2, marker = 'o')
plt.grid(True)
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Value")
plt.xticks(numpy.arange(2, maxNClusters, 1))
plt.show()

# Fit the 4 cluster solution'
kmeans = cluster.KMeans(n_clusters=4, random_state=20181010).fit(X_transformed)
X_transformed['Cluster ID'] = kmeans.labels_

# Draw the first two PC using cluster label as the marker color 
carray = ['red', 'orange', 'green', 'black']
plt.figure(figsize=(10,10))
for i in range(4):
    subData = X_transformed[X_transformed['Cluster ID'] == i]
    plt.scatter(x = subData[0],
                y = subData[1], c = carray[i], label = i, s = 25)
plt.grid(True)
plt.axis(aspect = 'equal')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.axis(aspect = 'equal')
plt.legend(title = 'Cluster ID', fontsize = 12, markerscale = 2)
plt.show()

#part h

clusId_mb=(X_transformed.groupby('Cluster ID').count()[0])
cid=list(clusId_mb)
cid=pandas.Series(cid)

clusterMember=pandas.DataFrame(cid,index=None)

clusterMember['Cluster']=range(0,4)
clusterMember['nCommunities']=clusterMember.iloc[:,0]

clusterMember=clusterMember.iloc[:,1:]

clusterMember

#part i


ClusterIndex= {i: numpy.where(kmeans.labels_ == i)[0] for i in range(kmeans.n_clusters)}

for k,v in ClusterIndex.items():
    names=[]
    for i in v:
        names.append(ChicagoDiabetes.iloc[i,1])
    print("Members in Cluster {} are\n {}\n".format(k,names))
    

#part j
year=[2000,2001,2002,2003,2004,2005,2006,2007,2008,2009,2010,2011]
refrt=[25.4,25.8,27.2,25.4,26.2,26.6,27.4,28.7,27.9,27.5,26.8,25.6]

for key,val in ClusterIndex.items():
    cruderate = 0
    avgrate=[]
    for j in range(2,ChicagoDiabetes.shape[1]-1,2):
        hosp=0
        pop=0
        for i in val:
            hosp=hosp+ChicagoDiabetes.iloc[i,j]
            pop=pop+(ChicagoDiabetes.iloc[i,j]/ChicagoDiabetes.iloc[i,j+1]*10000)
            
        cruderate=hosp/pop*10000
        avgrate.append(cruderate)
    plt.plot(year,avgrate)
    #plt.legend(key)
    
plt.plot(year,refrt,color='black')
plt.text(2004, 29, 'Ref')
plt.text(2004, 55, 'Cluster 3')
plt.text(2004, 43, 'Cluster 1')
plt.text(2004, 24, 'Cluster 2')
plt.text(2004, 15, 'Cluster 0')
plt.xlabel('Year')
plt.ylabel('Crude Hospitalization Rates (per 10000)')   
plt.axis(aspect = 'equal')
plt.grid(True)
plt.title('Crude Rates vs Year Plot')
plt.show()


####part 2

import numpy
import pandas as pd


data = pd.read_csv('Purchase_Likelihood.csv')
nTotal = len(data)

all=pd.crosstab(columns=data.A,index=0,normalize='index')

print('\nClass Probabilities:\n',all)



GrpSize=pd.crosstab(index=data.A,columns=data.group_size,normalize='index')

homeOwner=pd.crosstab(index=data.A,columns=data.homeowner,normalize='index')

marriedCouple=pd.crosstab(index=data.A,columns=data.married_couple,normalize='index')




def empircal_nb(class_prob, grp_size, homeown, married_coup):
    c0 = class_prob[0][0] * grp_size[0] * homeown[0] * married_coup[0]
    c1 = class_prob[1][0] * grp_size[1] * homeown[1] * married_coup[1]
    c2 = class_prob[2][0] * grp_size[2] * homeown[2] * married_coup[2]
    total = c0+c1+c2
    
    p0 = c0/total
    p1 = c1/total
    p2 = c2/total
    
    return [p0,p1,p2]



Grp=[1,2,3,4]
homeOwn=[0,1]
mc=[0,1]
prob_ls=[]


for g in Grp:
    for h in homeOwn:
        for m in mc:
            print('group_size=',g,' homeowner=',h,' married_couple=',m)
            prob = (empircal_nb(all,GrpSize[g],homeOwner[h],marriedCouple[m]))
            print(prob)
            prob_ls.append((g,h,m,prob))

sorted(prob_ls, key = lambda prob_ls: prob_ls[3][1], reverse = True)

prob_ls