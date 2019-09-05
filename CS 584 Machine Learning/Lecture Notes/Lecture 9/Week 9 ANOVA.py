import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats as stats

# Define a function to perform the ANOVA, return the F-value, the degrees of freedom, and the p-value
def ANOVA_Test (
   inData,          # input data frame
   grpPred,         # name of the categorical predictor
   yTarget,         # name of the interval target variable
   debug = "N"      # print debugging information
   ):

   # Extract the relevant columns
   thisData = inData[[grpPred, yTarget]]
   
   # Calculate the statistics of all observations
   nRowsTotal = thisData[yTarget].count()
   stdTotal = thisData[yTarget].std()

   # Calculate the statistics of each group
   nRowsGroup = thisData.groupby(grpPred).count()
   stdGroup = thisData.groupby(grpPred).std()

   # Determine the number of groups
   nGroup = nRowsGroup.shape[0]

   # Calculate the Total Sum of Squares
   ssTotal = (nRowsTotal - 1.0) * stdTotal * stdTotal

   # Calculate the Within Group Sum of Squares
   ssGroup = (nRowsGroup - 1.0) * stdGroup * stdGroup
   ssWithin = ssGroup.sum()

   # Calculate the Between Group Sum of Squares
   ssBetween = ssTotal - ssWithin

   # Calculate the Between and the Within Degrees of Freedom
   dfBetween = nGroup - 1.0
   dfWithin = nRowsTotal - nGroup

   # Calculate the Between and the Within Group Mean Squares
   msBetween = ssBetween / dfBetween
   msWithin = ssWithin / dfWithin

   # Calculate the F value
   fStat = msBetween / msWithin
   fPValue = stats.f.sf(fStat, dfBetween, dfWithin)

   if (debug == 'Y'):
      print('\n--------------- Group Statistics ---------------')
      print('Number of Groups =', nGroup)
      meanGroup = thisData.groupby(grpPred).mean()
      print("Counts:\n", nRowsGroup)
      print("Means:\n", meanGroup)
      print("Standard Deviations:\n", stdGroup)
      print('\n--------------- Overall Statistics ---------------')
      meanTotal = thisData[yTarget].mean()
      print("             Count = {:.4f}".format(nRowsTotal))
      print("              Mean = {:.4f}".format(meanTotal))
      print("Standard Deviation = {:.4f}".format(stdTotal))
      print('\n--------------- ANOVA Table ---------------')
      print('-- Between Group --')
      print('Degrees of Freedom = ', dfBetween)
      print('    Sum of Squares = ', ssBetween.values[0])
      print('      Mean Squares = ', msBetween.values[0])

      print('-- Within Group --')
      print('Degrees of Freedom = ', dfWithin)
      print('    Sum of Squares = ', ssWithin.values[0])
      print('      Mean Squares = ', msWithin.values[0])

      print('-- The F Test --')
      print('   Statistic = ', fStat.values[0])
      print('Significance = ', fPValue[0])      
 
   return(fPValue)

trainData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\simData4ANOVA.csv',
                       delimiter=',', usecols=['x1','x2', 'x3', 'y'])

# Display the boxplots by group
trainData.boxplot(column='y', by='x1', vert=False)
plt.title("Boxplot of y by Levels of x1")
plt.suptitle("")
plt.xlabel("y")
plt.ylabel("x1")
plt.grid(axis="y")
plt.show()

trainData.boxplot(column='y', by='x2', vert=False)
plt.title("Boxplot of y by Levels of x2")
plt.suptitle("")
plt.xlabel("y")
plt.ylabel("x2")
plt.grid(axis="y")
plt.show()

trainData.boxplot(column='y', by='x3', vert=False)
plt.title("Boxplot of y by Levels of x3")
plt.suptitle("")
plt.xlabel("y")
plt.ylabel("x3")
plt.grid(axis="y")
plt.show()

fPValue = ANOVA_Test(trainData, 'x1', 'y', 'Y')

fPValue = ANOVA_Test(trainData, 'x2', 'y', 'Y')

fPValue = ANOVA_Test(trainData, 'x3', 'y', 'Y')