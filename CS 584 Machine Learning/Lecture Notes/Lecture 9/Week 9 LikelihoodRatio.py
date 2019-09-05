import math
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats as stats
import statsmodels.api as smodel

# Define a function to perform the likelihood ratio test and return the p-value
def LRT_Test (
   inData,          # input data frame
   xPred,           # name of the interval predictor
   yTarget,         # name of the categorical target variable
   debug = "N"      # print debugging information
   ):

   # Extract the relevant columns
   thisData = inData[[xPred, yTarget]].dropna()

   nObs = thisData.shape[0]
   print('Number of Observations Read = ', nObs)

   # Intercept only model
   targetCount = thisData.groupby(yTarget).size()
   nTargetCat = targetCount.shape[0]
   targetProportion = targetCount / nObs
   logLLK0 = targetCount.dot(targetProportion.apply(math.log))

   # Include the predictor
   X = thisData[xPred]
   X = smodel.add_constant(X, prepend=True)

   logit = smodel.MNLogit(thisData[yTarget], X)
   thisFit = logit.fit(method = 'newton', maxiter = 100, tol = 1e-4)

   thisParameter = thisFit.params
   logLLK1 = logit.loglike(thisParameter.values)
   
   # Calculate the G-squared statistic
   gSquared = 2.0 * (logLLK1 - logLLK0)
   dfChi = (nTargetCat - 1.0)

   # Calculate the Chi-squared significance
   chiPValue = stats.chi2.sf(gSquared, dfChi)

   if (debug == 'Y'):
      print('\n--------------- Target Variable Statistics ---------------')
      print('Number of Target Categories =', nTargetCat)
      print('Target Category Counts: \n', targetCount)
      print('Target Category Proportion: \n', targetProportion)

      print('\n--------------- Intercept Only Model ---------------')
      print('Log-Likelihood Value of the Intercept-Only Model = ', logLLK0)
  
      print('\n--------------- Intercept + Predictor Model ---------------')
      print('Log-Likelihood Value of the Intercept+Predictor Model = ', logLLK1)
      print(thisFit.summary())

      print('-- The Chi-Squared Test --')
      print('         Statistic = ', gSquared)
      print('Degrees of Freedom = ', dfChi)
      print('      Significance = {:.7e}'.format(chiPValue))      
 
   return(chiPValue)

trainData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\cars.csv',
                       delimiter=',', usecols=['EngineSize','Horsepower', 'Weight', 'Length', 'Origin'])

chiPValue = LRT_Test(trainData, 'EngineSize', 'Origin', 'Y')

chiPValue = LRT_Test(trainData, 'Horsepower', 'Origin', 'Y')

chiPValue = LRT_Test(trainData, 'Weight', 'Origin', 'Y')

chiPValue = LRT_Test(trainData, 'Length', 'Origin', 'Y')
