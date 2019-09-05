import numpy
import pandas

import sklearn.naive_bayes as NB

# Define a function to visualize the percent of a particular target category by a nominal predictor
def RowWithColumn (
   rowVar,          # Row variable
   columnVar,       # Column predictor
   show = 'ROW'):   # Show ROW fraction, COLUMN fraction, or BOTH table

   countTable = pandas.crosstab(index = rowVar, columns = columnVar, margins = False, dropna = True)
   print("Frequency Table: \n", countTable)
   print( )

   if (show == 'ROW' or show == 'BOTH'):
       rowFraction = countTable.div(countTable.sum(1), axis='index')
       print("Row Fraction Table: \n", rowFraction)
       print( )

   if (show == 'COLUMN' or show == 'BOTH'):
       columnFraction = countTable.div(countTable.sum(0), axis='columns')
       print("Column Fraction Table: \n", columnFraction)
       print( )

   return

inData = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\CustomerSurveyData.csv',
                         delimiter=',')

# EBilling -> CreditCard, Gender, JobCategory
subData = inData[['CreditCard', 'Gender', 'JobCategory', 'EBilling']].dropna()

catEBilling = subData['EBilling'].unique()
catCreditCard = subData['CreditCard'].unique()
catGender = subData['Gender'].unique()
catJobCategory = subData['JobCategory'].unique()

print('Unique Values of EBilling: \n', catEBilling)
print('Unique Values of CreditCard: \n', catCreditCard)
print('Unique Values of Gender: \n', catGender)
print('Unique Values of JobCategory: \n', catJobCategory)

print('Unique Values of EBilling: \n', catEBilling)
print('Unique Values of CreditCard: \n', catCreditCard)
print('Unique Values of Gender: \n', catGender)
print('Unique Values of JobCategory: \n', catJobCategory)

RowWithColumn(rowVar = subData['EBilling'], columnVar = subData['CreditCard'], show = 'ROW')
RowWithColumn(rowVar = subData['EBilling'], columnVar = subData['Gender'], show = 'ROW')
RowWithColumn(rowVar = subData['EBilling'], columnVar = subData['JobCategory'], show = 'ROW')