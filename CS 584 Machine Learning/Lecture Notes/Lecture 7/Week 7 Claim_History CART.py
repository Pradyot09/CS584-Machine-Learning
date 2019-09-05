# Load the necessary libraries
import numpy
import pandas
import sklearn.tree as tree

claimHistory = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\claim_history.csv',
                               delimiter=',', usecols=['EDUCATION', 'OCCUPATION', 'CAR_USE'])

trainData = claimHistory.dropna()

# Convert the EDUCATION ordinal variable into numeric variable
ord_EDU1 = numpy.where(trainData['EDUCATION'].str.strip() == "Below High School", 1, 0)
ord_EDU2 = numpy.where(trainData['EDUCATION'].str.strip() == "High School", 2, 0)
ord_EDU3 = numpy.where(trainData['EDUCATION'].str.strip() == "Bachelors", 3, 0)
ord_EDU4 = numpy.where(trainData['EDUCATION'].str.strip() == "Masters", 4, 0)
ord_EDU5 = numpy.where(trainData['EDUCATION'].str.strip() == "PhD", 5, 0)

ord_EDUCATION = ord_EDU1 + ord_EDU2 + ord_EDU3 + ord_EDU4 + ord_EDU5

# Convert the OCCUPATION nominal variable into dummy variables
cat_occupation = trainData[['OCCUPATION']].astype('category')
occupation_inputs = pandas.get_dummies(cat_occupation)

# Join the columns to create the matrix of input predictors
X_inputs = pandas.DataFrame({'ord_EDUCATION': ord_EDUCATION})
X_inputs = X_inputs.join(occupation_inputs)
X_inputs_name = X_inputs.columns.values.tolist()

# Specify the target variable
Y_target = trainData['CAR_USE']

# How many missing values are there?
print('Number of Missing Observations:')
print(X_inputs.isnull().sum())
print(Y_target.isnull().sum())

# Specify the CART model
classTree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=60616)

# Build the CART model
claimHistory_CART = classTree.fit(X_inputs, Y_target)
print('Accuracy of Decision Tree classifier on training set: {:.6f}' .format(classTree.score(X_inputs, Y_target)))

# Render the tree diagram
import graphviz
dot_data = tree.export_graphviz(claimHistory_CART,
                                out_file=None,
                                impurity = True, filled = True,
                                feature_names = X_inputs_name,
                                class_names = ['Commercial', 'Private'])

graph = graphviz.Source(dot_data)

graph

graph.render('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Job\\claimHistory_CART')
