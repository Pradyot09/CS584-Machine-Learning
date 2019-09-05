import matplotlib.pyplot as plt
import pandas
import sklearn.svm as svm

Spiral = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\SpiralWithCluster.csv',
                         delimiter=',')

nObs = Spiral.shape[0]

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['SpectralCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

# Try the sklearn.svm.LinearSVC
XTrain = Spiral[['x', 'y']]
yTrain = Spiral['SpectralCluster']

svm_Model = svm.LinearSVC(verbose = 1, random_state = 20190410, max_iter = 1000)
thisFit = svm_Model.fit(XTrain, yTrain)

print('Intercept:\n', thisFit.intercept_)
print('Weight Coefficients:\n', thisFit.coef_)

y_predictClass = thisFit.predict(XTrain)

print('Mean Accuracy = ', metrics.accuracy_score(yTrain, y_predictClass))
XTrain['_PredictedClass_'] = y_predictClass

svm_Mean = XTrain.groupby('_PredictedClass_').mean()
print(svm_Mean)

# Plot the prediction
plt.scatter(XTrain[['x']], XTrain[['y']], c = XTrain[['_PredictedClass_']])
plt.plot([-4,4], [0.9,-0.9], color = 'red', linestyle = ':')
plt.xlabel('x')
plt.ylabel('y')
plt.title("Support Vector Machines Prediction")
plt.legend(fontsize = 12, markerscale = 3)
plt.grid(True)
plt.show()

