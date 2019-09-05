import matplotlib.pyplot as plt
import pandas
import sklearn.neural_network as nn

Spiral = pandas.read_csv('C:\\Users\\minlam\\Documents\\IIT\\Machine Learning\\Data\\SpiralWithCluster.csv',
                         delimiter=',')

nObs = Spiral.shape[0]

plt.scatter(Spiral[['x']], Spiral[['y']], c = Spiral[['SpectralCluster']])
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

def Build_NN_Toy (nLayer, nHiddenNeuron):

    # Build Neural Network
    nnObj = nn.MLPClassifier(hidden_layer_sizes = (nHiddenNeuron,)*nLayer,
                             activation = 'tanh', verbose = True,
                             max_iter = 10000, random_state = 60616)

    X = Spiral[['x', 'y']]
    y = Spiral[['SpectralCluster']]

    fit_nn = nnObj.fit(X, y) 
    pred_nn = nnObj.predict(X)

    print('Output Activation Function:', nnObj.out_activation_)
    print('             Mean Accuracy:', nnObj.score(X, y))

        # Plot the prediction
    X['myColor'] = pred_nn.copy()
    plt.scatter(X[['x']], X[['y']], c = X[['myColor']])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("%d Hidden Layers, %d Hidden Neurons" % (nLayer, nHiddenNeuron))
    plt.legend(fontsize = 12, markerscale = 3)
    plt.grid(True)
    plt.show()

Build_NN_Toy (nLayer = 2, nHiddenNeuron = 1)
Build_NN_Toy (nLayer = 2, nHiddenNeuron = 3)
Build_NN_Toy (nLayer = 2, nHiddenNeuron = 5)
Build_NN_Toy (nLayer = 2, nHiddenNeuron = 10)
