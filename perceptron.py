import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier
from sklearn.kernel_approximation import PolynomialCountSketch

# percept = Perceptron(random_state=1)
def perceptron_kernel(train_vect, train_lables, test_vect, test_lables):
    
    rbf = PolynomialCountSketch(degree=3, random_state=1, gamma=0.01)
    train_matrix_rbf = rbf.fit_transform(train_vect)
    test_matrix_rbf = rbf.fit_transform(test_vect)
    sgd = SGDClassifier(max_iter=400)

    train_matrix_rbf_rows, train_matrix_rbf_cols = train_matrix_rbf.shape

    print(train_vect.shape + 1)
    print(train_matrix_rbf.shape)

    sgd.fit(train_matrix_rbf[:, 0:100], train_lables)
    print(sgd.score(test_matrix_rbf[:, 0:100], test_lables))
