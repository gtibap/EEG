import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris 
import sklearn.neural_network as ann
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# X, y = np.arange(10).reshape((5, 2)), np.arange(5)
# print(f'{X}\n{y}\n')
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
# print(f'{X_train}\n{y_train}\n')
# print(f'{X_test}\n{y_test}')

# Load the iris dataset
iris = load_iris()
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.45)
print(f'{X_train.shape}\n{y_train.shape}\n')
print(f'{X_test.shape}\n{y_test.shape}')



# # Create an instance of the MLPClassifier class
# neural_network = ann.MLPClassifier(hidden_layer_sizes=(10, 10), activation='relu')
#  # Fit the model to the training data 
# neural_network.fit(X_train, y_train) 
# # Predict the labels of new data 
# y_pred = neural_network.predict(X_test)


# clf = RandomForestClassifier()
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)

cm = confusion_matrix(y_test, y_pred)
print(f'confusion matrix:\n{cm}')
# print(f'y test: {y_test}')
# print(f'y pred: {y_pred}')

# print(y_pred-y_test)