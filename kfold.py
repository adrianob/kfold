# stratified k-fold cross validation evaluation of xgboost model
from numpy import *
import xgboost
import operator
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import scale
# set_printoptions(threshold=nan)
import math
def euclideanDistance(instance1, instance2):
	distance = 0
	for x in range(len(instance1) - 1):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)

def knn(dtrain, dtest, k):
    distances = []
    for instance in dtrain:
        distance = euclideanDistance(instance, dtest)
        distances.append((instance, distance))

    distances.sort(key=operator.itemgetter(1))
    return distances[0:k]

# majority voting
def result(distances):
    n_negative = 0
    n_positive = 0
    for distance in distances:
        if distance[0][-1] == 0:
            n_negative += 1
        else:
            n_positive += 1
    return 1 if n_positive >= n_negative else 0

# load data
dataset = loadtxt('diabetes.csv', delimiter=",", skiprows=1)
# split data into X and y
X = array(dataset[:,0:8])
Y = array(dataset[:,8])

# scaling
# for x in range(X.shape[1]):
#     feature = X[:, x]
#     for index, data in enumerate(feature):
#         X[index][x] = ( data - mean(feature) ) / float(std(feature))

X = scale(X)
scaled_dataset = concatenate(( X,   Y[:, None] ), axis = 1)
X = array(scaled_dataset[:,0:8])
Y = array(scaled_dataset[:,8])
positive_samples = array([x for x in scaled_dataset if x[-1] == 1])
negative_samples = array([x for x in scaled_dataset if x[-1] == 0])

distances = knn(scaled_dataset, scaled_dataset[-1], 5)
result_class = result(distances)
print(result_class)

k = 10 #number of folds

# CV model
# model = xgboost.XGBClassifier()
# kfold = StratifiedKFold(n_splits=10, random_state=7)
# results = cross_val_score(model, X, Y, cv=kfold)
# print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
