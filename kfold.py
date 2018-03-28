from numpy import *
import operator
from sklearn.preprocessing import scale
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
    n_positive = sum([1 for x in distances if x[0][-1] == 1 ])
    return 1 if n_positive >= len(distances)/2 else 0

def cross_validation(dtrain, k):
    positive_samples = array([x for x in scaled_dataset if x[-1] == 1])
    negative_samples = array([x for x in scaled_dataset if x[-1] == 0])
    positive_fold_size = int(ceil(positive_samples.shape[0]/k))
    negative_fold_size = int(ceil(negative_samples.shape[0]/k))

    accuracies = []
    for current_fold in range(k):
        #copy list
        positive_train = list(positive_samples)
        negative_train = list(negative_samples)
        #slice test data
        positive_tests = positive_samples[current_fold*positive_fold_size : ( current_fold + 1 ) * positive_fold_size]
        negative_tests = negative_samples[current_fold*negative_fold_size : ( current_fold + 1 ) * negative_fold_size]
        #get remaining data for training
        positive_train[current_fold*positive_fold_size : ( current_fold + 1 ) * positive_fold_size] = []
        negative_train[current_fold*negative_fold_size : ( current_fold + 1 ) * negative_fold_size] = []

        positive_train = array(positive_train)
        negative_train = array(negative_train)
        tests = concatenate(( positive_tests, negative_tests ))
        train_data = concatenate(( positive_train, negative_train ))
        correct = sum([1 for instance in tests
                       if result(knn(train_data, instance, 5)) == instance[-1] ])
        accuracy = correct / float(len(tests))
        accuracies.append(accuracy)
        print(current_fold)
        print(accuracy)
        print()

    print("Accuracy: %.2f%% (%.2f%%)" % (mean(accuracies)*100, std(accuracies)*100))

# load data
dataset = loadtxt('diabetes.csv', delimiter=",", skiprows=1)
# split data into X and y
X = array(dataset[:,0:8])
Y = array(dataset[:,8])
X = scale(X)

scaled_dataset = concatenate(( X,   Y[:, None] ), axis = 1)
k = 10 #number of folds
cross_validation(scaled_dataset, k)
