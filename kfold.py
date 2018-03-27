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
    n_negative = 0
    n_positive = 0
    for distance in distances:
        if distance[0][-1] == 0:
            n_negative += 1
        else:
            n_positive += 1
    return 1 if n_positive >= n_negative else 0

def cross_validation(dtrain, k):
    positive_samples = array([x for x in scaled_dataset if x[-1] == 1])
    negative_samples = array([x for x in scaled_dataset if x[-1] == 0])
    positive_fold_size = int(positive_samples.shape[0]/k)
    negative_fold_size = int(negative_samples.shape[0]/k)

    for current_fold in range(k):
        #copy list
        positive_train = list(positive_samples)
        negative_train = list(negative_samples)
        if current_fold == k - 1:
            positive_tests = positive_samples[current_fold*positive_fold_size:]
            negative_tests = negative_samples[current_fold*negative_fold_size:]
            positive_train[current_fold*positive_fold_size:] = []
            negative_train[current_fold*negative_fold_size:] = []
            positive_train = array(positive_train)
            negative_train = array(negative_train)
        else:
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
        correct = 0
        accuracies = []
        for instance in tests:
            distances = knn(train_data, instance, 5)
            result_class = result(distances)
            if result_class == instance[-1]:
                correct += 1
        accuracy = correct / len(tests)
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
