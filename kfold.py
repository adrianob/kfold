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
    f1_scores = []
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
        true_positives = sum([1 for instance in positive_tests
                            if result(knn(train_data, instance, 5)) == instance[-1] ])
        true_negatives = sum([1 for instance in negative_tests
                            if result(knn(train_data, instance, 5)) == instance[-1] ])
        false_positives = len(positive_tests) - true_positives
        false_negatives = len(negative_tests) - true_negatives
        precision = true_positives/float(len(positive_tests))
        recall = true_positives/float(true_positives+false_negatives)
        f1_score = 2*precision*recall/float(precision+recall)
        f1_scores.append(f1_score)
        accuracy = (true_positives+true_negatives)/float(len(tests))
        accuracies.append(accuracy)
        print('Current_fold = ' +  str(current_fold))
        print('Accuracy = ' + str(accuracy))
        print('F1_score = ' + str(f1_score))
        print('------')

    print("Accuracy: %.2f%% (%.2f%%)" % (mean(accuracies)*100, std(accuracies)*100))
    print("F1 score: %.2f%% (%.2f%%)" % (mean(f1_scores)*100, std(f1_scores)*100))

# load data
dataset = loadtxt('diabetes.csv', delimiter=",", skiprows=1)
# split data into X and y
X = array(dataset[:,0:8])
Y = array(dataset[:,8])
X = scale(X)

scaled_dataset = concatenate(( X,   Y[:, None] ), axis = 1)
k = 10 #number of folds
cross_validation(scaled_dataset, k)
