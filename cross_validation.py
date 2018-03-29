#!/usr/bin/env python3

# Universidade Federal do Rio Grande do Sul
# INF01017 - Aprendizado de Máquina - 2018/1

# Trabalho: Repeated K-fold Cross Validation

# Adriano Carniel Benin
# Diogo Campos da Silva
# João Pedro Bielko Weit


import math
import random


INPUT_FILE = 'diabetes.csv'
OUTPUT_FILES = 'knn-%d.csv'

KNN = [1, 3, 5, 7, 9]
FOLDS = 5
REPEAT = 10


def main():
    print('KNN, %d-fold cross validation, %d repetitions' % (FOLDS, REPEAT))
    print()

    data = normalize_data(load_csv(INPUT_FILE))
    for num_neighbors in KNN:
        print('K =', num_neighbors)

        metrics = knn_cross_validation(data, num_neighbors, FOLDS, REPEAT)

        accuracies = [t[0] for t in metrics]
        f1_measures = [t[3] for t in metrics]

        print('Accuracy: mean = %f, sd = %f' % stats(accuracies))
        print('F1-measure: mean = %f, sd = %f' % stats(f1_measures))
        print()

        with open(OUTPUT_FILES % num_neighbors, 'w') as file:
            file.write('Accuracy,Precision,Recall,F1Measure')
            for tuple in metrics:
                file.write('\n' + ','.join(str(val) for val in tuple))


def knn_cross_validation(normalized_data, num_neighbors, num_folds, repeat):
    negatives = [instance for instance in normalized_data if instance[-1] == 0]
    positives = [instance for instance in normalized_data if instance[-1] == 1]

    metrics = []

    for repetition in range(repeat):
        neg_folds = split_into_random_folds(negatives, num_folds)
        pos_folds = split_into_random_folds(positives, num_folds)
        folds = [neg_folds[i] + pos_folds[i] for i in range(num_folds)]

        for i in range(num_folds):
            test_data = folds[i]
            other_folds = (fold for j, fold in enumerate(folds) if j != i)
            training_data = concat(other_folds)

            results = [[0, 0],  # true negatives, false negatives
                       [0, 0]]  # false positives, true positives

            for instance in test_data:
                predicted_class = knn(training_data, instance, num_neighbors)
                true_class = instance[-1]
                results[predicted_class][true_class] += 1

            accuracy = float(results[0][0] + results[1][1]) / len(test_data)

            precision = float(results[1][1]) / (results[1][1] + results[1][0])
            recall = float(results[1][1]) / (results[1][1] + results[0][1])
            f1_measure = 2 * precision * recall / (precision + recall)

            metrics.append((accuracy, precision, recall, f1_measure))

        print('.', end='', flush=True)

    print()
    return metrics


def split_into_random_folds(data, num_folds):
    random.shuffle(data)
    fold_size = int(math.ceil(len(data) / float(num_folds)))
    folds = [data[i * fold_size : (i + 1) * fold_size] for i in range(num_folds)]
    return folds


def concat(lists):
    result = []
    for list in lists:
        result.extend(list)
    return result


def knn(training_data, new_instance, num_neighbors):
    distances = []
    for instance in training_data:
        distance = euclidean_distance(instance, new_instance)
        klass = instance[-1]
        distances.append((distance, klass))

    distances.sort()
    nearest = distances[0 : num_neighbors]

    num_positives = sum(klass for (_, klass) in nearest)
    return 1 if num_positives > (num_neighbors / 2) else 0


def euclidean_distance(a, b):
    squares = (math.pow(a[i] - b[i], 2) for i in range(len(a) - 1))
    return math.sqrt(sum(squares))


def stats(values):
    mean = sum(values) / len(values)
    variance = sum(math.pow(val - mean, 2) for val in values) / len(values)
    standard_deviation = math.sqrt(variance)
    return (mean, standard_deviation)


def normalize_data(data):
    num_attributes = len(data[0]) - 1
    mins = data[0][0:-1]
    maxs = data[0][0:-1]

    for instance in data:
        for j in range(num_attributes):
            if instance[j] < mins[j]: mins[j] = instance[j]
            if instance[j] > maxs[j]: maxs[j] = instance[j]

    normalized_data = []

    for instance in data:
        normalized_instance = [
            float(value - mins[j]) / (maxs[j] - mins[j])
            for j, value in enumerate(instance[0:-1])
        ] + [instance[-1]]
        normalized_data.append(normalized_instance)

    return normalized_data


def load_csv(filename):
    data = []

    with open(filename) as file:
        file.readline()  # skip first line
        while True:
            line = file.readline()
            if not line: break

            fields = line.split(',')
            # read all attributes as floats, except for the class
            instance = [float(val) for val in fields[0:-1]] + [int(fields[-1])]
            data.append(instance)

    return data


if __name__ == '__main__':
    main()
