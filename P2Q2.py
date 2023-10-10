import matplotlib.pyplot as plt
import math
import numpy as np


def predict(test, train):
    TP, TN, FP, FN = 0, 0, 0, 0
    predictions = []

    i = 1
    for test_instance in test:
        predict = (get_prediction(test_instance, train))
        predictions.append(predict)
        print(str(i) + ". " + str(predict))
        i+=1

    for i in range(len(test)):
        if predictions[i] == test[i][-1]:
            if predictions[i] == 0 and test[i][-1] == 0:
                TN+=1
            else:
                TP+=1
        else:
            if predictions[i] == 0 and test[i][-1] == 1:
                FN+=1
            else:
                FP+=1

    return TP, TN, FP, FN

def get_prediction(test_instance, train):
    euclidean_distances = []
    test_instance_wo_y = test_instance[:-1]
    train_wo_y = []
    for instance in train:
        train_wo_y.append(instance[:-1])

    for instance in train_wo_y:
        euclidean_distances.append(get_euclidean_distance(test_instance_wo_y, instance))
    index = euclidean_distances.index(min(euclidean_distances))

    return train[index][-1]

def get_euclidean_distance(test_instance, train_instance):
    sum_of_squared_diffs = 0
    for i in range(len(train_instance)):
        sum_of_squared_diffs += math.pow(test_instance[i] - train_instance[i], 2)

    return math.sqrt(sum_of_squared_diffs)

def convert_to_int(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = int(data[i][j])

    return data

if __name__ == '__main__':
    data = []
    dataFile = open("emails.csv", "r")

    for line in dataFile.readlines():
        features = line.split()[1].split(',')
        features.pop(0)
        data.append(features)
    data.pop(0)
    data = convert_to_int(data)

    # Fold 1
    f1_test = data[:1000]
    f1_train = data[1000:5000]
    TP1, TN1, FP1, FN1 = predict(f1_test, f1_train)
    print("Fold 1: TP = " + str(TP1) + ", TN = " + str(TN1) + ", FP = " + str(FP1) + ", FN = " + str(FN1))

    # Fold 2
    f2_test = data[1000:2000]
    f2_train = data[:1000]
    f2_train[1000:4000] = data[2000:5000]
    TP2, TN2, FP2, FN2 = predict(f2_test, f2_train)
    print("Fold 2: TP = " + str(TP2) + ", TN = " + str(TN2) + ", FP = " + str(FP2) + ", FN = " + str(FN2))

    # Fold 3
    f3_test = data[2000:3000]
    f3_train = data[:2000]
    f3_train[2000:4000] = data[3000:5000]
    TP3, TN3, FP3, FN3 = predict(f3_test, f3_train)
    print("Fold 3: TP = " + str(TP3) + ", TN = " + str(TN3) + ", FP = " + str(FP3) + ", FN = " + str(FN3))

    # Fold 4
    f4_test = data[3000:4000]
    f4_train = data[:3000]
    f4_train[3000:4000] = data[4000:5000]
    TP4, TN4, FP4, FN4 = predict(f4_test, f4_train)
    print("Fold 4: TP = " + str(TP4) + ", TN = " + str(TN4) + ", FP = " + str(FP4) + ", FN = " + str(FN4))

    # Fold 5
    f5_test = data[4000:5000]
    f5_train = data[:4000]
    TP5, TN5, FP5, FN5 = predict(f5_test, f5_train)
    print("Fold 5: TP = " + str(TP5) + ", TN = " + str(TN5) + ", FP = " + str(FP5) + ", FN = " + str(FN5))