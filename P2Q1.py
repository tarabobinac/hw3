import matplotlib.pyplot as plt
import math
import numpy as np

def predict(x11, x21, x10, x20, x1_test, x2_test):
    x11_test, x21_test, x10_test, x20_test = [], [], [], []

    for i in range(len(x1_test)):
        prediction = get_1NN(x11, x21, x10, x20, x1_test[i], x2_test[i])
        if prediction == 1:
            x11_test.append(x1_test[i])
            x21_test.append(x2_test[i])
        else:
            x10_test.append(x1_test[i])
            x20_test.append(x2_test[i])

    return x11_test, x21_test, x10_test, x20_test

def get_1NN(x11, x21, x10, x20, x1t, x2t):
    closest_pos_instance = math.sqrt(math.pow(x11[0] - x1t, 2) + math.pow(x21[0] - x2t, 2))
    closest_neg_instance = math.sqrt(math.pow(x10[0] - x1t, 2) + math.pow(x20[0] - x2t, 2))

    for i in range(len(x11)):
        distance = math.sqrt(math.pow(x11[i] - x1t, 2) + math.pow(x21[i] - x2t, 2))
        if distance < closest_pos_instance:
            closest_pos_instance = distance

    for i in range(len(x10)):
        distance = math.sqrt(math.pow(x10[i] - x1t, 2) + math.pow(x20[i] - x2t, 2))
        if distance < closest_neg_instance:
            closest_neg_instance = distance

    if closest_pos_instance < closest_neg_instance:
        return 1

    return 0

if __name__ == '__main__':
    dataFile = open("D2z.txt", "r")
    x11, x21, x10, x20 = [], [], [], []

    for line in dataFile:
        xxy = line.split()
        if float(xxy[2]) == 1:
            x11.append(float(xxy[0]))
            x21.append(float(xxy[1]))
        else:
            x10.append(float(xxy[0]))
            x20.append(float(xxy[1]))

    test_point_range = np.arange(-2, 2.1, 0.1).tolist()
    x1_test, x2_test = [], []
    for x1 in range(len(test_point_range)):
        for x2 in range(len(test_point_range)):
            x1_test.append(round(test_point_range[x1], 1))
            x2_test.append(round(test_point_range[x2], 1))

    x11_test, x21_test, x10_test, x20_test = predict(x11, x21, x10, x20, x1_test, x2_test)

    plt.figure()
    plt.scatter(x11_test, x21_test, marker='.', s=10, color='orange')
    plt.scatter(x10_test, x20_test, marker='.', s=10, color='blue')
    plt.scatter(x11, x21)
    plt.scatter(x10, x20)
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.title('D2z Scatter Plot')
    plt.grid()
    plt.show()