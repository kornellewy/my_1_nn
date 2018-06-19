import numpy as np


def sigmoid(x, deriv=False):
    if (deriv is True):
        return x*(1-x)

    return 1/(1+np.exp(-x))


# imputy
X = np.array([[5, 5, 50],
              [0, 10, 100],
              [10, 0, 10],
              [10, 10, 10]])

# outputy
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

syn0 = 2*np.random.random((3, 4)) - 1
syn1 = 2*np.random.random((4, 1)) - 1

print(syn0)
print(syn1)

for i in range(1, 100000):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))
    l2 = sigmoid(np.dot(l1, syn1))
    l2_error = y - l2
    if(i % 10000) == 0:
        print("Error: " + str(np.mean(np.abs(l2_error))))
        print(syn0)
        print(syn1)

    l2_delta = l2_error * sigmoid(l2, deriv=True)

    l1_error = l2_delta.dot(syn1.T)

    l1_delta = l1_error * sigmoid(l1, deriv=True)

    syn0 += l0.T.dot(l1_delta)
    syn1 += l1.T.dot(l2_delta)

print("przewiduje: ")
print(l2)
