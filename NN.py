import numpy as np
import matplotlib.pyplot as plt
# Try to learn the XOR function


# Initialize the four possible values for XOR problem.
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([0, 1, 1, 0])

W_0 = (np.random.rand(2,2) - 0.5) * 2
b_0 = 0

W_1 = (np.random.rand(2,1) - 0.5) * 2
b_1 = 0

# Sigmoid Nonlinearity
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cross Entropy Loss
def CrossEntropy(y_predict, y):
    if y == 1:
        return -np.log(y_predict)
    else:
        return -np.log(1-y_predict)


alpha = 0.3

for epoch in range(100000):
    grad_W_0 = 0
    grad_W_1 = 0
    grad_b_0 = 0
    grad_b_1 = 0
    loss = 0

    for i in range(len(X)):
        
        # Forward pass
        z_1 = np.matmul(X[i,:], W_0) + b_0
        a_1 = sigmoid(z_1)
        z_2 = np.matmul(a_1, W_1) + b_1
        y_predict = sigmoid(z_2)
        loss += CrossEntropy(y_predict, y[i])
        # Backward

        # Add up gradient for weights
        grad_W_1_1 = (y_predict - y[i]) * a_1[0]
        grad_W_1_2 = (y_predict - y[i]) * a_1[1]
        
        grad_W_1 += np.reshape([grad_W_1_1, grad_W_1_2], (2,1)) # dL/dy * dy/ds * ds/dw
        

        grad_W_0_1 = (y_predict - y[i]) * W_1[0] * (a_1[0] * (1 - a_1[0]) * X[i, 0])
        grad_W_0_2 = (y_predict - y[i]) * W_1[1] * (a_1[1] * (1 - a_1[1]) * X[i, 0])
        grad_W_0_3 = (y_predict - y[i]) * W_1[0] * (a_1[0] * (1 - a_1[0]) * X[i, 1])
        grad_W_0_4 = (y_predict - y[i]) * W_1[1] * (a_1[1] * (1 - a_1[1]) * X[i, 1])
        
        grad_W_0 += np.reshape([[grad_W_0_1, grad_W_0_2], [grad_W_0_3, grad_W_0_4]], (2,2))

        # Add up gradient for bias parameters
        grad_b_1 += np.reshape([(y_predict - y[i])], (1))
        grad_b_0_1 = (y_predict - y[i]) * W_1[0] * (a_1[0] * (1 - a_1[0]))
        grad_b_0_2 = (y_predict - y[i]) * W_1[1] * (a_1[1] * (1 - a_1[1]))
        grad_b_0 += np.reshape([grad_b_0_1 + grad_b_0_2], (1))

    # Take a batch gradient descent step
    W_1 = W_1 - alpha * (grad_W_1 / len(X))
    W_0 = W_0 - alpha * (grad_W_0 / len(X))
    b_1 = b_1 - alpha * (grad_b_1 / len(X))
    b_0 = b_0 - alpha * (grad_b_0 / len(X))

    print(loss)

points = []
pointsN = []
for i in range(1000):
    randX = np.random.rand()
    randY = np.random.rand()

    # Forward Pass
    xval = [randX, randY]
    z_1 = np.matmul(xval, W_0) + b_0
    a_1 = sigmoid(z_1)
    z_2 = np.matmul(a_1, W_1) + b_1
    y_predict = sigmoid(z_2)

    # Add to 0 or 1 group based off network output
    if (y_predict > 0.5):
        points.append((randX, randY))
    else:
        pointsN.append((randX, randY))
    
points = np.array(points)
pointsN = np.array(pointsN)

# Plot points
plt.scatter(points[:,0], points[:,1])
plt.scatter(pointsN[:,0], pointsN[:,1])
plt.show()
