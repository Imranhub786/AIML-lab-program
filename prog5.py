Program5
........................................
import numpy
def sigmoid(sop):
    return 1.0/(1+numpy.exp(-1*sop))
def sigmoid_sop_deriv(sop):
    return sigmoid(sop)*(1.0-sigmoid(sop))
x1, x2 = 0.1, 0.3
target = 0.03
learning_rate = 0.1
w1 =numpy.random.rand()
w2 =numpy.random.rand()
for k in range(50000):
# Forward Pass
    y = w1*x1 + w2*x2
predicted = sigmoid(y)
# Backward Pass
g1 = 2*(predicted-target) # error_predicted_deriv
g2 = sigmoid_sop_deriv(y)
# Calculate Gradients
gradw1 = x1*g2*g1
gradw2 = x2*g2*g1
w1 = w1 - learning_rate*gradw1
w2 = w2 - learning_rate*gradw2
print("Inputs : ", x1, x2)
print("Expected Target : ", target)
print("Predicted Target: ", predicted)
print("Accuracy : ", (1-numpy.abs(target-predicted))*100, "%")
