'''
Early Neural Network for developing a power model of the ARL quad
This network will input multiple variables into a simple NN that should hopefully 
predict power from the given inputs

Spencer Folk 2019
'''

from numpy import *
# import scipy.io as sio   # Needed to read in matlab file
import scipy.optimize as opt
from matplotlib import pyplot as plt
# import pandas as pd  # Not really needed 

#################################################################################################################
############################################### FUNCTIONS #######################################################
#################################################################################################################

# def activationFunction(z):
#     z=z
#     return z
# #	'''
# #	Activation function used to compute hypothesis
# #	'''
# ##	# RELU
# ##	z = z*(z>0)
# #
# #	# Sigmoid
# #	# z = sigmoid(z)
# #
# #	# # Linear
# #	# z = z
# #
# #	return z

# def activationDerivative(z):
#    z=1
#    return z
# #	'''
# #	Derivative of activation function
# #	'''
# #	# RELU
# #	z[z<=0] = 0
# #	z[z>0] = 1
# #
# #	# # Sigmoid
# #	# z = sigmoidGradient(z)
# #
# #	# # Linear
# #	# z = 1
# #
# #
# #	return z

def activationFunction(z, fun):
	'''
	Activation function used to compute hypothesis
	'''

	if(fun=="RELU" or fun=="relu"):
		# RELU
		z = z*(z>0)
	elif(fun=="linear" or fun=="Linear"):
		z = z
	elif(fun=="sigmoid" or fun=="Sigmoid"):
		z = sigmoid(z)

	return z

def activationDerivative(z, fun):
#	'''
#	Derivative of activation function
#	'''
	if(fun=="RELU" or fun=="relu"):	
		# RELU
		z[z<=0] = 0
		z[z>0] = 1
	elif(fun=="linear" or fun=="Linear"):
		z=1
	elif(fun=="Sigmoid" or fun=="sigmoid"):
		z = sigmoidGradient(z)
#
#	# # Linear
#	# z = 1
#
#
	return z

def sigmoid(z):
	'''
	Computes sigmoid function of z, returns numpy array.
	This may or may not be useful for polynomial regression... usually only used for classification problems
	'''

	if type(z) is ndarray:
		g = zeros(z.shape)
	g = 1/(1+exp(z*-1))
	return g

def sigmoidGradient(z):
	'''
	The gradient of the sigmoid function
	'''

	dg = sigmoid(z)*(1-sigmoid(z))

	return dg

def featureNormalize(X):
	'''
	Normalize data for faster learning...
	Find num of columns of X (BEFORE padding 1s)
	'''

	C = X.shape[1]

	X_norm = X
	mu = zeros((1,C))
	sigma = zeros((1,C))

	for i in range(C):
		# For each column... compute mean, std, and normalize the column
		mu[0][i] = mean(X[:,i])
		sigma[0][i] = std(X[:,i])
		X_norm[:,i] = (X[:,i] - mu[0][i])/sigma[0][i]

	return X_norm, mu, sigma


def randInitializeWeights(L_in,L_out,epsilon):
	'''
	Returns randomized weights for a layer with L_in incoming connections and L_out outgoing connections.
	'''
	W = 2*epsilon*random.rand(L_out,L_in) - epsilon
	return W

def unpackWeights(Theta_nn, input_layer_size, hidden_layer_size, num_labels):
	'''
	Unroll Theta into Theta1 and Theta2
	'''

	temp_theta1 = Theta_nn[0:(hidden_layer_size*(input_layer_size+1))]
	Theta1 = reshape(temp_theta1, (hidden_layer_size,input_layer_size+1), 'F')
	temp_theta2 = Theta_nn[(hidden_layer_size*(input_layer_size+1)):]
	Theta2 = reshape(temp_theta2, (num_labels,hidden_layer_size+1), 'F')

	return Theta1, Theta2

def featureNormalize(X):
	'''
	Normalize data for faster learning...
	Find num of columns of X (BEFORE padding 1s)
	'''

	C = X.shape[1]

	X_norm = X
	mu = zeros((1,C))
	sigma = zeros((1,C))

	for i in range(C):
		# For each column... compute mean, std, and normalize the column
		mu[0][i] = mean(X[:,i])
		sigma[0][i] = std(X[:,i])
		X_norm[:,i] = (X[:,i] - mu[0][i])/sigma[0][i]

	return X_norm, mu, sigma

def nnCostFunction(Theta_nn, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
	'''
	Cost and Gradient function for neural net... does both forward and backpropagation
	Gradients and costs were verified in other code
	Theta_nn is the ROLLED OUT array of theta (The two thetas are then unpacked)
	'''

	m = X.shape[0]  # save num of trials (rows)
	J = 0;			# Cost to return

	# Unroll Theta into Theta1 and Theta2
	Theta1, Theta2 = unpackWeights(Theta_nn,input_layer_size, hidden_layer_size, num_labels)

	########### SANITY CHECK - MAKING SURE DIMENSIONS AGREE 
	# a1 = hstack((ones((m,1)),X))
	# z2 = Theta1@a1.transpose()
	# a2 = vstack((ones((m,1)).T,z2)).T
	# a3 = Theta2@a2.transpose()
	# h = a3
	########### END SANITY CHECK

	# Now h is a 5000x10 matrix, where each row is a test case and column probability of the test case being that number (corresponding to the index)

	# Compute h output from the inputs through hidden layer
	a1 = hstack((ones((m,1)),X)) 
	z2 = activationFunction(a1@Theta1.transpose(), "RELU")    
	a2 = hstack((ones((m,1)),z2))
	a3 = activationFunction(a2@Theta2.transpose(), "RELU")
	h = a3

	# Compute cost J

	######### My Method
	cost = zeros((m,1))


	for i in range(m):
		temp = y[i].reshape(-1,)
		# cost[i] = ((-temp.T@log(h[i,:]+1e-5))+(-(1-temp).T@log(1-h[i,:]+1e-5)))
		cost[i] = (temp - h[i,:]).T@(temp-h[i,:])

	cost = sum(cost)/m

	## Regularized cost function
	cost_reg = (lam/(2*m))*(sum(sum(square(Theta1[:,1:]))) + sum(sum(square(Theta2[:,1:]))))

	J =  cost + cost_reg

	############################ GRADIENTS ################################################## 

	delta3 = (a3 - y)   # 5000x10
	delta2 = (Theta2[:,1:].transpose()@delta3.transpose())*activationDerivative(z2.transpose(),"RELU")  # 25x10 * 10x5000 * 25x5000 = 25x5000
	# print(delta2.shape)

	Delta2 = delta3.transpose()@a2 				# Accumulate the gradients
	# print(Delta2.shape)
	Theta2_grad = Delta2/m

	Delta1 = delta2@a1				# Accumulate the gradients, recall no transpose required here because dimensions agree already
	Theta1_grad = Delta1/m

	# Implement regularization of gradients
	Reg1 = (1/m)*Delta1
	Reg1[:,0] = 0					# Bias term has no gradients, isn't changing

	Reg2 = (1/m)*Delta2 + (lam/m)*Theta2
	Reg2[:,0] = 0

	# Roll up and save gradients in grad to output
	grad = hstack((Theta1_grad.flatten('F'),Theta2_grad.flatten('F'))).reshape(-1,1)
	
	# print(a3.shape)
	# print(a2.shape)
	# print(a1.shape)
	# print(y.shape)

	return J, grad


def debugInitializeWeights(fan_out, fan_in):
	'''
	DEBUG
	Initializes the weights using sign such that it's repeatable yet still valid for initialization
	'''

	W = zeros((fan_out,1+fan_in))
	W = reshape(sin(arange(1,W.size+1)),W.shape,'F')/10

	return W

def checkNNGradients(lam):
	'''
	DEBUG
	Creates a small NN to check the gradient and cost function calculations
	'''
	input_layer_size = 3
	hidden_layer_size = 5
	num_labels = 3
	m = 5

	# Initialize weights
	Theta1 = debugInitializeWeights(hidden_layer_size, input_layer_size)
	Theta2 = debugInitializeWeights(num_labels, hidden_layer_size)

	X = debugInitializeWeights(m,input_layer_size-1)
	y = (1 + mod(arange(1,m+1),num_labels)).transpose()
	y = y.reshape(-1,1)

	# Unroll parameters
	Theta_nn = hstack((Theta1.flatten('F'),Theta2.flatten('F'))).reshape(-1,1)

	# Run cost and gradient function
	cost, grad = nnCostFunction(Theta_nn, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

	# Now compute numerical gradient using finite difference method
	numgrad = computeNumericalGradient(Theta_nn, input_layer_size, hidden_layer_size, num_labels, X, y, lam)

	# Print and inspect difference
	print("My Function"+"\t\t\t"+"Numerical Gradient"+"\t\t\t"+"Difference")
	for i in range(grad.size):
		print(grad[i],"\t\t\t",numgrad[i],"\t\t\t",abs(grad[i]-numgrad[i]))

	diff = linalg.norm(numgrad - grad)/linalg.norm(numgrad+grad)
	print("Relative Difference: ",diff)

def computeNumericalGradient(Theta_nn, input_layer_size, hidden_layer_size, num_labels, X, y, lam):
	'''
	DEBUG
	Computes numerical gradient with finite difference method, used to check NN computations
	'''

	numgrad = zeros(Theta_nn.shape)
	perturb = zeros(Theta_nn.shape)
	e = 1e-4

	for p in range(Theta_nn.size):
		# Set perturbation vector
		perturb[p] = e
		loss1 = nnCostFunction(Theta_nn-perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lam)[0]
		loss2 = nnCostFunction(Theta_nn+perturb, input_layer_size, hidden_layer_size, num_labels, X, y, lam)[0]

		# Compute numerical gradient
		numgrad[p] = (loss2 - loss1) / (2*e)
		perturb[p] = 0		# reset perturbation to make sure we're only looking at one weight at a time

	return numgrad

def predict(Theta1, Theta2, X):
	'''
	# Use trained weights (Theta1, Theta2) to output theta from new features generated in hidden NN
	'''
	m = X.shape[0]
	classifiers = zeros((Theta2.shape[0],1))

	pred = zeros((m,1))   #predictions for each trial case

	# First add ones
	X = hstack((ones((m,1)),X))

	# Move inputs through NN
	FirstLayer = activationFunction(X@Theta1.transpose(), "RELU")
	FirstLayer = hstack((ones((m,1)),FirstLayer))  # add null offset in the next layer

	SecondLayer = activationFunction(Theta2@FirstLayer.transpose(), "RELU")
	SecondLayer = SecondLayer.transpose()

	return SecondLayer.reshape(-1,)

#################################################################################################################
################################################# MAIN ##########################################################
#################################################################################################################
'''
The goal is to create a neural network that inputs various parameters from the flight logs, crosses through a 
hidden layer (most likely of the same size) and outputs an estimate of power. 
'''

# Use the following index variables to make life easier and the code below easier to read
mass_idx, time_idx, longitude_idx, latitude_idx, altitude_bar_idx, altitude_ekf_idx = 0,1,2,3,4,5
XY_speed_idx, global_speed_idx, climb_rate_idx = 6,7,8
voltage_idx, current_idx, power_idx = 9,10,11
roll_idx, pitch_idx, yaw_idx = 12,13,14
rc1_idx, rc2_idx, rc3_idx, rc4_idx = 15,16,17,18
motor1_rpm_idx, motor2_rpm_idx, motor3_rpm_idx, motor4_rpm_idx, mean_motor_rpm_idx = 19,20,21,22,23
roll_rate_idx, pitch_rate_idx, yaw_rate_idx = 24,25,26
rc3_rate_idx = 27
x_acc_idx, y_acc_idx, z_acc_idx = 28,29,30
climb_acc_idx, XY_acc_idx, global_acc_idx = 31,32,33
motor1_acc_idx, motor2_acc_idx, motor3_acc_idx, motor4_acc_idx, mean_motor_acc_idx = 34,35,36,37,38

'''
Loading in all data
'''

# Load all the data
all_data = loadtxt('12in_data_long_cruise.txt')
all_data = all_data[all_data[:,0]==2626.6,:]

print(all_data.shape)

all_data[:,roll_idx] = abs(all_data[:,roll_idx])
all_data[:,pitch_idx] = abs(all_data[:,pitch_idx])
all_data[:,yaw_idx] = abs(all_data[:,yaw_idx])

# We need to split up the data into training, cross validation, and testing sets
# These sets need to be sampled such that a wide variety of speeds, masses, etc is considered, so to guarantee that we'll do..

all_data = all_data[all_data[:,XY_speed_idx].argsort()]  # This sorts the data by the specified column (X) to ensure shuffling covers the whole range of X
shuffle_idx = random.permutation(all_data.shape[0])  # This then generates an array of randomly shuffled numbers in the range(length(data))
all_data = all_data[shuffle_idx,:]  # This shuffles the array based on random permutations

num_samples = all_data.shape[0] # Keep track of number of samples

# Now take 60% of data to be training set
train_end_idx = int(floor(0.6*num_samples))
training_data = all_data[0:train_end_idx]

# Take the next 20% for cross validation
cv_end_idx = train_end_idx + int(floor(0.2*num_samples))
cv_data = all_data[(train_end_idx+1):cv_end_idx]

# Take the last 20% to be the test data
test_data = all_data[(cv_end_idx+1):]

print("--- Shuffling Check: The Mean and STD of each data set should be close to each other...\n")
print("Mean\t\t\t\tSTD\t\t\t\t\tMin\t\t\t\tMax")
print(mean(training_data[:,XY_speed_idx]),"\t",std(training_data[:,XY_speed_idx]),min(training_data[:,XY_speed_idx]),max(training_data[:,XY_speed_idx]))
print(mean(cv_data[:,XY_speed_idx]),"\t",std(cv_data[:,XY_speed_idx]),min(cv_data[:,XY_speed_idx]),max(cv_data[:,XY_speed_idx]))
print(mean(test_data[:,XY_speed_idx]),"\t",std(test_data[:,XY_speed_idx]),min(test_data[:,XY_speed_idx]),max(test_data[:,XY_speed_idx]))
print("\n--- End Shuffling Check\n")

'''
Building data sets (this involves getting the inputs we want)
'''

# input_indices = array([XY_speed_idx, global_speed_idx, climb_rate_idx, \
# 	roll_idx, pitch_idx, yaw_idx, rc3_idx, \
# 	motor1_rpm_idx, motor2_rpm_idx, motor3_rpm_idx, motor4_rpm_idx, mean_motor_rpm_idx, \
# 	roll_rate_idx, pitch_rate_idx, yaw_rate_idx, \
# 	rc3_rate_idx, \
# 	climb_acc_idx, XY_acc_idx, global_acc_idx, \
# 	motor1_acc_idx, motor2_acc_idx, motor3_acc_idx, motor4_acc_idx, mean_motor_acc_idx])

input_indices = array([XY_speed_idx, climb_rate_idx, \
	roll_idx, pitch_idx, yaw_idx, \
	motor1_rpm_idx, motor2_rpm_idx, motor3_rpm_idx, motor4_rpm_idx, \
	roll_rate_idx, pitch_rate_idx, yaw_rate_idx, \
	motor1_acc_idx, motor2_acc_idx, motor3_acc_idx, motor4_acc_idx])

X = training_data[:,input_indices]
y = training_data[:,power_idx].reshape(-1,1)

Xval = cv_data[:,input_indices]
yval = cv_data[:,power_idx].reshape(-1,1)

Xtest = test_data[:,input_indices]
ytest = test_data[:,power_idx].reshape(-1,1)

# NORMALIZE ALL FEATURES
X, mu, sigma = featureNormalize(X)
Xval, muval, sigmaval = featureNormalize(Xval)
Xtest, mutest, sigmatest = featureNormalize(Xtest)


# These arrays are now m x (num of inputs) matrices

'''
Generate structure of NN
This includes the size of each layer and initializing the weights (Theta1, Theta2)
'''

input_layer_size = X.shape[1]    		# Input layer size is number of inputs
hidden_layer_size = input_layer_size    # For this NN we're not changing structure of inputs
num_labels = 1    						# Only one output in the output layer

lam = 0 								# Regularization parameter

m = X.shape[0]


# Training the Neural Network:
# epsilon_init = 0.12			# According to Ng this is a good value to choose for symmetry breaking and efficiency
# Ng mentions a good value for epsilon_init = sqrt(6)/sqrt(L_in + L_out)
epsilon_init = sqrt(6)/sqrt(input_layer_size+1+hidden_layer_size)
print("Epsilon value: ",epsilon_init)

initial_Theta1 = randInitializeWeights(input_layer_size+1,hidden_layer_size,epsilon_init)
initial_Theta2 = randInitializeWeights(hidden_layer_size+1,num_labels,epsilon_init)


# Roll up the initial thetas for the optimization routine
Theta_init = hstack((initial_Theta1.flatten('F'),initial_Theta2.flatten('F'))).reshape(-1,1)

'''
Optimization (Learning) Routine
'''

costFunction = lambda p: nnCostFunction(p, input_layer_size, hidden_layer_size, num_labels, X, y, lam)
optimization_routine = opt.minimize(fun = costFunction, 
                                 x0 = Theta_init, 
                                 method = 'TNC',
                                 jac = True, options={'gtol': 1e-4, 'disp': True, 'maxiter': 1000})

'''
Extract Parameters
'''

optimized_Theta_nn = optimization_routine.x
optimized_Theta_nn = optimization_routine.x.reshape(-1,1)

optimized_Theta1, optimized_Theta2 = unpackWeights(optimized_Theta_nn,input_layer_size, hidden_layer_size, num_labels)

'''
Assessing accuracy of model
'''

train_estimate = predict(optimized_Theta1, optimized_Theta2, X)
train_estimate = train_estimate.reshape(-1,1)

print("\n\n-----------------TRAINING SET------------------------")
print("\t\tPrediction\t\t\t\tActual")
print("Mean:\t",mean(train_estimate),"\t",mean(y))
print("STD:\t", std(train_estimate),"\t",std(y))

training_error = abs((y-train_estimate)/y)

print("\nAverage Training Error (Lower is good):\t\t",mean(training_error)*100,"%")

# Another metric: R^2
# R^2 = 1 - SSR/SST , SSR = sum((yi - yhati)^2) , SST = sum((yi - mean(y))^2)

train_SST = sum((y-mean(y))**2)
train_SSR = sum((y - train_estimate)**2)

train_Rsq = 1 - train_SSR/train_SST
print(train_SSR, train_SST)

print("\nR-squared value for training set: ",train_Rsq)

# plt.figure()
# plt.scatter(train_estimate, training_error*100, s=4, alpha=0.4)
# plt.xlabel('Power (W)')
# plt.ylabel('Error (%)')
# plt.ylim((0,100))
# plt.show(block = False)

# Now test against a cross validated set:

cv_estimate = predict(optimized_Theta1, optimized_Theta2, Xval)
cv_estimate = cv_estimate.reshape(-1,1)

print("\n\n--------------------CV SET---------------------------")
print("\t\tPrediction\t\t\t\tActual")
print("Mean:\t",mean(cv_estimate),"\t",mean(yval))
print("STD:\t", std(cv_estimate),"\t",std(yval))

cv_error = abs((yval-cv_estimate)/yval)

print("\nAverage Training Error (Lower is good):\t\t",mean(cv_error)*100,"%")

# Another metric: R^2
# R^2 = 1 - SSR/SST , SSR = sum((yi - yhati)^2) , SST = sum((yi - mean(y))^2)

cv_SST = sum((yval-mean(yval))**2)
cv_SSR = sum((yval - cv_estimate)**2)

cv_Rsq = 1 - cv_SSR/cv_SST

print("\nR-squared value for cross validation set: ",cv_Rsq)

# Now test against a test set:

test_estimate = predict(optimized_Theta1, optimized_Theta2, Xtest)
test_estimate = test_estimate.reshape(-1,1)

print("\n\n-------------------TEST SET--------------------------")
print("\t\tPrediction\t\t\t\tActual")
print("Mean:\t",mean(test_estimate),"\t",mean(ytest))
print("STD:\t", std(test_estimate),"\t",std(ytest))

test_error = abs((ytest-test_estimate)/ytest)

print("\nAverage Training Error (Lower is good):\t\t",mean(test_error)*100,"%")

# Another metric: R^2
# R^2 = 1 - SSR/SST , SSR = sum((yi - yhati)^2) , SST = sum((yi - mean(y))^2)

test_SST = sum((ytest-mean(ytest))**2)
test_SSR = sum((ytest - test_estimate)**2)

test_Rsq = 1 - test_SSR/test_SST

print("\nR-squared value for test set: ",test_Rsq)

'''
Plots
'''
# Undo normalization
X = sigma*X + mu
Xval = sigmaval*Xval + muval
Xtest = sigmatest*Xtest + mutest

plt.figure()

plt.subplot(1,2,1)
plt.scatter(X[:,0], y, s=4, alpha=0.4)
plt.scatter(X[:,0], train_estimate, s=4, alpha=0.4)
# plt.scatter(X[abs(X[:,-6])<0.005,-1], y[abs(X[:,-6])<0.005], c=[0,0,0], alpha=0.4, s=4)
# plt.scatter(X[abs(X[:,-6])<0.005,-1], train_estimate[abs(X[:,-6])<0.005], alpha=0.4, s=4)
plt.xlabel("Last Input")
plt.ylabel("Power (W)")
plt.title("Training")
# plt.legend({"Actual","Predicted"})
plt.grid()

plt.subplot(1,2,2)
plt.scatter(Xval[:,0], yval, s=4, alpha=0.4)
plt.scatter(Xval[:,0], cv_estimate, s=4, alpha=0.4)
plt.xlabel("Last Input")
plt.ylabel("Power (W)")
plt.title("Cross Validation")
# plt.legend({"Actual","Predicted"})
plt.grid()

plt.show()

if test_Rsq > 0.82:
	savetxt("optimized_Theta1.txt",optimized_Theta1,delimiter="\t")
	savetxt("optimized_Theta2.txt",optimized_Theta2,delimiter="\t")
