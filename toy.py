# Naive Bayes classification for Spambase data from UCI ML repository


import time
import math
import numpy as np

# set minimal standard deviation
min_std_dev = 0.0001

#timer
start = time.perf_counter()

# import the data to a numpy array
raw_data = np.loadtxt("spambase.data", dtype='float', delimiter=',')
# raw_data = np.array([(1,2,3,0),(4,5,6,0),(7,8,9,1),(10,11,12,1),(13,14,15,1),(16,17,18,0),(1,2,4,1),(1,3,5,1),(3,5,7,0)])

# shuffle the data
raw_data = np.random.permutation(raw_data)

train_spam = np.count_nonzero(raw_data, axis=0) / len(raw_data)

# divide into two sets, with even division of spam instances
data = np.split(raw_data, [int(len(raw_data) / 2)])
train_data = data[0]
test_data = data[1]
len_train = len(data[0])
len_test = len(data[1])
attributes = len(data[0][0])
print("Data set sizes: ", len_train, ", ", len_test)

# divide the test data into data points and labels
print("test data:\n", test_data)
test_labels = test_data[0:len_test, attributes - 1]
test_data = np.delete(test_data, attributes - 1, 1)
print("test labels:\n ", test_labels)
print("test data: \n", test_data)

# get p_1 for both divided sets
train_p_1 = (np.count_nonzero(data[0], axis=0) / len_train)[attributes - 1]
test_p_1 = (np.count_nonzero(data[1], axis=0) / len_test)[attributes - 1]

# get p_0 for the training set
train_p_0 = 1 - train_p_1

# check that spam instances were evenly divided between the test and training sets
print("Percent spam in train: ", train_p_1, ",   Percent spam in test: ", test_p_1)
if train_p_1 < 0.1 or test_p_1 < 0.1:
    print ("No spam in training or test set, quitting")
    quit() 
if train_p_1 < 0.38 or test_p_1 < 0.38:
    print ("Spam instances not evenly divided")
    quit() 

# print("Spam instances evenly divided between training and test sets")
print("P(1): ", train_p_1 , " P(0): ", train_p_0)


# separate out the labels for training
train_labels = train_data[0:len_train, attributes - 1]
if np.count_nonzero(train_labels) / len_train != train_p_1:
    print("train_labels doesn't match actual training labels")
else:
    print("train_labels matches actual training labels")

# toy_labels = toy_data[0:5, 3]
# print(toy_labels)
# print(np.take(toy_data, np.argwhere(toy_labels == 1)))
# print (np.ndarray.flatten(np.argwhere(toy_labels ==1)))
# print(toy_data[np.ndarray.flatten(np.argwhere(toy_labels == 1)), :])
# print(toy_data[[2,4], :])

# separate the training data into spam and not spam
train_spam_data = train_data[np.ndarray.flatten(np.argwhere(train_labels == 1)), :]
train_real_data = train_data[np.ndarray.flatten(np.argwhere(train_labels == 0)), :]
print("train spam data: \n", train_spam_data)
print("train real data: \n", train_real_data)

# delete the last element of each row
train_spam_data = np.delete(train_spam_data, attributes - 1, 1)
train_real_data = np.delete(train_real_data, attributes - 1, 1)
print("train spam data: \n", train_spam_data)
print("train real data: \n", train_real_data)
# print("shape of train spam data: ", np.shape(train_spam_data))
# print("shape of train real data: ", np.shape(train_real_data))

# compute the mean for each of the 57 features
train_spam_means = np.mean(train_spam_data, axis=0)
train_real_means = np.mean(train_real_data, axis=0)
print("train spam means: \n", train_spam_means)
print("train real means: \n", train_real_means)

# compute the standard deviation of the 57 features
train_spam_std = np.std(train_spam_data, axis=0)
train_real_std = np.std(train_real_data, axis=0)
print("train spam std: \n", train_spam_std)
print("train real std: \n", train_real_std)

# set any standard deviation that is 0 to a non-zero value to avoid divide by zero errors
spam_std_zeros = attributes - 1 - np.count_nonzero(train_spam_std)
real_std_zeros = attributes - 1 - np.count_nonzero(train_real_std)
print("zeros in spam std dev: ", spam_std_zeros, "   zeros in real std dev: ", real_std_zeros)
train_spam_std = np.where(train_spam_std < min_std_dev, min_std_dev, train_spam_std)
train_real_std = np.where(train_real_std < min_std_dev, min_std_dev, train_real_std)
spam_std_zeros = attributes - 1 - np.count_nonzero(train_spam_std)
real_std_zeros = attributes - 1 - np.count_nonzero(train_real_std)
print("after removing zeros,")
print("zeros in spam std dev: ", spam_std_zeros, "   zeros in real std dev: ", real_std_zeros)

# collect real and spam arrays into a single array for standard deviation and means
# print(train_real_means)
# print(train_spam_means)
# train_mean = np.array([train_real_means])
# train_mean = np.append(train_mean, [train_spam_means], axis=0)
# print(train_mean)
# print(train_real_std)
# print(train_spam_std)
# train_std = np.array([train_real_std])
# train_std = np.append(train_std, [train_spam_std], axis=0)
# print(train_std)


# subtract the mean from the data point
# print(train_real_means)
# train_real_means = np.append(train_real_means, 0)
# print("\ntest data: ", test_data)
# print("real mean: ", train_real_means)
real_diff = np.subtract(test_data, train_real_means)
print("test data - real mean: \n", real_diff)

# square the difference to get the numerator for the exponent
numerator = np.power(real_diff, 2)
print("numerator: \n", numerator)

# get the denominator
denominator = 2 * np.power(train_real_std, 2)
print("denominator: \n", denominator)

# get the exponent
exponent = -1 * np.divide(numerator, denominator)
print("exponent: \n", exponent)

# raise e to the exponent
exponential = np.exp(exponent)
print("exponential: \n", exponential)

# divide the exponential by sqrt(2 pi) * standard deviation
real_values = np.divide(exponential, math.sqrt(2 * np.pi) * train_real_std)
print("real values: \n", real_values)

# take the log of all the values
logs = np.log(real_values) 
print("logs: \n", logs)

# get the probability that it's real
sum = np.sum(logs)
p_real = math.log(train_p_0) + sum
print("probability that it's real: ", p_real)







print("Time elapsed: ", time.perf_counter() - start)