# Naive Bayes classification for Spambase data from UCI ML repository


import time
import math
import random
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



# divide into two sets, with even division of spam instances
data = np.split(raw_data, [int(len(raw_data) / 2)])
train_data = data[0]
test_data = data[1]
len_train = len(data[0])
len_test = len(data[1])
attributes = len(data[0][0])
print("Even division of test and training sets", end='\t')
if len_train - len_test > 1 or len_train - len_test < -1:
    print("FAILED")
    print("len_train: ", len_train, "   len_test: ", len_test)
    quit()
else:
    print("PASSED")



# divide test data into data points and labels
test_labels = test_data[0:len_test, attributes - 1]
test_data = np.delete(test_data, attributes - 1, 1)
print("Creation of test labels array:", end='\t')
if np.max(test_labels) > 1 or np.min(test_labels) < 0:
    print("FAILED")
    quit()
else:
    print("PASSED")
print("Deletion of data labels for testing", end='\t')
if len(test_data[0]) != attributes - 1:
    print("FAILED")
    print("length of new data field: ", len(test_data[0]))
    quit()
else:
    print("PASSED")



# get p_1 for both divided sets
train_p_1 = (np.count_nonzero(data[0], axis=0) / len_train)[attributes - 1]
test_p_1 = (np.count_nonzero(data[1], axis=0) / len_test)[attributes - 1]

# check that spam instances were evenly divided between the test and training sets
print("Spam instances evenly divided between test and train:", end='\t')
if train_p_1 < 0.1 or test_p_1 < 0.1:
    print("FAILED")
    print ("No spam in training or test set, quitting")
    quit() 
elif train_p_1 < 0.38 or test_p_1 < 0.38:
    print("FAILED")
    print("Just run it again")
    quit() 
else:
    # print("Train P(1): ", train_p_1 , "  Train P(0): ", 1 - train_p_1)
    # print("Test P(1): ", test_p_1, "  Test P(0): ", 1 - test_p_1)
    print("PASSED")



# separate out the labels for training
train_labels = train_data[0:len_train, attributes - 1]
print("Separate out the labels for training: ", end='\t')
if np.count_nonzero(train_labels) / len_train != train_p_1:
    print("FAILED")
    print("train_labels doesn't match actual training labels")
    quit()
else:
    print("PASSED")



# separate the training data into spam and not spam
train_spam_data = train_data[np.ndarray.flatten(np.argwhere(train_labels == 1)), :]
train_real_data = train_data[np.ndarray.flatten(np.argwhere(train_labels == 0)), :]
print("Separate the training data into spam and not spam", end='\t')
if np.max(train_spam_data[0:len_train - 1, attributes - 1]) > 1 or np.min(train_spam_data[0:len_train - 1, attributes - 1]) < 1: 
    print("FAILED") 
    quit()
elif np.max(train_real_data[0:len_train - 1, attributes - 1]) > 0 or np.min(train_real_data[0:len_train - 1, attributes - 1]) < 0: 
    print("FAILED") 
    quit()
else:
    print("PASSED")



# delete the last element of each row
# train_spam_data is then [~900 x 57]
# train_real_data is then [~1400 x 57]
train_spam_data = np.delete(train_spam_data, attributes - 1, 1)
train_real_data = np.delete(train_real_data, attributes - 1, 1)
print("train spam data shape: ", np.shape(train_spam_data), " train_real_data shape: ", np.shape(train_real_data))
print("Delete last element of each row:", end='\t')
if len(train_spam_data[0]) != attributes - 1 or len(train_real_data[0]) != attributes - 1:
    print("FAILED")
    quit()
else:
    print("PASSED")



# compute the mean for each of the 57 features
# train spam and real means are then [57 x 1]
train_spam_means = np.mean(train_spam_data, axis=0)
train_real_means = np.mean(train_real_data, axis=0)
print("compute the mean for each of the 57 features", end='\t')
random.seed()
index = random.randrange(attributes - 1)
sum = 0
size = len(train_spam_data)
for i in range(size):
    sum += train_spam_data[i][index]
if train_spam_means[index] != sum / size:
    print("FAILED")
    quit()
else:
    print("PASSED")
random.seed()
index = random.randrange(attributes - 1)
sum = 0
size = len(train_real_data)
for i in range(size):
    sum += train_real_data[i][index]
if train_real_means[index] != sum / size:
    print("\t\t\t\tFAILED")
    quit()
else:
    print("\t\t\t\tPASSED")



# compute the standard deviation of the 57 features
# standard deviation arrays are then [57 x 1]
train_spam_std = np.std(train_spam_data, axis=0)
train_real_std = np.std(train_real_data, axis=0)
print("computing the standard deviation of the 57 features", end='\t')
random.seed()
index = random.randrange(attributes - 1)
sum = 0
size = len(train_spam_data)
for i in range(size):
    sum += pow((train_spam_data[i][index] - train_spam_means[index]), 2)
if train_spam_std[index] != math.sqrt(sum / size):
    print("FAILED")
    quit()
else:
    print("PASSED")
random.seed()
index = random.randrange(attributes - 1)
sum = 0
size = len(train_real_data)
for i in range(size):
    sum += pow((train_real_data[i][index] - train_real_means[index]), 2)
if train_real_std[index] != math.sqrt(sum / size):
    print("\t\t\t\t\tFAILED")
    quit()
else:
    print("\t\t\t\t\tPASSED")



# set any standard deviation that is 0 to a non-zero value to avoid divide by zero errors
train_spam_std = np.where(train_spam_std < min_std_dev, min_std_dev, train_spam_std)
train_real_std = np.where(train_real_std < min_std_dev, min_std_dev, train_real_std)
spam_std_zeros = attributes - 1 - np.count_nonzero(train_spam_std)
real_std_zeros = attributes - 1 - np.count_nonzero(train_real_std)
print("Set minimum standard deviation to non-zero value", end='\t')
if spam_std_zeros != 0 or real_std_zeros != 0:
    print("FAILED")
    quit()
elif np.min(train_spam_std) < min_std_dev or np.min(train_real_std) < min_std_dev:
    print("FAILED")
    quit()
else:
    print("PASSED")
    range = np.ptp(train_spam_std, axis=0)
    print("Range of standard deviations: ")
    print("spam: ", range)
    range = np.ptp(train_real_std)
    print("Range of standard deviations: ")
    print("real: ", range)
    range = np.ptp(train_spam_data)
    print("Range of spam data: ")
    print("spam: ", range)
    range = np.ptp(train_real_data)
    print("Range of real data: ")
    print("real: ", range)
    quit()


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
p_real = math.log(1 - train_p_1) + sum
print("probability that it's real: ", p_real)







print("Time elapsed: ", time.perf_counter() - start)