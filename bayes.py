# Naive Bayes classification for Spambase data from UCI ML repository


import time
import numpy as np

# set minimal standard deviation
min_std_dev = 0.0001

#timer
start = time.perf_counter()

# import the data to a numpy array
raw_data = np.loadtxt("spambase.data", dtype='float', delimiter=',')

# shuffle the data
raw_data = np.random.permutation(raw_data)

train_spam = np.count_nonzero(raw_data, axis=0) / len(raw_data)

# divide into two sets, with even division of spam instances
data = np.split(raw_data, [int(len(raw_data) / 2)])
train_data = data[0]
test_data = data[1]
toy_data = np.array([(1,2,3,0),(4,5,6,0),(7,8,9,1),(10,11,12,0),(13,14,15,1)])
len_train = len(data[0])
len_test = len(data[1])
attributes = len(data[0][0])
print("Data set sizes: ", len_train, ", ", len_test)

# get p_1 for both divided sets
train_p_1 = (np.count_nonzero(data[0], axis=0) / len_train)[attributes - 1]
test_p_1 = (np.count_nonzero(data[1], axis=0) / len_test)[attributes - 1]

# get p_0 for the training set
train_p_0 = 1 - train_p_1

# check that spam instances were evenly divided between the test and training sets
print("Percent spam in train: ", train_p_1, ",   Percent spam in test: ", test_p_1)
if train_p_1 < 0.38 or test_p_1 < 0.38:
    print ("Spam instances not evenly divided")
    quit() 

print("Spam instances evenly divided between training and test sets")
print("P(1): ", train_p_1 , " P(0): ", train_p_0)


# separate out the labels
train_labels = train_data[0:len_train, attributes - 1]
if np.count_nonzero(train_labels) / len_train != train_p_1:
    print("train_labels doesn't match actual training labels")
else:
    print("train_labels matches actual training labels")

toy_labels = toy_data[0:5, 3]
# print(toy_labels)
# print(np.take(toy_data, np.argwhere(toy_labels == 1)))
# print (np.ndarray.flatten(np.argwhere(toy_labels ==1)))
# print(toy_data[np.ndarray.flatten(np.argwhere(toy_labels == 1)), :])
# print(toy_data[[2,4], :])

# separate the training data into spam and not spam
train_spam_data = train_data[np.ndarray.flatten(np.argwhere(train_labels == 1)), :]
train_real_data = train_data[np.ndarray.flatten(np.argwhere(train_labels == 0)), :]

# delete the last element of each row
train_spam_data = np.delete(train_spam_data, attributes - 1, 1)
train_real_data = np.delete(train_real_data, attributes - 1, 1)
print("shape of train spam data: ", np.shape(train_spam_data))
print("shape of train real data: ", np.shape(train_real_data))

# compute the mean for each of the 57 features
train_spam_means = np.mean(train_spam_data, axis=0)
train_real_means = np.mean(train_real_data, axis=0)

# compute the standard deviation of the 57 features
train_spam_std = np.std(train_spam_data, axis=0)
train_real_std = np.std(train_real_data, axis=0)

# set any standard deviation that is 0 to a non-zero value to avoid divide by zero errors
spam_std_zeros = attributes - 1 - np.count_nonzero(train_spam_std)
real_std_zeros = attributes - 1 - np.count_nonzero(train_real_std)
print("zeros in spam std dev: ", spam_std_zeros, "   zeros in real std dev: ", real_std_zeros)
train_spam_std = np.where(train_spam_std < min_std_dev, min_std_dev, train_spam_std)
train_real_std = np.where(train_real_std < min_std_dev, 0.001, train_real_std)
spam_std_zeros = attributes - 1 - np.count_nonzero(train_spam_std)
real_std_zeros = attributes - 1 - np.count_nonzero(train_real_std)
print("after removing zeros,")
print("zeros in spam std dev: ", spam_std_zeros, "   zeros in real std dev: ", real_std_zeros)

# collect real and spam arrays into a single array for standard deviation and means
train_mean = np.array(train_real_means, train_spam_means)
train_std = np.array(train_real_std, train_spam_std)







print("Time elapsed: ", time.perf_counter() - start)