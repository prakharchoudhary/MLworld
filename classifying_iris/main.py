
from matplotlib import pyplot as plt
import numpy as np

from sklearn.datasets import load_iris

data = load_iris()

# load_iris returns an object with iris dataset
features = data.data
feature_names = data.feature_names
target = data.target
target_names = data.target_names

for t in range(3):
	if t==0:
		c = 'r'
		marker = '>'

	elif t==1:
		c = 'g'
		marker = 'o'

	elif t==2:
		c = 'b'
		marker = 'x'

	plt.scatter(features[target == t,0],
				features[target == t,1],
				marker=marker,
				c=c
				)

#we use numpy fancy indexing to get an array of strings:
labels = target_names[target]

#The petal length is the feature at position 2
plength = features[:, 2]

#Build an array of booleans
is_setosa = (labels == 'setosa')

#This is the important step:
max_setosa = plength[is_setosa].max()
min_non_setosa = plength[~is_setosa].min()
print('Maximum of setosa: {0}.'.format(max_setosa))
print ('Minimum of others: {0}.'.format(min_non_setosa))

# ~ is the boolean negation operator
features = features[~is_setosa]
labels = labels[~is_setosa]
# Build a new target variable, is_virginica
is_virginica = (labels == 'virginica') 

#initialise the best accuracy to impossibly low value
best_acc = -1.0
for fi in range(features.shape[1]):
	#we are going to test all possible thresholds
	thresh = features[:, fi]
	for t in thresh:
		#get the vectors for feature 'fi'
		feature_i = features[:, fi]
		#apply thresholf 't'
		pred = (feature_i > t)
		acc = (pred == is_virginica).mean()
		rev_acc = (pred == ~is_virginica).mean()
		if rev_acc > acc:
			reverse = True
			acc = rev_acc
		else:
			reverse = False

		if acc > best_acc:
			best_acc = acc
			best_fi = fi
			best_t = t
			best_reverse = reverse

def is_virginica_test(fi, i, reverse, example):
	'Apply threshold model to a new example'
	test = example[fi]>t
	if reverse:
		test = not test

	return test

# Training accuracy was 96.0%.
# Testing accuracy was 90.0% (N = 50).

from threshold import fit_model, accuracy, predict
correct = 0.0 
for ei in range(len(features)):
	# select all but the one at position 'ei'
	training = np.ones(len(features), bool)
	training[ei] = False
	testing = ~training
	model = fit_model(features[training], is_virginica[training])
	predictions = predict(model, features[testing])
	correct += np.sum(predictions == is_virginica[testing])

acc = correct/float(len(features))
print('Accuracy: {0:.1%}'.format(acc))

##############SEEDS DATASET################################
from load import load_dataset

feature_names = [
    'area',
    'perimeter',
    'compactness',
    'length of kernel',
    'width of kernel',
    'asymmetry coefficien',
    'length of kernel groove',
]

features, labels = load_dataset('seeds')


from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=1)

from sklearn.cross_validation import KFold

kf = KFold(len(features), n_folds=5, shuffle=True)
# 'means' will be a lidt of mean accuracies.
means = []
for training, testing in kf:
	# We fit a model to this fold and then apply it to the
	# testing data with 'predict':

	classifier.fit(features[training], labels[training])
	prediction = classifier.predict(features[testing])

	# np.mean on an arrays of booleans returns a fraction of
	# correct decisions for this fold:
	curmean = np.mean(prediction == labels[testing])
	means.append(curmean)

print("Mean accuracy: {:.1%}".format(np.mean(means)))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

classifier = KNeighborsClassifier(n_neighbors=1)
classifier = Pipeline([('norm', StandardScaler()), ('knn', classifier)])

means = []
for training, testing in kf:
	# We fit a model to this fold and then apply it to the
	# testing data with 'predict':

	classifier.fit(features[training], labels[training])
	prediction = classifier.predict(features[testing])

	# np.mean on an arrays of booleans returns a fraction of
	# correct decisions for this fold:
	curmean = np.mean(prediction == labels[testing])
	means.append(curmean)

print("Mean accuracy: {:.1%}".format(np.mean(means)))
