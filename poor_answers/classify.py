
def fetch_posts():
	for line in open("data.tsv", "r"):
		post_id, text = line.split("\t")
		yield int(post_id), text.strip()

import re
import numpy as np
from sklearn import neighbors

# knn = neighbors.KNeighborsClassfier(n_neighbors=2)
# print knn

code_match = re.compile('<pre>(.*?)</pre>',re.MULTILINE | re.DOTALL)
link_match = re.compile('<a href="http://.*?".*?>(.*?)</a>',re.MULTILINE | re.DOTALL)
tag_match = re.compile('<[^>]*>',re.MULTILINE | re.DOTALL)

def extract_features_from_body(s):

	link_count_in_code = 0
	#count links in code to later subtract them
	for match_str in code_match.findall(s):
		link_count_in_code += len(link_match.findall(match_str))

	return len(link_match.findall(s)) - link_count_in_code


## Training the classifier
X = np.asarray([extract_features_from_body(text) for post_id, text in fetch_posts() if post_id in all_answers])
knn = neighbors.KNeighborsClassifier()
knn.fit(X, Y)

###### Measuring the classifier's performance
from sklearn.cross_validation import KFold
scores=[]

cv = KFold(n=len(X), k=10, indices=True)

for train, test in cv:
	X_train, y_train = X[train], Y[train]
	X_test, y_test = X[test], Y[test]
	clf = neighbors.KNeighborsClassifier()
	clf.fit(X,Y)
	scores.append(clf.score(X_test, y_test))

print("Mean(scores)=%.5f\tStddev(scores)=%.5ff"%(np.mean(scores), np.std(scores)))