
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import DATA_DIR

from nltk.stem import SnowballStemmer
english_stemmer = SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
	def build_analyzer(self):
		analyzer = super(StemmedCountVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

class StemmedTfidfVectorizer(TfidfVectorizer):
	def build_analyzer(self):
		analyzer = super(TfidfVectorizer, self).build_analyzer()
		return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

vectorizer = StemmedTfidfVectorizer(min_df=1, stop_words="english", decode_error="ignore")  #stop_words is used to not consider those words which occur quite frequently that too in varying contexts.
# print(vectorizer)

# content = ["How to format my hard disk", "Hard disk format problems"]
# X = vectorizer.fit_transform(content)
# print vectorizer.get_feature_names()
# print(X.toarray().transpose())

posts = [open(os.path.join(DATA_DIR, f)).read() for f in os.listdir(DATA_DIR)]	

X_train = vectorizer.fit_transform(posts)
num_samples, num_features = X_train.shape
print("#samples: %d, #features: %d" % (num_samples, num_features))

print(vectorizer.get_feature_names())

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])

print new_post_vec 
#(a,b) is the location of the feature or each word in new post and it is followed by the number of times it occurs
print new_post_vec.toarray() 
#returns a one-d matrix which has the number of times each feature occurs in the new post

#===================== Calculating the Euclidean Distance ========================================#
import scipy as sp

def dist_raw(v1, v2):
	delta = v1-v2
	return sp.linalg.norm(delta.toarray())
	# norm() function calculates the euclidean function.


# print(X_train.getrow(3).toarray())
# print(X_train.getrow(4).toarray())

# Both the files 04 and 05 have the same text. Only file 04 has every word repeated thrice.
# So they should have the same distance whch is not the case, hence simply finding the euclidean distance
# work. We must normalize the data.

#================================= Normalizing the word count vectors =================================#


def dist_norm(v1, v2):
	v1_normalised = v1/sp.linalg.norm(v1.toarray())
	v2_normalised = v2/sp.linalg.norm(v2.toarray())
	delta = v1_normalised - v2_normalised
	return sp.linalg.norm(delta.toarray())


import sys
best_doc = None
best_dist = sys.maxint
best_i = None
for i, post in enumerate(posts):
	if post == new_post:
		continue
	post_vec = X_train.getrow(i)
	d = dist_norm(post_vec, new_post_vec)
	print("=== Post %i with dist = %.2f: %s" % (i, d, post))

	if d<best_dist:
		best_dist = d
		best_i = i
print("Best post is %i with dist=%.2f"%(best_i, best_dist))