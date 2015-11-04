import numpy as np
import re

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from nltk import word_tokenize
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from collections import OrderedDict

# corpus > document > entity
# length of comp.graphic(documents): 584
# number of entities : 10592

def lemmatize(token, tag):
	if tag[0].lower() in ['n', 'v']:
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

# 0. loading data
corpus = fetch_20newsgroups(subset='train', categories=(['comp.graphics']), remove=('headers', 'footers', 'quotes'))

corpsize = len(corpus.data)

# remove email
for doc in range(0, corpsize):
	corpus.data[doc] = re.sub(r'\w+[\w.]*@[\w.]+\.\w+','', corpus.data[doc])

# remove punctuation
for doc in range(0, corpsize):
	corpus.data[doc] = re.sub(r'[^a-zA-Z0-9\s]','', corpus.data[doc])

# lemmatize
lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(doc)) for doc in corpus.data]
l_entities = [[lemmatize(token, tag) for token, tag in doc] for doc in tagged_corpus]

l_list = []
words = ""
count_str = ""

for doc in l_entities:
	for entity in doc:
		words += str(entity)+" "
	l_list.append(words)
	count_str += words
	words = ""

counter = [count_str]

# 2. vectorize
tvect = TfidfVectorizer(norm=None, smooth_idf=False, stop_words='english')
tfidf = tvect.fit_transform(l_list)

print "number of entities:", len(tvect.vocabulary_)
entities = tvect.get_feature_names()
b_entities = tfidf.toarray()
print b_entities

# 2-2. count vectorize
cvect = CountVectorizer(stop_words='english')
count = cvect.fit_transform(counter).todense()
cmaxval = count.max()
cmaxidx = np.unravel_index(count.argmax(), count.shape)
print "count-max:", cvect.get_feature_names()[4914], cmaxval, cmaxidx
count_ = np.array(count)
print np.argwhere(count_>=228)

#print type(count_)
#print count_[[0, 5333]]
print "count-tfidf-max:", cvect.get_feature_names()[5333]

print "b_entities_length:", len(b_entities)
maxval = b_entities.max()
maxidx = np.unravel_index(b_entities.argmax(), b_entities.shape)
print "TF-IDF max:", entities[5333], maxval, maxidx
print b_entities[366][3969]
print b_entities[366][4094]
print b_entities[366][9659]
print entities[3969]
print entities[4094]
print entities[9659]

tf_count = cvect.fit_transform([l_list[366]]).todense()
tf_count_val = tf_count.max()
print "TF_IDF max count(in doc):", tf_count_val
print "TF_IDF max doc's entity num.:", len(cvect.get_feature_names())
print "TF_IDF max doc's wordcount:", len(l_entities[366])

# find index of
#for e in range(0, 10325):
#	if cvect.get_feature_names()[e] == u'jpeg':
#		count_index = e
#print cvect.get_feature_names()[5333]
