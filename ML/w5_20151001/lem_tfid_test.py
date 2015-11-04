from nltk import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

wordnet_tags = ['n', 'v']
corpus = ['He ate the sandwiches sandwich', 'Every sandwich was eaten by him']
stemmer = PorterStemmer()

print 'Stemmed:', [[stemmer.stem(token) for token in word_tokenize(doc)] for doc in corpus]

def lemmatize(token, tag):
	if tag[0].lower() in ['n', 'v']:
		return lemmatizer.lemmatize(token, tag[0].lower())
	return token

lemmatizer = WordNetLemmatizer()
tagged_corpus = [pos_tag(word_tokenize(doc)) for doc in corpus]
lem_ =  [[lemmatize(token, tag) for token, tag in doc] for doc in tagged_corpus]
print 'Lemmatized:', lem_ 

lem_list = []
b = ""
for j in lem_:
	for i in j:
		b += str(i)+" "
	lem_list.append(b)
	b = ""


print lem_list
# only to-list
#c = [" "+str(word) for word in lem_[1]]
#print c
