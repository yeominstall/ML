from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances

corpus = ['UNC played Duke in basketball', 'Duke lost the basketball game', 'I ate a sandwich']

# no filtering
# vectorizer = CountVectorizer()
# stop word filtered
vectorizer = CountVectorizer(stop_words='english')

arr =  vectorizer.fit_transform(corpus).todense()
print arr
print vectorizer.vocabulary_
print "length: ", len(arr)

print euclidean_distances(arr[0], arr[1])
print euclidean_distances(arr[0], arr[2])
