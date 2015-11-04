from sklearn.feature_extraction import DictVectorizer

onehot_encoder = DictVectorizer()
instances = [{'city':'New York'}, {'city':'san francisco'}, {'city':'chapel hill'}, {'city':'seoul'}]

print onehot_encoder.fit_transform(instances).toarray()
print onehot_encoder.get_feature_names()
