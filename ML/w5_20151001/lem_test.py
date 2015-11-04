from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

print lem.lemmatize('is', 'v')
print lem.lemmatize('least', 'v')
print lem.lemmatize('than', 'n')
print lem.lemmatize('thanks', 'n')
