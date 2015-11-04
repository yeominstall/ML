import numpy as np
import re

from sklearn.datasets import fetch_20newsgroups


corpus = fetch_20newsgroups(subset='train', categories=(['comp.graphics']), remove=('headers', 'footers', 'quotes'))

print corpus.data[583]
