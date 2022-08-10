from nltk.corpus import stopwords

import tfidf

corpus = open('sample_data.txt', 'r').read().splitlines()
stpwrds = stopwords.words('english')

tfidf = tfidf.TF_IDF(corpus, stpwrds)
print(tfidf.tf_idf())