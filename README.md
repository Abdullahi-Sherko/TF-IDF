# implementation of Term Frequency Inverse Document Frequency
TF-IDF measures the importance of a word for a document in a corpus.

two metrics are multiplied to calculate TF-IDF
  1) TF frequency of a word in a document
  2) IDF relevance rate of a word in a document in corpus
  
# running
simply 
1) cd TF-IDF/
2) python main.py

if you want run the code on your own corpus
first add your_corpus and your_stopwords 
then
1) import tfidf
2) tf-idf = tfidf.TF-IDF(your_corpus, your_stopwords)
3) tfidf.tf_idf()
