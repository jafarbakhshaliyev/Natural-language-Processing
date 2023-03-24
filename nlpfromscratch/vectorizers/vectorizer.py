import numpy as np
from scipy.sparse import lil_matrix
from collections import Counter

class Vectorizer:
    
    def BOW(self, X, preprocess_fn):
        
        tokenized_docs, vocab, word_index = self.vocab_index(X, preprocess_fn)
    
        vec = lil_matrix((len(tokenized_docs), len(vocab)), dtype=np.float64)

        for i, doc in enumerate(tokenized_docs):
            for word in doc:
                vec[i, word_index[word]] = doc.count(word)
            
        return vec 
    
    def TF_IDF(self, X, preprocess_fn):
        
        tokenized_docs, vocab, word_index = self.vocab_index(X, preprocess_fn)
        word_counts = Counter([word for doc in tokenized_docs for word in set(doc)])
        
        tf_matrix = lil_matrix((len(tokenized_docs), len(vocab)), dtype=np.float64)

        for i, doc in enumerate(tokenized_docs):
            for word in doc:
                tf_matrix[i, word_index[word]] = doc.count(word)

        idf_matrix = lil_matrix((len(vocab), len(vocab)), dtype=np.float64)
        
        for i, word in enumerate(vocab):
            idf_matrix[i, i] = np.log((len(tokenized_docs) + 1) / word_counts[word])

        tf_idf_matrix = tf_matrix * idf_matrix
        
        return tf_idf_matrix
    
    def vocab_index(self, X, preprocess_fn):
        
        tokenized_docs = [preprocess_fn(X.iloc[i]['text']) for i in range(X.shape[0])]
        vocab = set([word for doc in tokenized_docs for word in doc])
        word_index = {word: i for i, word in enumerate(vocab)}
        
        return tokenized_docs, vocab, word_index
        