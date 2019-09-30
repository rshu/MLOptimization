# Cosine similarity is a metric used to measure how similar the documents are irrespective of their size.
# Mathematically, it measures the cosine of the angle between two vectors projected in a multi-dimensional space.
#  The smaller the angle, higher the cosine similarity.

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

# Define the documents
doc_trump = "Mr. Trump became president after winning the political election. Though he lost the support of some republican friends, Trump is friends with President Putin"

doc_election = "President Trump says Putin had no political interference is the election outcome. He says it was a witchhunt by political parties. He claimed President Putin is a friend who had nothing to do with the election"

doc_putin = "Post elections, Vladimir Putin became President of Russia. President Putin had served as the Prime Minister earlier in his political career"

documents = [doc_trump, doc_election, doc_putin]

# Create the Document Term Matrix
# count_vectorizer = CountVectorizer(stop_words='english')

# TfidfVectorizer downweighted words that occur frequently across docuemnts
count_vectorizer = TfidfVectorizer(stop_words='english')
sparse_matrix = count_vectorizer.fit_transform(documents)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix,
                  columns=count_vectorizer.get_feature_names(),
                  index=['doc_trump', 'doc_election', 'doc_putin'])

print(df)
print(cosine_similarity(df, df))