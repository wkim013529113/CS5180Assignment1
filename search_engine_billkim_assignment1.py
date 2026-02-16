#-------------------------------------------------------------
# AUTHOR: Bill Kim
# FILENAME: search_engine_billkim_assignment1
# SPECIFICATION: description of the program
# FOR: CS 5180- Assignment #1
# TIME SPENT: 1 and half hour
#-----------------------------------------------------------*/

# ---------------------------------------------------------
#Importing some Python libraries
# ---------------------------------------------------------
import csv
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import PorterStemmer

documents = []

# ---------------------------------------------------------
# Reading the data in a csv file
# ---------------------------------------------------------
with open('collection.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
         if i > 0:  # skipping the header
            documents.append (row[0])

# ---------------------------------------------------------
# I rather use pandas for read data. Standard and Easy
# ---------------------------------------------------------
df = pd.read_csv("collection.csv")
documents = df["Document"].tolist()

# ---------------------------------------------------------
# Print original documents
# ---------------------------------------------------------
print(f"Print original documents:: {documents}")
print("")

# ---------------------------------------------------------
# Instantiate CountVectorizer informing 'word' as the analyzer, Porter stemmer as the tokenizer, stop_words as the identified stop words,
# unigrams and bigrams as the ngram_range, and binary representation as the weighting scheme
# ---------------------------------------------------------

# ---------------------------------------------------------
# Define Porter-stemmer tokenizer
# ---------------------------------------------------------
stemmer = PorterStemmer()

def porter_tokenizer(text: str):
    # surface normalization: lowercase + keep only words/numbers as tokens
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return [stemmer.stem(t) for t in tokens]

# ---------------------------------------------------------
# Define stop words (example: pronouns, conjunctions, articles)
# ---------------------------------------------------------
stop_words = ["i", "she", "her", "they", "their", "and", "a", "an", "the"]

# ---------------------------------------------------------
# Instantiate CountVectorizer
# ---------------------------------------------------------
def preprocess_tokens(text: str):
    tokens = porter_tokenizer(text)              # lowercase + regex tokenize + stem
    tokens = [t for t in tokens if t not in stop_words]  # <-- APPLY STOPPING HERE
    return tokens

processed_docs = [preprocess_tokens(d) for d in documents]

unigrams = sorted({t for doc in processed_docs for t in doc})
print(f"Unigrams: {unigrams}")
bigrams = sorted({" ".join(bg) for doc in processed_docs for bg in zip(doc, doc[1:])})
print(f"Unique Bigrams: {bigrams}")
ordered_vocab = unigrams + bigrams
print(f"Ordered Vocabulary {ordered_vocab}")

# Create vectorizer with fixed vocabulary order
vectorizer = CountVectorizer(
    analyzer="word",
    tokenizer=porter_tokenizer,
    stop_words=stop_words,
    ngram_range=(1, 2),
    binary=True,
    token_pattern=None,
    vocabulary=ordered_vocab
)

# ---------------------------------------------------------
# Fit the vectorizer to the documents and encode the them
# ---------------------------------------------------------
vectorizer.fit(documents)
print("Vectorizer vocabulary::")
print(vectorizer.vocabulary_)

document_matrix = vectorizer.transform(documents)
print("Document matrix::")
print(document_matrix.toarray())

# ---------------------------------------------------------
# Inspect vocabulary
# ---------------------------------------------------------
print("Vocabulary:", vectorizer.get_feature_names_out().tolist())

# ---------------------------------------------------------
# Fit the vectorizer to the query and encode it
# ---------------------------------------------------------
query = "I love dogs"

# Encode the query using the learned vocab
query_vector = vectorizer.transform([query])

# Print results
print("Query vector:", query_vector.toarray())

# ---------------------------------------------------------
# Convert matrices to plain Python lists
# ---------------------------------------------------------
doc_vectors = document_matrix.toarray().tolist()

query_vectors = query_vector.toarray().tolist()[0]

# Print results
print("\nDocument matrix (list):")
print(doc_vectors)

print("\nQuery vector (list):")
print(query_vectors)

# ---------------------------------------------------------
# Compute dot product
# ---------------------------------------------------------
scores = []

for i, doc_vector in enumerate(doc_vectors):
    score = sum(q * d for q, d in zip(query_vectors, doc_vector))
    scores.append((i+1, score))  # (document_number, score)

print("Document scores:")
print(scores)

# ---------------------------------------------------------
# Sort documents by score (descending)
# ---------------------------------------------------------
ranking = []

ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
print("\nRanked documents:")
for doc_id, score in ranked_docs:
    print(f"Doc{doc_id} → Score: {score}")

ranking = [doc_id for doc_id, score in sorted(scores, key=lambda x: x[1], reverse=True)]
print("\nDocuments ranking:")
print(ranking)