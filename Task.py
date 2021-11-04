# TDT4117 - Information Retrieval - Autumn 2020
# Assignment 3 - gensim
# Students: Marianne Hernholm and Taheera Ahmed

# 1.0 Data loading and preprocessing
import random; random.seed(123)
import nltk
import string 
from string import punctuation
import codecs
import gensim
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.similarities import MatrixSimilarity
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk import RegexpTokenizer
nltk.download('punkt')

# 1.1
filename = "text.txt"
filecontent =  codecs.open(filename, "r", "utf-8").read()

# 1.2
# Convert new line symbols: win->unix
filecontent = filecontent.replace("\r\n", "\n")

# Partition into paragraphs
documents = []

for d in filecontent.split("\n\n"):
    d.strip()
    documents.append(d)

# 1.3
# Remove headers and footers:
documents = [d for d in documents if "Gutenberg" not in d]

# 1.4
# Remove empty documents
documents = [d for d in documents if len(d)>0]

# 1.5 og 1.6 Remember to remove from the text punctuation
# Clean and split into words
remove_punctuation = lambda d: "".join([ (c if c not in string.punctuation+"\n\r\t" else " ") for c in d])
stemmer = PorterStemmer()
tokenize = lambda d: [stemmer.stem(w.lower()) for w in d.split(" ") if len(w)>0]
texts = [tokenize(remove_punctuation(d)) for d in documents]

# 2. Dictionary building
# 2.1 Build Dictionary
dictionary = gensim.corpora.Dictionary(texts)

# 2.2 Filter out stopwords
stopwords_text = "stopwords.txt"
stoplist =  codecs.open(stopwords_text, "r", "utf-8").read().split(",")

stop_id = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
dictionary.filter_tokens(stop_id)

# 2.3 Map paragraphs into Bags-of-Words using the dictionary
corpus = []
for paragraph in texts: 
    corpus.append(dictionary.doc2bow(paragraph))

# 3. Retrieval Models
# 3.1 Build TF-IDF model using corpus 
tfidf_model = gensim.models.TfidfModel(corpus)

# 3.2 Map Bags-of-Words into TF-IDF weights 
tfidf_corpus = tfidf_model[corpus]

# 3.3 Construct MatrixSimilarity object that let us calculate similarities between paragraphs and queries
index =  gensim.similarities.MatrixSimilarity(corpus, dictionary)

# 3.4 Repeat procedure for LSI model
lsi_model = gensim.models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=100)
lsi_corpus = lsi_model[corpus]
lsi_similarity = gensim.similarities.MatrixSimilarity(lsi_corpus)

# 3.5  Report and try to interpret first 3 LSI topics.
lsi_model.show_topics()

# 4. Querying
# 4.1 
query = "What is the function of money?"

# Removing stopwords from query and creating query_list
query_list = []
for q in query.split(" "):
    query_list.append(q)
    for stopword in stoplist:  

        if q == stopword:
            query_list.remove(q)

# Stemming query (after stopwords have been removed)
# Unfortunately, we were not able to apply the stemmer to the query
for q in query_list: 
    stemmer = PorterStemmer()
    stemmer.stem(q)

query = dictionary.doc2bow(query_list) 

print(query_list)
# print(convertQuery(query))
        


