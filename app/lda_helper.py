from gensim import corpora, models
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("punkt")
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def run_lda(texts, num_topics=3):
    tokenized = [
        [w for w in word_tokenize(t.lower()) if w.isalpha() and w not in stop_words]
        for t in texts
    ]
    dictionary = corpora.Dictionary(tokenized)
    corpus = [dictionary.doc2bow(text) for text in tokenized]
    lda = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=10)
    return lda.print_topics()
