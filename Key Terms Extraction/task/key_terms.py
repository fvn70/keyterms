import string
import nltk as nltk
from lxml import etree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


def do_dic(voc, vec):
    # Create dictionary {term: tf-idf}
    # on base doc vocabulary and it's vector
    dic = {}
    for i in range(len(vec)):
        if vec[i] > 0:
            dic[voc[i]] = vec[i]
    return dic


def sort_by_tf_idf(text):
    vectorizer = TfidfVectorizer()
    vectorizer.fit(text)
    terms = vectorizer.get_feature_names_out()
    sort_text = []
    for doc in text:
        vector = vectorizer.transform([doc])
        vector = vector.toarray()[0]
        tf_dic = do_dic(terms, vector)
        cnts = sorted({tf_dic[k] for k in tf_dic}, reverse=True)
        lst = []
        for n in cnts:
            lst += sorted([k for k in tf_dic if tf_dic[k] == n], reverse=True)
            if len(lst) >= 5:
                break
        sort_text.append(' '.join(lst[:5]))
    return sort_text


def lemma_text(text):
    tizer = WordNetLemmatizer()
    stw = stopwords.words('english')
    pnkt = list(string.punctuation)
    lemmas = []
    for word in text:
        lemma = tizer.lemmatize(word)
        if not lemma in stw + pnkt and nltk.pos_tag([lemma])[0][1] == 'NN':
            lemmas.append(lemma)
    return lemmas


def init():
    nltk.download('wordnet')
    nltk.download('stopwords')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger')


def main():
    # init()
    corpus = etree.parse('news.xml').getroot()
    news = corpus[0]
    new_text = []
    heads = []
    for n in news:
        heads.append(n[0].text)
        doc = n[1].text
        tokens = nltk.tokenize.word_tokenize(doc.lower())
        doc = lemma_text(tokens)
        new_text.append(' '.join(doc))
    news = sort_by_tf_idf(new_text)
    for i in range(len(heads)):
        print(f'{heads[i]}:')
        print(f'{news[i]}\n')


if __name__ == '__main__':
    main()