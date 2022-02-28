import string
import nltk as nltk
from lxml import etree
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords


def sort_text(text):
    dic = {t: text.count(t) for t in text}
    dic = {k: dic[k] for k in sorted(dic, key=dic.get, reverse=True)}
    cnts = sorted({dic[k] for k in dic}, reverse=True)
    lst = []
    for n in cnts:
        lst += sorted([k for k in dic if dic[k] == n], reverse=True)
        if len(lst) >= 5:
            break
    print(' '.join(lst[:5]), '\n')


def lemma_text(text):
    tizer = WordNetLemmatizer()
    stw = stopwords.words('english')
    pnkt = list(string.punctuation)
    lemmas = []
    for word in text:
        lemma = tizer.lemmatize(word)
        if not lemma in stw + pnkt:
            lemmas.append(lemma)
    return lemmas


def main():
    # nltk.download('wordnet')
    # nltk.download('stopwords')
    # nltk.download('omw-1.4')
    corpus = etree.parse('news.xml').getroot()
    news = corpus[0]
    for n in news:
        head = n[0].text
        print(f'{head}:')
        text = n[1].text
        tokens = nltk.tokenize.word_tokenize(text.lower())
        sort_text(lemma_text(tokens))


if __name__ == '__main__':
    main()