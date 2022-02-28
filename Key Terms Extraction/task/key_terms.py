import nltk as nltk
from lxml import etree

def token_text(text):
    tokens = nltk.tokenize.word_tokenize(text.lower())
    dic = {t: tokens.count(t) for t in tokens}
    dic = {k: dic[k] for k in sorted(dic, key=dic.get, reverse=True)}
    cnts = sorted({dic[k] for k in dic}, reverse=True)
    lst = []
    for n in cnts:
        lst += sorted([k for k in dic if dic[k] == n], reverse=True)
        if len(lst) >= 5:
            break
    print(' '.join(lst[:5]), '\n')


def main():
    corpus = etree.parse('news.xml').getroot()
    news = corpus[0]
    for n in news:
        head = n[0].text
        print(f'{head}:')
        text = n[1].text
        token_text(text)


if __name__ == '__main__':
    main()