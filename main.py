import os
import re
import numpy as np
import pandas as pd
from pymystem3 import Mystem
import matplotlib.pyplot as plt
from numpy.linalg import svd
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split


def text_upload(path):
    text = ''
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            text = text + line.replace('...', '.').replace('..', '').strip() + ' '
    text = re.sub(r'[«»]', "'", text).strip()
    return text


def table_constructor(first_author, second_author):
    a1 = pd.DataFrame({'Binary': 0, 'Author': first_author, 'Title': os.listdir('./Books/' + first_author)})
    a2 = pd.DataFrame({'Binary': 1, 'Author': second_author, 'Title': os.listdir('./Books/' + second_author)})
    data = pd.concat([a1, a2], ignore_index=True)
    data['Text'] = [text_upload('./Books/' + author + '/' + title) for author, title in zip(data['Author'], data['Title'])]
    data = data.sample(frac=1).reset_index(drop=True)
    return data


def text_preparation(texts):
    result = []
    for text in texts:
        text = sent_tokenize(text, language='russian')
        text = ' sent '.join(text)
        result.append(' '.join(re.sub(r'[^\w\s-]', '', text.lower()).split()))
    result = ' stop '.join(result) + ' stop'
    return result


def text_analysis(texts):
    token_texts, part_texts = [], []
    token_text, part_text = [], []
    token_sent, part_sent = [], []
    tokens = Mystem().analyze(texts)
    tokens = list(filter(lambda t: t != {"text": " "} and t != {"text": "-"}, tokens))[:-1]
    for token in tokens:
        if token['text'] == 'sent':
            token_text.append(token_sent)
            part_text.append(part_sent)
            token_sent, part_sent = [], []
        else:
            if token['text'] == 'stop':
                token_text.append(token_sent)
                part_text.append(part_sent)
                token_texts.append(token_text)
                part_texts.append(part_text)
                token_sent, part_sent = [], []
                token_text, part_text = [], []
            else:
                try:
                    if token['analysis'][0]['lex'] not in stopwords.words("russian"):
                        token_sent.append(token['analysis'][0]['lex'])
                        part_sent.append(re.split(r'[,=]', token['analysis'][0]['gr'])[0])
                except IndexError:
                    token_sent.append(token['text'])
                    part_sent.append('DL')
                except KeyError:
                    pass
    return token_texts, part_texts


def text_statistic(words, parts, texts):
    data = pd.DataFrame()
    texts = [sent_tokenize(text, language='russian') for text in texts]
    data['Sentences'] = [len(text) for text in parts]
    data['Words'] = [sum(len(sent) for sent in text) for text in parts]
    data['Length sum'] = [sum(sum(len(word) for word in sent) for sent in text) for text in words]
    data['Noun'] = [sum(sent.count('S') for sent in text) / len(text) for text in parts]
    data['Verb'] = [sum(sent.count('V') for sent in text) / len(text) for text in parts]
    data['Adjective'] = [sum(sent.count('A') for sent in text) / len(text) for text in parts]
    data['Adverb'] = [sum(sent.count('ADV') for sent in text) / len(text) for text in parts]
    data['Average word count'] = data['Words'] / data['Sentences']
    data['Average word length'] = data['Length sum'] / data['Words']
    data['Comma'] = [text.count(',') for text in texts]
    data['QMark'] = [text.count('!') for text in texts]
    data['EMark'] = [text.count('?') for text in texts]
    return data.drop(['Sentences', 'Words', 'Length sum'], 1)


def class_definition(x_data, y_data, mp1, mp2):
    result = []
    points = x_data.loc[y_data[y_data == 0].index].values
    for p in points:
        if (p[0] - mp1[0]) / (mp2[0] - mp1[0]) - (p[1] - mp1[1]) / (mp2[1] - mp1[1]) > 0:
            result.append(0)
        else:
            result.append(1)
    if (result == y_data[y_data == 0]).mean() * 100 > 50:
        return [0, 1]
    else:
        return [1, 0]


def classification(x_data, classes, mp1, mp2):
    result = []
    points = x_data.values
    for p in points:
        if (p[0] - mp1[0]) / (mp2[0] - mp1[0]) - (p[1] - mp1[1]) / (mp2[1] - mp1[1]) > 0:
            result.append(classes[0])
        else:
            result.append(classes[1])
    return result


def near_point(point, data):
    dist = []
    for dot in data:
        dist.append(((point[0] - dot[0]) ** 2 + (point[1] - dot[1]) ** 2) ** (1 / 2))
    p = data[np.array(dist).argmin()]
    return p


def decisive_points(x_data, y_data):
    first = x_data.loc[y_data[y_data == 0].index].values
    second = x_data.loc[y_data[y_data == 1].index].values
    p1 = first[np.argmin([p[0] + p[1] for p in first])]
    np1 = near_point(p1, second)
    mp1 = (p1 + np1) / 2
    first = first[first != p1].reshape((-1, 2))
    p2 = second[np.argmax([p[0] + p[1] for p in second])]
    np2 = near_point(p2, first)
    mp2 = (p2 + np2) / 2

    first = x_data.loc[y_data[y_data == 0].index].values
    second = x_data.loc[y_data[y_data == 1].index].values
    p1 = first[np.argmin([p[1] for p in first])]
    np1 = near_point(p1, second)
    mp1_ = (p1 + np1) / 2
    first = first[first != p1].reshape((-1, 2))
    p2 = second[np.argmax([p[1] for p in second])]
    np2 = near_point(p2, first)
    mp2_ = (p2 + np2) / 2

    classes = class_definition(x_data, y_data, mp1, mp2)
    test = (classification(x_data, classes, mp1, mp2) == y_data).mean() * 100
    classes_ = class_definition(x_data, y_data, mp1_, mp2_)
    test_ = (classification(x_data, classes_, mp1_, mp2_) == y_data).mean() * 100

    if test > test_:
        return mp1, mp2, classes
    else:
        return mp1_, mp2_, classes_


def dependency_plot(x_data, y_data, mp1, mp2, x_data_=None, y_data_=None, step=1):
    f_class = x_data.loc[y_data[y_data == 0].index]
    s_class = x_data.loc[y_data[y_data == 1].index]

    plt.figure(figsize=(x_data.iloc[:, 0].max() + step, x_data.iloc[:, 1].max() + step))
    plt.xlim(x_data.iloc[:, 0].min() - step, x_data.iloc[:, 0].max() + step)
    plt.ylim(x_data.iloc[:, 1].min() - step, x_data.iloc[:, 1].max() + step)

    plt.scatter(x=f_class.iloc[:, 0], y=f_class.iloc[:, 1], color='red', s=20, label="Бунин")
    plt.scatter(x=s_class.iloc[:, 0], y=s_class.iloc[:, 1], color='blue', s=20, label="Тургненев")

    if x_data_ is not None and y_data_ is not None:
        f_class_ = x_data_.loc[y_data_[y_data_ == 0].index]
        s_class_ = x_data_.loc[y_data_[y_data_ == 1].index]
        plt.scatter(x=f_class_.iloc[:, 0], y=f_class_.iloc[:, 1], color='#FFD7D7', s=20)
        plt.scatter(x=s_class_.iloc[:, 0], y=s_class_.iloc[:, 1], color='#D7DBFF', s=20)

    plt.scatter(mp1[0], mp1[1], color='#D4FF00', s=40)
    plt.scatter(mp2[0], mp2[1], color='#D4FF00', s=40)

    x = np.linspace(x_data.iloc[:, 0].min() - step, x_data.iloc[:, 0].max() + step, 10000)
    y = (x - mp1[0]) / (mp2[0] - mp1[0]) * (mp2[1] - mp1[1]) + mp1[1]
    plt.plot(x, y, marker='o', markersize=0.00001, color='green')
    plt.legend()
    plt.show()


table = table_constructor('Бунин', 'Тургенев')
raw_text = text_preparation(table['Text'].to_list())
lem_text, lem_part = text_analysis(raw_text)

# X = pd.concat(text_statistic(lem_text, lem_part, table['Text'].to_list()), axis=1)

#X = X.drop(['Sentences', 'Words', 'Length sum', 'Dot[2]', 'Comma[2]',
#            'QMark[2]', 'EMark[2]', 'Sentences[2]', 'Words[2]', 'Length sum[2]',
#            'Average word length[2]', 'Average word count[2]'], 1)

y = table['Binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# without PCA

col1 = 'Average word length'
col2 = 'Average word count'

X_train_ = X_train[[col1, col2]]
X_test_ = X_test[[col1, col2]]

mp1, mp2, classes = decisive_points(X_train_, y_train)
result = classification(X_test_, classes, mp1, mp2)
check = classification(X_train_, classes, mp1, mp2)

dependency_plot(X_train_, y_train, mp1, mp2, X_test_, y_test)

# with PCA [SVD]

X_train_pca = X_train - X_train.mean(axis=0)

u, s, vt = svd(X_train_pca, full_matrices=0)
Z_train = X_train_pca.dot(vt[:2].T)

mp1_, mp2_, classes_ = decisive_points(Z_train, y_train)

X_test_pca = X_test - X_train.mean(axis=0)
Z_test = X_test_pca.dot(vt[:2].T)

result_ = classification(Z_test, classes_, mp1_, mp2_)
check_ = classification(Z_train, classes_, mp1_, mp2_)

dependency_plot(Z_train, y_train, mp1_, mp2_, Z_test, y_test, step=10)

# Results
print("- - - - - - - - - - - - - - - ")
print('Without PCA: ', (result == y_test.values).mean() * 100)
print('Without PCA (train data):', (check == y_train.values).mean() * 100)
print("\n")
print('With PCA:    ', (result_ == y_test.values).mean() * 100)
print('With PCA (train data):    ', (check_ == y_train.values).mean() * 100)
print("- - - - - - - - - - - - - - - ")

# wp.append((result == y_test.values).mean() * 100))
# wpt.append((check == y_train.values).mean() * 100))
# p.append((result_ == y_test.values).mean() * 100))
# pt.append((check_ == y_train.values).mean() * 100))
