import os,re,string,unicodedata
from langdetect import detect
import spacy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import OrderedDict
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.metrics import classification_report,accuracy_score

# nlp = spacy.load('en_web_core', parse=True, tag=True, entity=True)
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')

data_path = 'jigsaw-multilingual-toxic-comment-classification'
os.listdir(data_path)

train_df = pd.read_csv(os.path.join(data_path,'jigsaw-toxic-comment-train.csv'))
val_df = pd.read_csv(os.path.join(data_path,'validation.csv'))
test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))


def display_scores(clf, x_train, x_test, y_train, y_test):
    print("In Sample Data:")
    train_preds = clf.predict(x_train)
    print(classification_report(y_true=y_train, y_pred=train_preds))
    print(accuracy_score(y_true=y_train, y_pred=train_preds))

    print("Out of Sample Data:")
    preds = clf.predict(x_test)
    print(classification_report(y_true=y_test, y_pred=preds))
    print(accuracy_score(y_true=y_test, y_pred=preds))

    return

def clean_text(text):

    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\\n',' ',text)
    text = re.sub("\[\[User.*",'',text)
    text = re.sub('\w*\d\w*', '', text)
    text = text.strip()
    return text


def additional_tags(text):
    text = re.sub('[‘’_“”…]', '', text)
    text = re.sub("\(http://.*?\s\(http://.*\)",'',text)
    text = re.sub('\n', '', text)
    text = re.sub('\t', '', text)
    text = re.sub("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}",'',text)
    text = re.sub(r'—+— ','',text)
    text = re.sub(r'\d+', '', text)
    return text

def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def removeDup(string):
    return ''.join(OrderedDict.fromkeys(string))


def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text



def simple_stemmer(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

list_classes = ['toxic']
y = train_df[list_classes].values

common_text_train = train_df['comment_text']
content_txt_test = test_df['content']

cleaned_text = [clean_text(x) for x in common_text_train]
X = [additional_tags(x) for x in cleaned_text]
cleaned_data = [remove_accented_chars(i) for i in X]
str_def = [removeDup(i) for i in cleaned_data]
stem_words = [simple_stemmer(i) for i in str_def]
words = [remove_stopwords(i) for i in stem_words]


# tfidf
idf = TfidfVectorizer(use_idf=True)
word_vocab = idf.fit(words)
word_corpus = idf.transform(words)
# word_corpus.size

# train_test_split
x_train,x_test,y_train,y_test = train_test_split(word_corpus,y, test_size=0.2, random_state=36)

# classification algorithm
clf = RandomForestClassifier(max_depth=2, random_state=0,n_estimators=100)

clf.fit(x_train,np.ravel(y_train))
pred = clf.predict(word_corpus)

display_scores(clf,x_train,x_test,y_train,y_test)



cleaned_text_t = [clean_text(x) for x in content_txt_test]
X_t = [additional_tags(x) for x in cleaned_text_t]
cleaned_data_t = [remove_accented_chars(i) for i in X_t]
str_def_t = [removeDup(i) for i in cleaned_data_t]
stem_words_t = [simple_stemmer(i) for i in str_def_t]
words_t = [remove_stopwords(i) for i in stem_words_t]

word_corpus_t = idf.transform(words_t)
test_preds = clf.predict(word_corpus_t)


submission_df = pd.DataFrame(columns=['id', 'prediction'])
submission_df['id'] = submission_df.index
submission_df['prediction'] = test_preds
submission_df.to_csv(os.path.join(data_path,'submission.csv'), index=False)




# In Sample Data:
#              precision    recall  f1-score   support
#            0       0.90      1.00      0.95    161802
#            1       0.00      0.00      0.00     17037
#     accuracy                           0.90    178839
#    macro avg       0.45      0.50      0.47    178839
# weighted avg       0.82      0.90      0.86    178839
# 0.9047355442604801
# Out of Sample Data:
#               precision    recall  f1-score   support
#            0       0.90      1.00      0.95     40363
#            1       0.00      0.00      0.00      4347
#     accuracy                           0.90     44710
#    macro avg       0.45      0.50      0.47     44710
# weighted avg       0.81      0.90      0.86     44710
# 0.9027734287631403

