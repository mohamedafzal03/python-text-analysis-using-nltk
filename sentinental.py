import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names
import pandas as pd

def word_feats(words):
    #print ("haha",dict([(word, True) for word in words]))
    return dict([(word, True) for word in words])

#positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)']
#negative_vocab = ['bad', 'terrible', 'useless', 'hate', ':(']
neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know', 'words', 'not']


#input positive
df=pd.read_csv('positive.csv',header=None)
df = df[0].tolist()
'''
df=df.values.tolist()
print(df)
df=str(df)
df=df.replace("[", "")
df=df.replace("]", "")
df=df.replace("'", "")
df=df.replace(",", "")
print(df)
'''
positive_vocab=df[:30]
#positive_vocab=positive_vocab.split()
print("positive csv",positive_vocab)
print("positive csv",type(positive_vocab))




#input negative

df=pd.read_csv('negative.csv',header=None)
df = df[0].tolist()

'''
df=df.values.tolist()
df=str(df)
df=df.replace("[", "")
df=df.replace("]", "")
df=df.replace("'", "")
df=df.replace(",", "")
'''
negative_vocab=df[:30]
print(negative_vocab)
#negative_vocab=negative_vocab.split()
print("negative csv",negative_vocab)
print("negative csv",type(negative_vocab))
#print("negative csv",negative_vocab)
#df=df.values.tolist()


#we are training these words awesome outstanding as postive words
positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
print("afzal:",positive_features)
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]
print("afzal:",negative_features)

train_set = negative_features + positive_features + neutral_features
#print("##############################",train_set)
classifier = NaiveBayesClassifier.train(train_set)
print(classifier)
# Predict
neg = 0
pos = 0
sentence = "abnormal terrible abominable"
sentence = sentence.lower()
words = sentence.split(' ')
#print(words)
for word in words:
    classResult = classifier.classify(word_feats(word))
    print(classResult)
    if classResult == 'pos':
        pos = pos + 1
    if classResult == 'neg':
        neg = neg + 1


print('Positive: ' + str(float(pos) / len(words)))
print('Negative: ' + str(float(neg) / len(words)))
print(nltk.classify.accuracy(classifier, train_set))