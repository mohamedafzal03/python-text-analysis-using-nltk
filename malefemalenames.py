from nltk.corpus import names
import nltk
print("haha")
#nltk.download('Names')
def gender_features(word):
    return {'last_letter': word[-1]}
labeled_names = ([(name, 'male') for name in names.words('male.txt')] + [(name, 'female') for name in names.words('female.txt')])
print(labeled_names)
import random
random.shuffle(labeled_names)
print(labeled_names)

featuresets = [(gender_features(n), gender) for (n, gender) in labeled_names]
print(featuresets)
train_set, test_set = featuresets[800:], featuresets[:200]
classifier = nltk.NaiveBayesClassifier.train(train_set)
print(classifier.classify(gender_features('kumar')))
print(nltk.classify.accuracy(classifier, test_set))