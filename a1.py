import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


def clean(line):
    """
    make lowercase -> tokenize -> remove punctuation and special characters -> remove stopwords -> stem -> reassemble
    :param line: line of text
    :return: cleaned line of text
    """

    # make words lowercase
    line = line.lower()

    # tokenize words
    words = nltk.word_tokenize(line)

    # remove punctuation and special characters
    words = [word for word in words if word.isalnum()]

    # remove stopwords
    stopwords = nltk.corpus.stopwords.words('english')
    words = [word for word in words if word not in stopwords]

    # stem words
    stemmer = nltk.stem.PorterStemmer()
    words = [stemmer.stem(word) for word in words]

    # piece tokens back into lines
    line = ' '.join(words)

    return line


def vectorize(facts, fakes):
    """
    vectorizes features into a matrix and labels into a list
    :param facts: list of cleaned texts containing cat facts
    :param fakes: list of cleaned texts containing cat fakes
    :return: matrix of features and list of labels
    """

    # vectorize the features (turn words into numerical values)
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(facts + fakes).toarray()

    # create labels
    Y = [1] * len(facts) + [0] * len(fakes)

    return X, Y


def split_data(X, Y, test_size):
    """
    splits data set into training and testing parts
    :param X: matrix of features
    :param Y: list of labels
    :param test_size: proportion of data set to be the testing data set
    :return: X_train, X_test, Y_train, Y_test
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X_train, X_test, Y_train, Y_test


def train_MultinomialNB(X_train, Y_train):
    classifier = MultinomialNB()
    classifier.fit(X_train, Y_train)
    return classifier


def train_LR(X_train, Y_train):
    classifier = LogisticRegression()
    classifier.fit(X_train, Y_train)
    return classifier


def train_SVM(X_train, Y_train):
    classifier = SVC(kernel='linear', C=1.0)
    classifier.fit(X_train, Y_train)
    return classifier


with open('facts.txt', 'r', encoding='utf-8') as facts_file:
    facts = facts_file.readlines()

with open('fakes.txt', 'r', encoding='utf-8') as fakes_file:
    fakes = fakes_file.readlines()


facts = [clean(line) for line in facts]
fakes = [clean(line) for line in fakes]

X, Y = vectorize(facts, fakes)

X_train, X_test, Y_train, Y_test = split_data(X, Y, 0.1)


MultinomialNB_classifier = train_MultinomialNB(X_train, Y_train)
MultinomialNB_y_predict = MultinomialNB_classifier.predict(X_test)
MultinomialNB_accuracy = accuracy_score(Y_test, MultinomialNB_y_predict)
MultinomialNB_report = classification_report(Y_test, MultinomialNB_y_predict)

LR_classifier = train_LR(X_train, Y_train)
LR_y_predict = LR_classifier.predict(X_test)
LR_accuracy = accuracy_score(Y_test, LR_y_predict)
LR_report = classification_report(Y_test, LR_y_predict)

SVM_classifier = train_SVM(X_train, Y_train)
SVM_y_predict = SVM_classifier.predict(X_test)
SVM_accuracy = accuracy_score(Y_test, SVM_y_predict)
SVM_report = classification_report(Y_test, SVM_y_predict)


print(f"MultinomialNB_Accuracy: {MultinomialNB_accuracy}")
print(MultinomialNB_report)

print(f"LR_Accuracy: {LR_accuracy}")
print(LR_report)

print(f"SVM_Accuracy: {SVM_accuracy}")
print(SVM_report)
