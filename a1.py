import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score, classification_report


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


def CV_HPT(classifier, param_grid, name):
    """
    perform cross validation and hyperparameter tuning using a GridSearchCV object and assess the accuracy of best result
    :param classifier: linear classifier model being tested
    :param param_grid: hyperparameters of classifier being tested
    :param name: the name of the classifier model
    :return: the optimal parameters found for that model
    """

    # create a GridSearchCV object
    grid_search = GridSearchCV(classifier, param_grid, cv=5, scoring=make_scorer(accuracy_score))

    # fit the GridSearchCV object to the data
    grid_search.fit(X, Y)

    best_classifier = grid_search.best_estimator_
    best_parameters = grid_search.best_params_
    best_score = grid_search.best_score_
    y_predict = cross_val_predict(best_classifier, X, Y, cv=5)

    print(f"Model: {name}")

    # print the best parameters found by Grid Search
    print(f"Best Parameters: {best_parameters}")

    # print the best accuracy using the best parameters
    print(f"Best accuracy: {best_score}")

    # print classification report with the best parameters
    print(f"Classification Report:")
    print(classification_report(Y, y_predict))

    return best_classifier


with open('facts.txt', 'r', encoding='utf-8') as facts_file:
    facts = facts_file.readlines()

with open('fakes.txt', 'r', encoding='utf-8') as fakes_file:
    fakes = fakes_file.readlines()


facts = [clean(line) for line in facts]
fakes = [clean(line) for line in fakes]

X, Y = vectorize(facts, fakes)


MNB_param_grid = {
    'alpha': [0.1, 0.5, 1.0]
}

LR_param_grid = {
    'C': [0.1, 1.0, 10.0],
    'max_iter': [100, 200, 300]
}

SVM_param_grid = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['linear', 'rbf', 'poly']
}

MNB_classifier = MultinomialNB()
LR_classifier = LogisticRegression()
SVC_classifier = SVC()

MNB_param = CV_HPT(MNB_classifier, MNB_param_grid, "Multinomial Naive Bayes")
LR_param = CV_HPT(LR_classifier, LR_param_grid, "Logistic Regression")
SVM_param = CV_HPT(SVC_classifier, SVM_param_grid, "Support Vector Machine")
