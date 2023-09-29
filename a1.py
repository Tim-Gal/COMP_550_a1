import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def preprocess(line):
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
    words = ' '.join(words)

    return words


# open files facts.txt and fakes.txt
with open('facts.txt', 'r', encoding='utf-8') as facts_file:
    facts = facts_file.readlines()

with open('fakes.txt', 'r', encoding='utf-8') as fakes_file:
    fakes = fakes_file.readlines()


# preprocess features
facts = [preprocess(line) for line in facts]
fakes = [preprocess(line) for line in fakes]

# vectorize the features (turn words into numerical values)
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(facts + fakes).toarray()

# create labels
Y = [1] * len(facts) + [0] * len(fakes)

# split data set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

classifier = LogisticRegression()
classifier.fit(X_train, Y_train)

y_pred = classifier.predict(X_test)

accuracy = accuracy_score(Y_test, y_pred)

report = classification_report(Y_test, y_pred)

# print(f"Accuracy: {accuracy}")
# print(report)
