import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# nltk resources
nltk.download('stopwords')

# Load the imdb.vocab file
vocab_path = 'C:/Users/52748/Desktop/Final/imdb.vocab'
with open(vocab_path, 'r', encoding='utf-8') as vocab_file:
    vocab = vocab_file.read().splitlines()

# Paths to the dataset folders
train_pos_path = 'C:/Users/52748/Desktop/Final/aclImdb/train/pos'  # Update this path if necessary
train_neg_path = 'C:/Users/52748/Desktop/Final/aclImdb/train/neg'  # Update this path if necessary
test_pos_path = 'C:/Users/52748/Desktop/Final/aclImdb/test/pos'    # Update this path if necessary
test_neg_path = 'C:/Users/52748/Desktop/Final/aclImdb/test/neg'    # Update this path if necessary

# Function to load the dataset from files into a pandas dataframe
def load_dataset(pos_path, neg_path):
    reviews = []
    sentiments = []

    # Positive reviews
    for filename in os.listdir(pos_path):
        if filename.endswith('.txt'):
            with open(os.path.join(pos_path, filename), 'r', encoding='utf-8') as file:
                reviews.append(file.read())
                sentiments.append(1)  # Positive sentiment

    # Negative reviews
    for filename in os.listdir(neg_path):
        if filename.endswith('.txt'):
            with open(os.path.join(neg_path, filename), 'r', encoding='utf-8') as file:
                reviews.append(file.read())
                sentiments.append(0)  # Negative sentiment

    return pd.DataFrame({'review': reviews, 'sentiment': sentiments})

# Preprocessing function
def preprocess_text(text):
    print("Original Text: ", text[:3])  # Print the first 100 characters of the original text

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()
    print("Text after removing HTML tags: ", text[:3])  # Print the first 100 characters after removing HTML

    # Remove non-letters
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    print("Text after removing non-letters: ", letters_only[:3])  # Print the first 100 characters after removing non-letters

    # Convert to lower case, split into individual words
    words = letters_only.lower().split()
    print("Words after lowercasing and splitting: ", words[:3])  # Print the first 10 words

    # Convert the stop words to a set
    stops = set(stopwords.words("english"))

    # Remove stop words
    meaningful_words = [w for w in words if not w in stops]
    print("Words after removing stop words: ", meaningful_words[:3])  # Print the first 10 words after removing stop words

    # Join the words back into one string separated by space, and return the result.
    final_text = " ".join(meaningful_words)
    print("Final preprocessed text: ", final_text[:3])  # Print the first 100 characters of the final preprocessed text
    return final_text

# Load and preprocess the dataset
train_data = load_dataset(train_pos_path, train_neg_path)
test_data = load_dataset(test_pos_path, test_neg_path)

train_data['review'] = train_data['review'].apply(preprocess_text)
test_data['review'] = test_data['review'].apply(preprocess_text)

# Vectorizing the text data using the vocabulary from imdb.vocab
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                             stop_words=None, max_features=5000, vocabulary=vocab)
train_data_features = vectorizer.fit_transform(train_data['review'])
test_data_features = vectorizer.transform(test_data['review'])

# Train a logistic regression model on the training data
logistic_regression_model = LogisticRegression(max_iter=1000, solver='liblinear')
logistic_regression_model.fit(train_data_features, train_data['sentiment'])

# Predict sentiment on the test set
y_pred_test = logistic_regression_model.predict(test_data_features)

# Calculate the accuracy on the test set
test_accuracy = accuracy_score(test_data['sentiment'], y_pred_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Print Confusion Matrix
conf_matrix = confusion_matrix(test_data['sentiment'], y_pred_test)
print("Confusion Matrix:\n", conf_matrix)

# Print Classification Report
class_report = classification_report(test_data['sentiment'], y_pred_test)
print("Classification Report:\n", class_report)

# Calculate and Print ROC AUC
y_pred_proba = logistic_regression_model.predict_proba(test_data_features)[:, 1]
roc_auc = roc_auc_score(test_data['sentiment'], y_pred_proba)
print("ROC AUC Score:", roc_auc)

# Plot ROC Curve
fpr, tpr, _ = roc_curve(test_data['sentiment'], y_pred_proba)
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# The variables train_data_features and test_data_features are now ready to be used for model training and validation
print("Data preprocessing and sentiment analysis are complete.")


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense



# Vectorizing the text data using the vocabulary from imdb.vocab
vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None,
                             stop_words=None, max_features=5000, vocabulary=vocab)
train_data_features = vectorizer.fit_transform(train_data['review'])
test_data_features = vectorizer.transform(test_data['review'])

# 将训练数据分割为训练集和验证集
X_train, X_validation, y_train, y_validation = train_test_split(
    train_data_features, train_data['sentiment'], test_size=0.2, random_state=42)

# 参数设定
max_features = 5000   # 词汇表大小
maxlen = 500          # 序列的最大长度
embedding_size = 128  # 嵌入层的维度

# 数据准备
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_data['review'])
X_train_seq = tokenizer.texts_to_sequences(train_data['review'])
X_test_seq = tokenizer.texts_to_sequences(test_data['review'])

# 将序列填充到相同长度
X_train_pad = pad_sequences(X_train_seq, maxlen=maxlen)
X_test_pad = pad_sequences(X_test_seq, maxlen=maxlen)

# 构建模型
model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# 编译模型
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 训练模型
model.fit(X_train_pad, train_data['sentiment'], batch_size=32, epochs=3, validation_split=0.2)

# 评估模型
test_loss, test_accuracy = model.evaluate(X_test_pad, test_data['sentiment'])
print(f"Test Accuracy: {test_accuracy:.2f}")