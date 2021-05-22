import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score,recall_score, f1_score

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, Dropout, LSTM, Bidirectional
import itertools

#load in data
data = pd.read_csv('all_data.txt', header=None, keep_default_na=False)
data = data.to_numpy()
X = data[:,0].astype('str')
y = data[:,1].astype('int')

#split data into training and test
X_train, X_test, y_train, y_test = train_test_split(X,y)

#pre hyperparameters
# file = 'glove.6B.300d.txt'
# def get_coefs(word,*arr): #from huai99
#     return word, np.asarray(arr, dtype='float32')
# embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(file))

# all_embs = np.stack(embeddings_index.values())
# emb_mean,emb_std = all_embs.mean(), all_embs.std()
# embed_size = all_embs.shape[1]

embed_size = 100 # how big is each word vector
max_feature = 50000 # how many unique words to use (i.e num rows in embedding vector)
max_len = 2000 # max number of words in a sentence to use
drop_val = 0.2
num_epochs = 30
max_words = 40000

# embed_size = 300 # how big is each word vector
# max_feature = 50000 # how many unique words to use (i.e num rows in embedding vector)
# max_len = 2000 # max number of words in a sentence to use
# drop_val = 0.1
# num_epochs = 30
# max_words = 40000

# for i in X_train:
#     cur_len = len(i)
#     if cur_len > max_len:
#         max_len = cur_len

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

word_idx = tokenizer.word_index
tot_words = len(word_idx)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_seq = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)


#Dense Architecture
model = Sequential()
model.add(Embedding(input_dim=max_feature, output_dim=embed_size, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(24, activation='relu'))
model.add(Dropout(drop_val))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam' ,metrics=['accuracy'])
'''use ‘binary_crossentropy’ as a loss function because of binary output, 
‘adam’ as an optimiser which makes use of momentum to avoid local minima and 
‘accuracy’ as a measure of model performance.'''

model.summary()

# early_stop = EarlyStopping(monitor='val_loss', patience=3)
#history = model.fit(X_train_pad, y_train, batch_size=512, epochs=num_epochs, validation_data=(X_test_pad, y_test), callbacks =[early_stop], verbose=2)
history = model.fit(X_train_pad, y_train, epochs=num_epochs, validation_data=(X_test_pad, y_test))
results  = model.evaluate(X_test_pad, y_test)

#plot
fig1 = plt.figure(figsize=(10, 10))
plt.plot(history.history['accuracy'], color = 'darkblue')
plt.plot(history.history['val_accuracy'], color = 'sienna')
#plt.title('Model Accuracy')
plt.ylabel('accuracy', fontsize=18)
plt.xlabel('epoch', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(['train', 'test'], loc='best', fontsize=18)
plt.show()
fig1.savefig('DL_ModelAccuracy.png')

fig2 = plt.figure(figsize=(10,10))
plt.plot(history.history['loss'], color = 'darkblue')
plt.plot(history.history['val_loss'], color = 'sienna')
#plt.title('Model Loss')
plt.ylabel('loss', fontsize=18)
plt.xlabel('epoch', fontsize=18)
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
plt.legend(['train', 'test'], loc='best', fontsize=18)
plt.show()
fig2.savefig('DL_ModelLoss.png')

# predict

DL_predictions = [1.0 if i > 0.5 else 0.0 for i in model.predict(X_test_pad)]

title = 'Deep Learning'
cm = confusion_matrix(y_test,DL_predictions)
tn, fp, fn, tp = confusion_matrix(y_test,DL_predictions).ravel()
precision = precision_score(y_test, DL_predictions)
recall = recall_score(y_test, DL_predictions)
f1score = f1_score(y_test, DL_predictions)


print(title)
print("Precision: {:.2f}%".format(100 * precision))
print("Recall: {:.2f}%".format(100 * recall))
print("F1 Score: {:.2f}%".format(100 * f1score))

#graph confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title=title,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print('Confusion Matrix - %s with Normalization' %(title))
    else:
        print('Confusion Matrix - %s' %(title))

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title('Confusion Matrix - %s' %(title))
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

labels =['Non-Spam', 'Spam']
plot_confusion_matrix(cm, labels) 

def save(tn, fp, fn, tp, precision, recall, f1score, title):
    NB_save = {'Type': ['TN', 'FP', 'FN', 'TP', 'Precision', 'Recall', 'F1 Score'],
        'Value': [tn, fp, fn, tp, precision, recall, f1score]}
    NB_df = pd.DataFrame(NB_save, columns=['Type','Value'])
    NB_df.to_csv(title + 'results.txt', index=False)

save(tn, fp, fn, tp, precision, recall, f1score, title)