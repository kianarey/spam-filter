import numpy as np
import glob
from sklearn.model_selection import train_test_split
import email
import re
import nltk
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import pandas as pd

# load data from Spam Assassin
easy_ham = glob.glob('easy_ham/*') 
'''easy_ham: 
   2500 non-spam messages. Typically easy to differentiate from spam.
   They frequently do not contain any spammish signatures (like HTML etc).'''

easy_ham_2 = glob.glob('easy_ham_2/*')
'''1400 non-spam messages.  A more recent addition to the set.'''

hard_ham = glob.glob('hard_ham/*')
'''250 non-spam messages which are closer in many respects to typical spam.
   Ex) use of HTML, unusual HTML markup, coloured text, "spammish-sounding" 
   phrases etc.'''

spam = glob.glob('spam/*')
'''500 spam messages, all received from non-spam-trap sources.'''

spam_2 = glob.glob('spam_2/*')
'''1397 spam messages.  Again, more recent.'''

'''combined data into two for easier handling'''
non_spam_email = [easy_ham, easy_ham_2, hard_ham]
spam_email = [spam, spam_2] 


'''Definitions'''
def split_data(data):
    train_data = np.array([])
    test_data = np.array([])
    for i in data:
        train_data = np.concatenate((train_data,i[0]), axis=0)
        test_data = np.concatenate((test_data,i[1]), axis=0)
        
    return train_data, test_data

def shuffle_and_get_data(data):
    shuffle_data_idx = np.random.randint(low=0, high=len(data)-1, size=len(data))
    ret_data = data[shuffle_data_idx]
    return ret_data

def get_email_content(data): #from huai99
    email_content = []
    for i in data:
        file = open(i, encoding = 'latin1')
        try:
            message = email.message_from_file(file)
            for part in message.walk():
                if part.get_content_type() == 'text/plain':
                    email_content.append(part.get_payload())
                
        except Exception as e:
            print(e)
    return email_content

def clean_up(content): #borrowed from huai99, but altered
    content = re.sub(r"http\S+","",content) #remove hyperlink
    content = content.lower() #change to lower case
    content = re.sub(r'\d+'," ", content) #remove number
    content = re.sub(r"[^a-zA-Z0-9]"," ", content) # remove all characters that are not alphanumeric 
    content = re.sub(r"\b[a-zA-Z]\b", " ", content) # remove all single letters
    content = content.replace('\n','')
    
    return content

def remove_stopwords(content):
    '''['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 
    "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 
    'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', 
    "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 
    'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 
    'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do',
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 
    'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 
    'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 
    'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 
    'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 
    'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 
    'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn',
    "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', 
    "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 
    'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 
    'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]'''
    sentence = [word for word in content if not word in stopwords.words('english')]
    return sentence

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def do_lemmatization(content): 
    sentence = [lemmatizer.lemmatize(i) for i in content]
    
    return sentence


#split data into training and test subsets
ns_data = np.asarray([train_test_split(i) for i in non_spam_email])
s_data = np.asarray([train_test_split(i) for i in spam_email])

ns_train, ns_test = split_data(ns_data)
s_train, s_test = split_data(s_data)

ns_train = ns_train.reshape(len(ns_train),)
ns_test = ns_test.reshape(len(ns_test),)
s_train = s_train.reshape(len(s_train),)
s_test = s_test.reshape(len(s_test),)

# #get email content
ns_train_emails = get_email_content(ns_train)
ns_test_emails = get_email_content(ns_test)
s_train_emails = get_email_content(s_train)
s_test_emails = get_email_content(s_test)

#clean up email content and tokenize - round 1
ns_train_emails_cleaned = [clean_up(i) for i in ns_train_emails]
ns_test_emails_cleaned = [clean_up(i) for i in ns_test_emails]
s_train_emails_cleaned = [clean_up(i) for i in s_train_emails]
s_test_emails_cleaned = [clean_up(i) for i in s_test_emails]

ns_train_emails_cleaned = [i.split() for i in ns_train_emails_cleaned]
ns_test_emails_cleaned = [i.split() for i in ns_test_emails_cleaned]
s_train_emails_cleaned = [i.split() for i in s_train_emails_cleaned]
s_test_emails_cleaned = [i.split() for i in s_test_emails_cleaned]

#clean up email - round 2 - remove stop words and lemmatize
'''Lemmatization = It looks beyond word reduction and considers a language’s 
full vocabulary to apply a morphological analysis to words, aiming to remove 
inflectional endings only and to return the base or dictionary form of a word, 
which is known as the lemma.'''

ns_train_emails_cleaned = [remove_stopwords(i) for i in ns_train_emails_cleaned]
ns_test_emails_cleaned = [remove_stopwords(i) for i in ns_test_emails_cleaned]
s_train_emails_cleaned = [remove_stopwords(i) for i in s_train_emails_cleaned]
s_test_emails_cleaned = [remove_stopwords(i) for i in s_test_emails_cleaned]

ns_train_emails_cleaned = [do_lemmatization(i) for i in ns_train_emails_cleaned]
ns_test_emails_cleaned = [do_lemmatization(i) for i in ns_test_emails_cleaned]
s_train_emails_cleaned = [do_lemmatization(i) for i in s_train_emails_cleaned]
s_test_emails_cleaned = [do_lemmatization(i) for i in s_test_emails_cleaned]


ns_train_emails_cleaned = [" ".join(o) for o in ns_train_emails_cleaned]
ns_test_emails_cleaned = [" ".join(o) for o in ns_test_emails_cleaned]
s_train_emails_cleaned = [" ".join(o) for o in s_train_emails_cleaned]
s_test_emails_cleaned = [" ".join(o) for o in s_test_emails_cleaned]

#for visualization - bargraph
ns_train_emails_cleaned = np.asarray(ns_train_emails_cleaned).reshape(len(ns_train_emails_cleaned),1)
ns_test_emails_cleaned = np.asarray(ns_test_emails_cleaned).reshape(len(ns_test_emails_cleaned),1)
s_train_emails_cleaned = np.asarray(s_train_emails_cleaned).reshape(len(s_train_emails_cleaned),1)
s_test_emails_cleaned = np.asarray(s_test_emails_cleaned).reshape(len(s_test_emails_cleaned),1)

ns = np.concatenate((ns_train_emails_cleaned, ns_test_emails_cleaned))
s = np.concatenate((s_train_emails_cleaned, s_test_emails_cleaned))

ns_df = pd.DataFrame(ns)
ns_df.to_csv('ns_data.txt', header=False, index=False)

s_df = pd.DataFrame(s)
s_df.to_csv('s_data.txt', header=False, index=False)





    

