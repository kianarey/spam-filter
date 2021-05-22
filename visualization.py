import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter


#load processed data
ns_df = pd.read_csv('ns_data.txt', header=None, keep_default_na=False)
s_df = pd.read_csv('s_data.txt',header=None, keep_default_na=False)

ns_df = pd.DataFrame.to_numpy(ns_df)
s_df = pd.DataFrame.to_numpy(s_df) 

ns_df = ns_df.ravel()
s_df = s_df.ravel()

ns_d = dict()
s_d = dict()


def get_dict_and_count(data):
    d = dict()
    for i in range(len(data)):
        sentence = data[i].split()
        #iterate over each word in line
        for word in sentence:
            #check if  word is already in dictionary
            if word in d:
                # increment count of word by 1
                d[word] = d[word] + 1
                #print(ns_d[word])
            else:
                # Add the word to dictionary with count 1
                d[word] = 1
    return d

top = 20
def most_common(dictionary):
    dict_counter = Counter(dictionary)
    
    #find top 20 most common
    most_common = dict_counter.most_common(top) #this is a list
    most_common_dict = dict(most_common) #transorm back to dict
    return most_common_dict
    
    
def visualize_barh(dictionary, name):
    words = list(dictionary.keys())
    values = list(dictionary.values())
      
    fig = plt.figure(figsize=(10, 8))
      
    # creating the bar plot
    plt.barh(words, values, color='sienna', align='center')
      
    plt.xlabel("No. of Occurrences", fontsize=15)
    plt.ylabel("Token", fontsize=15)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.title("Most Common Tokenized Word in %s Emails" %(name), fontsize=20)
    plt.show()
    return fig


ns_dict = get_dict_and_count(ns_df)
ns_mostcommon = most_common(ns_dict)
ns_fig = visualize_barh(ns_mostcommon, name='Non-Spam')
ns_fig.savefig('ns_mostcommon.png')

s_dict = get_dict_and_count(s_df)
s_mostcommon = most_common(s_dict)
s_fig = visualize_barh(s_mostcommon, name='Spam')
s_fig.savefig('s_mostcommon.png')



