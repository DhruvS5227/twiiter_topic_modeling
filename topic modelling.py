#!/usr/bin/env python
# coding: utf-8

# In[1]:

import tweepy
import pandas as pd
import re
import spacy


# In[2]:

#your user credentials
consumer_key = "#consumer_key"
consumer_secret = "#consumer_secret"
access_token = "#access_token"
access_token_secret = "#access_token_secret"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit = True)


# In[3]:

#create a dataframe
df=[]
posts= tweepy.Cursor(api.user_timeline,id="#username",tweet_mode="extended").items(n) # n= number of tweets to be extracted
for i in posts:
    df.append([i.full_text,i.favorite_count,i.created_at])

df=pd.DataFrame(df,columns=['tweets','likes','time'])
df


# In[4]:

#cleaning of the extrected data
import string
def clean_text(text):
    ''' , and '''
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text
  
tweets_df_clean = pd.DataFrame(df.tweets.apply(lambda x: clean_text(x)))
tweets_df_clean


# In[5]:

#import spacy.load
nlp = spacy.load("en_core_web_lg")


# In[6]:

#lemmatization of data
def lemmatizer(text):        
    sent = []
    doc = nlp(text)
    for word in doc:
        sent.append(word.lemma_)
    return " ".join(sent)
tweets_df_clean = pd.DataFrame(tweets_df_clean.tweets.apply(lambda x: lemmatizer(x)))
tweets_df_clean['tweets'] = tweets_df_clean['tweets'].str.replace('-PRON-', '')
tweets_df_clean['tweets']


# In[7]:
#distribution od tweets chatracter length

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.figure(figsize=(10,6))
doc_lens = [len(d) for d in tweets_df_clean.tweets]
plt.hist(doc_lens, bins = 100)
plt.title('Distribution of Tweets character length')
plt.ylabel('Number of Tweets')
plt.xlabel('Tweets character length')
sns.despine();


# In[8]:
#wordcloud 

import matplotlib as mpl
from subprocess import check_output
import wordcloud
from wordcloud import WordCloud, STOPWORDS
mpl.rcParams['figure.figsize']=(12.0,12.0)  
mpl.rcParams['font.size']=12            
mpl.rcParams['savefig.dpi']=100             
mpl.rcParams['figure.subplot.bottom']=.1 
stopwords = set(STOPWORDS)
wordcloud = WordCloud(
                          background_color='white',
                          stopwords=stopwords,
                          max_words=500,
                          max_font_size=40, 
                          random_state=100
                         ).generate(str(tweets_df_clean.tweets))
print(wordcloud)
fig = plt.figure(1)
plt.imshow(wordcloud)
plt.axis('off')
plt.show();


# In[9]:
#unigram of tweets

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words='english').fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]

common_words = get_top_n_words(tweets_df_clean.tweets, 10)
unigram = pd.DataFrame(common_words, columns = ['unigram' , 'count'])
unigram



# topic modelling


# In[10]:


from sklearn.decomposition import LatentDirichletAllocation
vectorizer = CountVectorizer(
analyzer='word',       
min_df=3,
stop_words='english',
lowercase=True,
token_pattern='[a-zA-Z0-9]{3,}',
max_features=5000,
                            )
data_matrix = vectorizer.fit_transform(tweets_df_clean.tweets)
data_matrix


# In[12]:


lda_model = LatentDirichletAllocation(
n_components=10, # Number of topics
learning_method='online',
random_state=20,       
n_jobs = -1  # Use all available CPUs
                                     )
lda_output = lda_model.fit_transform(data_matrix)


# In[13]:
# top 10 words from the tweets

for i,topic in enumerate(lda_model.components_):
    print(f'Top 10 words for topic #{i}:')
    print([vectorizer.get_feature_names()[i] for i in topic.argsort()[-10:]])
    print('\n')


# In[14]:
#applying LDA

import pyLDAvis
import pyLDAvis.sklearn
pyLDAvis.enable_notebook()
pyLDAvis.sklearn.prepare(lda_model, data_matrix, vectorizer, mds='tsne')





