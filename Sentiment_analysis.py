#!/usr/bin/env python
# coding: utf-8

# In[29]:


# Data manipulation and handling
import pandas as pd
import numpy as np
from datetime import datetime
import string

# Data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Text preprocessing
import contractions # Expanding English language contractions
import re # Regular expression operations

# Language detection
from langdetect import detect
import langid

# Natural Language Processing (NLP) and sentiment analysis
import spacy
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import PorterStemmer # Word stemming

# Downloads for NLTK
nltk.download('vader_lexicon')  
nltk.download('stopwords')  
nltk.download('punkt')  # Tokenizing sentences

# Word cloud visualization
from wordcloud import WordCloud

# Suppress warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df_reviews = pd.read_csv("reviews_final.csv")
df_listings = pd.read_csv("listings_final.csv")


# In[3]:


df_reviews['listing_id'].isin(df_listings['id']).value_counts()


# In[4]:


df = pd.merge(df_reviews, df_listings[['id', 
                                       'review_scores_rating', 
                                       'host_is_superhost', 
                                       'room_type', 
                                       'neighbourhood_cleansed', 
                                       'host_identity_verified']], 
              left_on = 'listing_id', 
              right_on = 'id')


# In[5]:


# Calculate the total missing values
total_missing = df.isnull().sum()
total_missing


# In[6]:


# Calculate the percentage of missing values

missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
percent_missing = (missing_values / len(df)) * 100

missing_info = pd.DataFrame({
    'Missing Values': missing_values,
    '% of Total Values': percent_missing.round(2)
})

print(f"The selected dataframe has {df.shape[1]} columns.")
print(f"There are {missing_info.shape[0]} columns that have missing values.")
print(missing_info)


# In[7]:


df.dropna(inplace = True)


# In[8]:


# Convert the 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')

# Filter the dataset based on the desired duration
start_date = datetime.strptime('01/01/2020', '%d/%m/%Y')
end_date = datetime.strptime('31/12/2023', '%d/%m/%Y')
df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
df = df_filtered


# In[10]:


# Set the seed for reproducibility
random.seed(1)

# Create a new column 'language' to store the detected language
df['language'] = df['comments'].apply(lambda x: langid.classify(x)[0] if isinstance(x, str) and re.search('[a-zA-Z]', str(x)) and str(x).strip() != '' else 'unknown' if not pd.isna(x) else '')

# Display the DataFrame with the 'comments' and 'language' columns
print(df[['comments', 'language']])


# In[12]:


# Calculate the percentage of unique language

country_counts = df['language'].value_counts()
language_percent = (df['language'].value_counts() / df['language'].count()) * 100

language_counts = pd.DataFrame({
    'Language counts': country_counts,
    '% of Language counts': language_percent.round(2)
})

print(language_counts)


# In[13]:


df_review = df[df['language'] == 'en']


# In[14]:


reviews = df_review['comments'].to_list()
reviews[:3]


# In[15]:


def clean_data(text):
    expanded_words = []
    for word in text.split():
        expanded_words.append(contractions.fix(word))
    expanded_text = ' '.join(expanded_words)
    
    bad_symbols = '!"#%&\'*+,-<=>?[\\]^_`{|}~'
    text_without_symbols = expanded_text.translate(str.maketrans('', '', bad_symbols))
    text_without_punctuation = text_without_symbols.translate(str.maketrans('', '', string.punctuation))
    leave_letters_only = ' '.join(re.findall("[a-zA-Z]+", text_without_punctuation))
    return leave_letters_only


# In[16]:


clean_text = [clean_data(text) for text in reviews]


# In[17]:


df_review['comments'] = clean_text


# # Preprocessing review text

# In[18]:


import nltk
from nltk.corpus import stopwords  

nltk.download('stopwords')  


# In[19]:


# lower case the text
df_review['pre_review'] = df_review['comments'].str.lower()


# In[20]:


# tokenize the text
df_review['pre_review'] = df_review['pre_review'].apply(nltk.word_tokenize)


# In[21]:


# remove stop words
stop_words = set(stopwords.words('english'))
df_review['pre_review'] = df_review['pre_review'].apply(lambda x: [word for word in x if word not in stop_words])


# In[22]:


# stem the words
stemmer = PorterStemmer()
df_review['pre_review'] = df_review['pre_review'].apply(lambda x: [stemmer.stem(word) for word in x])


# # Sentiment Analysis with NLTK Library
# 

# In[23]:


# Initialize the sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# analyze sentiment in every review texts ('pre_review' column)
df_review['polarity'] = df_review['pre_review'].apply(lambda x: analyzer.polarity_scores(' '.join(x))['compound'])

# Apply sentiment analysis to each review text and calculate scores
df_review['sentiment_scores'] = df_review['pre_review'].apply(lambda x: analyzer.polarity_scores(' '.join(x)))

# Define function to map compound scores to sentiment labels
def vader_sentiment(score):
    if score >= 0.05:
        return 'Positive'
    elif score <= -0.05:
        return 'Negative'
    else:
        return 'Neutral'

# Apply the sentiment function to create the sentiment column
df_review['sentiment'] = df_review['polarity'].apply(vader_sentiment)


# In[24]:


# calculate mean of the review sentiments in each listing
df_review_avg = df_review.groupby('listing_id')['polarity'].mean().reset_index(name='avg_sentiment')


# In[25]:


# calculate review counts for each listing
df_review_cnt = df_review.groupby('listing_id').size().reset_index(name='review_cnt')


# In[26]:


# merge two dataframes by 'listing_id'
df_review_summary = df_review_avg.merge(df_review_cnt, on='listing_id')


# In[30]:


# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Plot the histogram for average sentiment on the left subplot using seaborn
sns.histplot(df_review_summary['avg_sentiment'], color='#FF5A5F', ax=ax1, kde=False, bins=20)

# Plot the histogram for review count on the right subplot using seaborn
# Set the binrange to [0, 50]
sns.histplot(df_review_summary['review_cnt'], color='#00A699', ax=ax2, kde=False, bins=20, binrange=[0, 50])

# Set titles for the subplots 
ax1.set_title('Average Polarity score', fontsize=16)
ax2.set_title('Number of Review Count', fontsize=16)

# Set labels for the x-axis and y-axis
ax1.set_xlabel('Average Sentiment', fontsize=14)
ax1.set_ylabel('Count', fontsize=14)

ax2.set_xlabel('Review Count', fontsize=14)
ax2.set_ylabel('Count', fontsize=14)

# Adjust the layout
fig.tight_layout()

# Display the plots
plt.show()


# # Overall Word Count

# In[31]:


from collections import Counter  # count the frequency of elements in a list

# Combine rows into a single list of words
words = [word for row in df_review['pre_review'] for word in row]

# Count the word frequencies
word_freq = Counter(words)

# Sort the dictionary by frequency in descending order
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse = True)

# Print the words and their frequencies
for word, freq in sorted_word_freq[ :20]:
    print(word, freq)


# In[32]:


# Filter the DataFrame for positive sentiment
mask = df_review['sentiment'] == 'Positive'
pos_words = df_review[mask]['pre_review'].apply(lambda x: x)

# Count word frequencies
pos_word_count = Counter()

for word_list in pos_words:
    pos_word_count.update(word_list)

# Create a DataFrame with the most common positive words
pos_words_df = pd.DataFrame(pos_word_count.most_common(15), columns=['words', 'count'])

# Set color palette
colors = ['#00A699' if i != 7 else '#FF5A5F' for i in range(len(pos_words_df))]

# Plot the bar chart with flipped orientation
plt.figure(figsize=(8, 10), dpi=100)
sns.barplot(data=pos_words_df, y='words', x='count', palette=colors, orient='h')
plt.title("Most Frequent Words in Positive Comments", fontsize=16)
plt.show()


# In[33]:


# Filter the DataFrame for negative sentiment
mask = df_review['sentiment'] == 'Neutral'
neu_words = df_review[mask]['pre_review'].apply(lambda x: x)

# Count word frequencies
neu_word_count = Counter()

for word_list in neu_words:
    neu_word_count.update(word_list)

# Create a DataFrame with the most common negative words
neu_words_df = pd.DataFrame(neu_word_count.most_common(15), columns=['words', 'count'])

# Set color palette
colors = ['#00A699' if i != 3 else '#FF5A5F' for i in range(len(neu_words_df))]

# Plot the bar chart
plt.figure(figsize=(8, 10), dpi=100)
sns.barplot(data=neu_words_df, y='words', x='count', palette=colors)
plt.title("Most Frequent Words in Neutral Comments", fontsize=16)
plt.show()


# In[34]:


# Filter the DataFrame for negative sentiment
mask = df_review['sentiment'] == 'Negative'
neg_words = df_review[mask]['pre_review'].apply(lambda x: x)

# Count word frequencies
neg_word_count = Counter()

for word_list in neg_words:
    neg_word_count.update(word_list)

# Create a DataFrame with the most common negative words
neg_words_df = pd.DataFrame(neg_word_count.most_common(15), columns=['words', 'count'])

# Set color palette
colors = ['#00A699' if i != 1 else '#FF5A5F' for i in range(len(neg_words_df))]

# Plot the bar chart with flipped orientation
plt.figure(figsize=(8, 10), dpi=100)
sns.barplot(data=neg_words_df, y='words', x='count', palette=colors)
plt.title("Most Frequent Words in Negative Comments", fontsize=16)
plt.show()


# # Host Review

# In[35]:


# Filter out the reviews that contains the word "host"
df_review_host = df_review[df_review['pre_review'].apply(lambda x: 'host' in x)]

# Analyze sentiment of reviews that contains the word "host"
df_review_host['host_sentiment'] = df_review_host['pre_review'].apply(lambda x: analyzer.polarity_scores(' '.join(x))['compound'])


# In[36]:


# Filter the DataFrame for positive sentiment
pos_words_host = df_review_host[df_review_host['sentiment'] == 'Positive']['pre_review']

# Count word frequencies
pos_word_host_count = Counter(word for words_list in pos_words_host for word in words_list)

# Initialize and generate WordCloud using positive word frequencies
wc = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(pos_word_host_count)

# Plotting the word cloud
plt.figure(figsize=(12, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Most Frequent Words in Positive Comments", fontsize=16)
plt.show()


# In[37]:


# Filter the DataFrame for positive sentiment
pos_words_host = df_review_host[df_review_host['sentiment'] == 'Positive']['pre_review']

# Count word frequencies
pos_word_host_count = Counter(word for words_list in pos_words_host for word in words_list)

# Initialize and generate WordCloud using positive word frequencies
wc = WordCloud(background_color="white", width=800, height=400).generate_from_frequencies(pos_word_host_count)

# Plotting the word cloud
plt.figure(figsize=(12, 10))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Word Cloud of Most Frequent Words in Positive Comments", fontsize=16)
plt.show()


# In[38]:


#df_review.to_csv('sentiment_all.csv', index=False)
#df_review_host.to_csv('sentiment_host.csv', index=False)

