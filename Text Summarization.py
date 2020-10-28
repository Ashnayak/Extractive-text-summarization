#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np 
import pandas as pd 
import os


# In[2]:


# List files in NLP folder
for dirname, _, filenames in os.walk(r'C:/Users/PreethiBharathy/Documents/Preethi Bharathy/Engineering/NLP'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[3]:


# Converting data into dataframe
news_data_path = r"C:/Users/PreethiBharathy/Documents/Preethi Bharathy/Engineering/NLP/News-dataset/News_dataset.csv"
news_summary_path = r"C:/Users/PreethiBharathy/Documents/Preethi Bharathy/Engineering/NLP/News-dataset/news_summary_more.csv"
news_data_df = pd.read_csv(news_data_path,encoding = "ISO-8859-1")
news_summary_df = pd.read_csv(news_summary_path,encoding = "ISO-8859-1")

# print Head
print(news_data_df.head())
print(news_summary_df.head())

# Print size of DataFrame
print(news_data_df.shape)
print(news_summary_df.shape) # Has 94081 entries, more when compared to news_data_df


# In[4]:


# obserevation - test column in news_dataset is same as the summary in news_summary

# creating new column for summaries (Article_summary)
news_data_df["Article_summary"] = news_data_df['text']
news_data_df = news_data_df.rename(columns={"headlines": "Headlines", "ctext": "Article_content"})

# Remove unwanted columns - text, author, date and read_more
news_data_df = news_data_df.drop(columns=['text','author', 'date', 'read_more'])
news_data_df


# In[5]:


# Expanding contraction words if any, using a default list

CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}


# In[7]:


# Download stop words
import nltk
nltk.download('stopwords')


# In[11]:


# Preprocess data to expand contacted words part 2
from nltk.corpus import stopwords
import re

stop_words = stopwords.words('english')
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    text = text.split() # to convert have'nt -> have not
    for i in range(len(text)):
        word = text[i]
        if word in contraction_mapping:
            text[i] = contraction_mapping[word]
    text = " ".join(text)
    text = text.split()
    newtext = []
    for word in text:
        if word not in stop_words:
            newtext.append(word)
    text = " ".join(newtext)
    text = text.replace("'s",'') # to convert your's -> your
    text = re.sub(r'\(.*\)','',text) # remove (words)
    text = re.sub(r'[^a-zA-Z0-9. ]','',text) # remove punctuations
    text = re.sub(r'\.',' . ',text)
    return text

sample = "(hello) hi there .man tiger caller who's that isn't it ? WALL-E"
after_contraction_removal = expand_contractions(sample)
print(after_contraction_removal)


# In[12]:


# Testing on Dataset

news_summary_df['headlines'] = news_summary_df['headlines'].apply(lambda x:expand_contractions(x))
news_summary_df['text'] = news_summary_df['text'].apply(lambda x:expand_contractions(x))
print(news_summary_df['headlines'][20],news_summary_df['text'][20])


# In[61]:


nltk.download('punkt')


# In[13]:


nltk.download('wordnet')


# In[67]:


# Pre-processing

# create feature matrix
feature_matrix = pd.DataFrame(columns=[
                                       'Article#',
                                       'Article_sentence# ',
                                       'Sentence',
                                       'TitleFeature',
                                       'SentenceLength',
                                       'SentencePosition',
                                       'AvgTermFrequency',
                                       'SimilarityScore',
                                       'In summary?'])


# In[68]:


# Functions for pre-processing

def title_feature(sentence, heading):
    tf_count=0
    sentence_array=sentence.split()
    lenght_heading=len(heading)
    for word_in_sen in sentence_array:
        for word_in_head in heading:
            if word_in_sen==word_in_head:
                tf_count=tf_count+1
    tf_value=float(tf_count/lenght_heading)
    return round(tf_value,3)


def sentence_length(max_sentence_length, sentence):
    words_in_sentence = len(sentence.split(' '))
    return round(words_in_sentence/max_sentence_length, 3)


def sentence_position(positon, sentence, end):
    if(positon==0 or sentence == end):  #len_sentence: should be last sentence
        return 1
    else:
        return 0
    
def similarity_score(cosine_similarity):
    cosine_similarity_sum = 0
    for i in range(0, len(cosine_similarity)):
        cosine_similarity_sum += cosine_similarity[i]
    return(round(cosine_similarity_sum,3))


# In[69]:


import re
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
len(news_data_df)
for i in range(len(news_data_df)):
    index = i+1
    print("Article " + str(i+1))
    
    # Removing contraction words if any in headings
    
    heading = expand_contractions(str(news_data_df['Headlines'][i]))
    #print("heading---", heading)
    
    # Removing contraction words if any in articles and article summaries
    
    article_content = expand_contractions(str(news_data_df['Article_content'][i]))
    #print("article_content---", article_content)
    summary_content = expand_contractions(str(news_data_df['Article_summary'][i]))
    #print("summary_content---", summary_content)
    
    # Sentence tonkenization - Splitting paras into array of sentences (for articles and summarizes)
    
    article_sentence_tokens = sent_tokenize(article_content)
    #print("article_sentence_tokens---", article_sentence_tokens)
    summary_sentence_tokens = sent_tokenize(summary_content)
    #print("summary_sentence_token---", summary_sentence_tokens)
    
    # Word tokenization - tonkenize sentences into array of words (for headings and articles)
    
    heading_word_tokens = word_tokenize(heading)
    article_word_tokens = word_tokenize(article_content)
    
    # Remove stop words - words like in/on/the etc (from headings and articles)

    heading_without_sw = [word for word in heading_word_tokens if not word in stopwords.words('english')]
    article_without_sw = [word for word in article_word_tokens if not word in stopwords.words('english')]
    #print("heading_without_sw---", heading_without_sw)
    #print("article_without_sw---", article_without_sw)
    
    # Stemming or Lemmatization - Used to reduce to root word (Lemmatization has better precision)
    lemmatizer = WordNetLemmatizer()
    heading_lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in heading_without_sw])
    article_lemmatized = ' '.join([lemmatizer.lemmatize(w) for w in article_without_sw])
#     print("heading_lemmatized---", heading_lemmatized)
#     print("article_lemmatized---", article_lemmatized)
    
    # Checking results with stemming just in case uisng porter's algorithm 
#     from nltk.stem.porter import PorterStemmer
#     porter = PorterStemmer()
#     stem_sentence=[]
#     for word in heading_without_sw:
#         stem_sentence.append(porter.stem(word))
#         stem_sentence.append(" ")
#         stem_heading = "".join(stem_sentence)
#     print("stem_heading---", stem_heading) # lemmatization gave better results
    
    # Sentence segmentation - dividing a paragraph based on sentences ( An other way of sentence tokenizing)
    sentence = re.split('\. |\n\n',article_lemmatized)
    #print("sent---", sentence)
    
    try: # in case of value error (for example if the tonkenized sentences has only " ", "." or any special character)


        # title feature  - To idenftify the ratio of title words used in a sentence
        title_feature_list = []
        heading = heading_lemmatized
        for i in range(0, len(sentence)):
            title_feature_list.append(title_feature(sentence[i], heading))
        #print('title_feature_list ---', title_feature_list)

        
       #Sentence length - ratio of lenght of the sentence to the maximum length of a sentence in a list of sentences
        individual_length = []
        sentence_length_feature = []
        for i in range(0, len(sentence)):     #to find max length sentence
            individual_length.append(len(sentence[i].split()))
        max_length = max(individual_length)
        for i in range(0, len(sentence)):
            sentence_length_feature.append(sentence_length(max_length, sentence[i]))
        #print("sentence_length_feature ---", sentence_length_feature)

        
        #Sentence Position - The sentence in the fisrt and last position get more priority
        sentence_position_feature = []
        reverse = sentence[::-1]
        for i in range(0, len(sentence)):
            sentence_position_feature.append(sentence_position(i, sentence[i], reverse[0]))
        # print("sentence_position ---", sentence_position_feature)

        # Term frequency, also know as tf–idf
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf_idf_list = []
        templist = []
        temp = []
        # convert sentences to its vectors
        vectorizer = TfidfVectorizer()
        vector = vectorizer.fit_transform(sentence)
        vector_list = vector.toarray().tolist()
        #print("vector_list ---", vector_list)
        j = 0
        for sublist in vector_list:
            temp = [round(i,3) for i in sublist]
            templist.append(temp)
            tf_idf_list.append(round(np.mean(temp),3))
            j = j+1
        #print("tf_idf_list ---", tf_idf_list)

        # Sentence similarity score
        
        from sklearn.metrics.pairwise import cosine_similarity
        vect_for_sim = TfidfVectorizer(analyzer = 'char')
        Sentence_similarity_list= []
        vector = vect_for_sim.fit_transform(sentence)
        cosine_similarity_list = cosine_similarity(vector,vector)
        for i in range(0,len(cosine_similarity_list)):
            Sentence_similarity_list.append(similarity_score(cosine_similarity_list[i])/len(cosine_similarity_list))
        #print("Sentence_similarity_list ---", Sentence_similarity_list)

        #class 'In_Summary' -- to identify of the sentence is in summary or not
        
        in_summary = []
        for sentence in article_sentence_tokens:
            if sentence in summary_sentence_tokens:
                in_summary.append(1)
            else:
                in_summary.append(0)
        #print("in_summary ---", in_summary)
        
        # To find the count of sentences in the tonkenized sentence list
        list_index_count = []
        for i in range(len(article_sentence_tokens)):
            list_index_count.append(i)
        print("list_index_count---", list_index_count)
        
        #Add to feature matrix
        for sentence in range(len(list_index_count)):
            feature_matrix = feature_matrix.append({
                                           'Article#': index,
                                           'Article_sentence# ' : sentence,
                                           'Sentence': article_sentence_tokens[sentence],
                                           'TitleFeature': title_feature_list[sentence],
                                           'SentenceLength': sentence_length_feature[sentence],
                                           'SentencePosition': sentence_position_feature[sentence],
                                           'AvgTermFrequency':tf_idf_list[sentence],
                                           'SimilarityScore': Sentence_similarity_list[sentence],
                                           'In summary?': in_summary[sentence]},ignore_index=True)
    except ValueError as ve:
        # Print the exception
        print(f'The Article {str(i+1)}, with sentence --- {sentence}, is not a valid sentence')

        # To find the count of sentences in the tonkenized sentence list
        list_index_count = []
        for i in range(len(article_sentence_tokens)):
            list_index_count.append(i)

        # Store feature set values as in this case
        title_feature_list.append(0)
        sentence_length_feature.append(0)
        sentence_position_feature.append(0)
        tf_idf_list.append(0)
        Sentence_similarity_list.append(0)
        in_summary.append(1)

        #Add to feature matrix
        for sentence in range(len(list_index_count)):
            feature_matrix = feature_matrix.append({
                                           'Article# ' : index,
                                           'Article_sentence# ' : sentence,
                                           'Sentence': article_sentence_tokens[sentence],
                                           'TitleFeature': title_feature_list[sentence],
                                           'SentenceLength': sentence_length_feature[sentence],
                                           'SentencePosition': sentence_position_feature[sentence],
                                           'AvgTermFrequency':tf_idf_list[sentence],
                                           'SimilarityScore': Sentence_similarity_list[sentence],
                                           'In summary?': in_summary[sentence]},ignore_index=True)


# In[70]:


print(feature_matrix)
feature_matrix.to_csv(r'C:/Users/PreethiBharathy/Documents/Preethi Bharathy/Engineering/NLP/Feature_matrix_News_data_v1.csv', index=False)

