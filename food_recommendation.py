# -*- coding: utf-8 -*-
"""food_recommendation.ipynb

@author: Bhuvanjeet

**Food Recommendation Engine**

To recommend food items based on similarity in 'brand' and 'ingredients'.


**Project Overview:**

**1-Exploratory Data Analysis - EDA**

**2-Vectorization**

**3-Cosine Similarity**

**4-Input and Output**
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('food_items.txt')

df.drop('index',axis=1,inplace=True)

df.drop(df.iloc[:,5:],axis=1,inplace=True)

df.head()

"""or you could simply have done:

df = df [ ['brand','categories','ingredients','manufacturer','title'] ]

this will automatically drop other columns
"""

#data cleaning
#removing duplicates from df
df.drop_duplicates(inplace=True)

#filling null places with ''
df=df.fillna(value='')

"""We want to predict 'title' of food item. So from the dataframe, we can see that it depends basically on two features: 'brand' and 'ingredients'.So, we will combine these two columns into 1 column."""

df['prepared_data'] = df['brand'] + " " + df['ingredients']

"""**Vectorization**"""

from sklearn.feature_extraction.text import CountVectorizer
#converts a collection of text documents to a matrix of token counts

#bow - bag-of-words model
bow_transformer = CountVectorizer() 
vector_matrix = bow_transformer.fit_transform(df['prepared_data'])


"""**Cosine Similarity**"""

from sklearn.metrics.pairwise import cosine_similarity

cos_similar = cosine_similarity(vector_matrix)

"""**Input**"""

food_item = input('Enter the title of the food : ')

def search_index(title):
    return df[df['title'] == title].index

index_item=search_index(food_item)

#making a list of similar items to the item entered by the user
similar_items = list(enumerate(cos_similar[int(index_item[0])]))


#sorting in descending order
sorted_similar_items=sorted(similar_items,key=lambda x:x[1],reverse=True)

sorted_similar_items[:10]     #to display first 10 elements of the sorted list

"""**Output**"""

def search_title(index):
    return df['title'][index]

i=0
for item in sorted_similar_items:
    print(search_title(item[0]))
    i=i+1
    if(i>50):     #printing top 50 similar food items
        break

"""So, we have successfully built a food recommendation engine."""
