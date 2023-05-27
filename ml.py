import numpy as np
import pandas as pd
movie =pd.read_csv('tmdb_5000_movies.csv')
credit =pd.read_csv('tmdb_5000_credits.csv')
movie.head(1)
# credit.head(1)['cast'].values
credit.head(1)
movie=movie.merge(credit,on='title')
#gener , id , keywords,title,overview,cast,crew
movie=movie[['movie_id','title','overview','genres','keywords','cast','crew']]
movie.head(1)
# for checking missing data 
movie.isnull().sum()
movie.dropna(inplace=True)# drop the 3 overview with null
movie.isnull().sum()# no missing data is now
movie.duplicated().sum() # checking any dublicate values in dataset
#movie.head(1)['genres'].values # both same but this shows array
movie.iloc[0].genres
import ast
# ast.literal_eval()
def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L
movie['genres']=movie['genres'].apply(convert)
movie['keywords']=movie['keywords'].apply(convert)
movie.head(1)
#movie['cast'][0]
import ast
# ast.literal_eval()

def convert3(obj):
    L=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L
movie['cast']=movie['cast'].apply(convert3)
#movie['crew'][0]
def feth_dire(obj):
    L=[]
    
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            L.append(i['name'])
            break   
    return L
movie['crew']=movie['crew'].apply(feth_dire)
movie.head(1)
movie['overview']=movie['overview'].apply(lambda x:x.split())
movie.head(1)
movie['genres']=movie['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movie['keywords']=movie['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movie['cast']=movie['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movie['crew']=movie['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movie['tags']=movie["overview"]+movie["genres"]+movie["keywords"]+movie["cast"]+movie['crew']
new_df=movie[['movie_id',"title",'tags']]




#!pip install nltk for natural lang processing
new_df['tags']=new_df['tags'].apply(lambda x:' '.join(x))
new_df.head()
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
import nltk
from nltk.stem.porter import PorterStemmer
ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags']=new_df['tags'].apply(stem)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=5000,stop_words="english")
vectors=cv.fit_transform(new_df['tags']).toarray()
cv.get_feature_names()
from sklearn.metrics.pairwise import cosine_similarity

similarity=cosine_similarity(vectors)# simailarity with each rest movie 
def recommend(movies):
    movies_index=new_df[new_df['title']==movies].index[0]
    distance=similarity[movies_index]
    movie_list=sorted(list(enumerate(distance)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        #print(i[0])
import pickle
pickle.dump(similarity,open('similarity.pkl','wb'))

pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(new_df,open('movies.pkl','wb'))