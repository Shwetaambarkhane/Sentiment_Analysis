from django.shortcuts import render
import requests
import pickle
#from . import preprocess

from django.shortcuts import HttpResponse
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from textblob import Word
import re

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB

import plotly.offline as opy
import plotly.graph_objs as go
import io

# Create your views here.
def predict(request):
    if request.method=='POST':
        message=request.POST.get('message',None)
        clean_message = preprocess.clean(message)

        with open('model.pickle', 'rb') as f:
            ml_model = pickle.load(f)
        
        result={'res':ml_model.classify(clean_message).upper()}
        
        return render(request,'ContactFrom/results.html',result)

    else:
        return render(request,'ContactFrom/index.html')


def index(request):

	if request.method == 'POST':

		# Dataset
		data = pd.read_csv('text_emotion.csv', skiprows = [i for i in range(3, 32000)])
		data = data.drop('author', axis=1)

		# Making all letters lowercase
		data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

		# Removing Punctuation, Symbols
		data['content'] = data['content'].str.replace('[^\w\s]',' ')

		# Removing Stop Words using NLTK
		stop = stopwords.words('english')
		data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

		#Lemmatisation
		data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
		#Correcting Letter Repetitions

		def de_repeat(text):
			pattern = re.compile(r"(.)\1{2,}")
			return pattern.sub(r"\1\1", text)

		data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

		# Code to find the top 10,000 rarest words appearing in the data
		freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

		# Removing all those rarely appearing words from the data
		freq = list(freq.index)
		data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

		#Encoding output labels 'sadness' as '1' & 'happiness' as '0'
		lbl_enc = preprocessing.LabelEncoder()
		y = lbl_enc.fit_transform(data.sentiment.values)
		ylist=y.tolist()

		p=[5,7,11,1,2]
		n=[0,6,12,10,3,4]
		positive=[]
		negative=[]
		neutral=[]

		for q in ylist:
			if q in p:
				positive.append(q)
			elif q in n:
				negative.append(q)
			else:
				neutral.append(q)

		# Splitting into training and testing data in 90:10 ratio
		X_train, X_val, y_train, y_val = train_test_split(data.content.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)

		# Extracting TF-IDF parameters
		tfidf = TfidfVectorizer(max_features=1000, analyzer='word',ngram_range=(1,3))
		X_train_tfidf = tfidf.fit_transform(X_train)
		X_val_tfidf = tfidf.fit_transform(X_val)

		# Extracting Count Vectors Parameters
		count_vect = CountVectorizer(analyzer='word')
		count_vect.fit(data['content'])
		X_train_count =  count_vect.transform(X_train)
		X_val_count =  count_vect.transform(X_val)

		mnb = MultinomialNB(alpha=0.001)
		mnb.fit(X_train_count, y_train)
		y_pred = mnb.predict(X_val_count)

		tweets = request.POST.get('message',None)

		# Doing some preprocessing on these tweets as done before
		#tweets[0] = tweets[0].replace('[^\w\s]',' ')
		#from nltk.corpus import stopwords
		stop = stopwords.words('english')
		tweets = pd.Series(tweets)
		tweets = tweets.apply(lambda x: " ".join(x for x in x.split() if x not in stop))
		#from textblob import Word
		tweets = tweets.apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))

		# Extracting Count Vectors feature from our tweets
		tweet_count = count_vect.transform(tweets)

		#Predicting the emotion of the tweet using our already trained linear SVM
		tweet_pred = mnb.predict(tweet_count)
		#print(tweet_pred)
		x = lbl_enc.inverse_transform(tweet_pred)
		s = ''
		if tweet_pred in p:
			s = "Thank you for your response. We're so grateful for your kind words"
		elif tweet_pred in n:
			s = "Sorry for your inconvenience. We will use the feedback to make us better"
		else:
			s = "Thank you for letting us know about this."

		return render(request,'ContactFrom/results.html',{'res':x[0], 'quot':s})

	else:
		return render(request,'ContactFrom/index.html')

def analysis(request):
	
	data = pd.read_csv('text_emotion.csv', skiprows = [i for i in range(3, 32000)])
	data = data.drop('author', axis=1)

	data['content'] = data['content'].apply(lambda x: " ".join(x.lower() for x in x.split()))

	# Removing Punctuation, Symbols
	data['content'] = data['content'].str.replace('[^\w\s]',' ')

	# Removing Stop Words using NLTK
	stop = stopwords.words('english')
	data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

	#Lemmatisation
	data['content'] = data['content'].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
	#Correcting Letter Repetitions

	def de_repeat(text):
		pattern = re.compile(r"(.)\1{2,}")
		return pattern.sub(r"\1\1", text)

	data['content'] = data['content'].apply(lambda x: " ".join(de_repeat(x) for x in x.split()))

	# Code to find the top 10,000 rarest words appearing in the data
	freq = pd.Series(' '.join(data['content']).split()).value_counts()[-10000:]

	# Removing all those rarely appearing words from the data
	freq = list(freq.index)
	data['content'] = data['content'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

	#Encoding output labels 'sadness' as '1' & 'happiness' as '0'
	lbl_enc = preprocessing.LabelEncoder()
	y = lbl_enc.fit_transform(data.sentiment.values)
	ylist=y.tolist()

	p=[5,7,11,1,2]
	n=[0,6,12,10,3,4]
	positive=[]
	negative=[]
	neutral=[]

	for q in ylist:
		if q in p:
			positive.append(q)
		elif q in n:
			negative.append(q)
		else:
			neutral.append(q)
	#calculating length of all three list for ploting	
	poscnt=len(positive)
	negcnt=len(negative)
	neucnt=len(neutral)

	emotion1=["positive","negative","neutral"]

	slices1=[poscnt,negcnt,neucnt]
	colors1=['g','r','y']
	myexplode=[0,0.1,0]

	senti=[]
	senti=data['sentiment'].to_list()

	#ploting bar graph for positive
	pone=senti.count("fun")
	ptwo=senti.count("enthusiasm")
	pthree=senti.count("love")
	pfour=senti.count("happiness")
	pfive=senti.count("surprise")

	emotion2=["fun","enthusiasm","love","happiness","surprise"]

	slices2=[pone,ptwo,pthree,pfour,pfive]
	colors2=['r','y','g','b','m']

	#negative cluster plot

	none=senti.count("anger")
	ntwo=senti.count("hate")
	nthree=senti.count("worry")
	nfour=senti.count("boredom")
	nfive=senti.count("empty")
	nsix=senti.count("sadness")

	emotion3=["anger","hate","worry","boredom","empty","sadness"]

	slices3 = [none,ntwo,nthree,nfour,nfive,nsix]
	colors3=['r','y','g','b','m','k']

	#neutral cluster plot

	one=senti.count("relief")
	two=senti.count("neutral")

	emotion4=["relief","neutral"]

	slices4=[one,two]
	colors4=['r','y']

	#overall
	
	emotion5 = ["fun", "enthusiasm", "love", "happiness", "surprise", "anger", "hate", "worry", "boredom", "empty", "sadness", "relief", "neutral"]

	slices5=[pone,ptwo,pthree,pfour,pfive,none,ntwo,nthree,nfour,nfive,nsix,one,two]
	

	return render(request, 'ContactFrom/analysis.html', {'slices1':slices1 , 'emotion1': emotion1, 'slices5':slices5 , 'emotion5': emotion5})
