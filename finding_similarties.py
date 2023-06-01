from __future__ import division
#import matplotlib.pyplot as plt
import nltk
import random
import re, pprint, os, numpy
import sys
from nltk import cluster
from nltk.cluster import KMeansClusterer
from nltk.stem.porter import PorterStemmer
from collections import defaultdict
stemmer = PorterStemmer()
from sklearn.metrics.cluster import entropy
nltk.download('stopwords')
nltk.download('punkt')
#_______________________________________________

# Code to read in a directory of text files, create nltk.Text objects out of them,
# load an nltk.TextCollection object and create a BOW with TF*IDF values.

# First set the variable path to the directory path.  Use
# forward slashes (/), even on Windows.  Make sure you
# leave a trailing / at the end of this variable.

#path = '/home/xyz/Desktop/web/docs'
path = r'C:\Users\vaibh\OneDrive\Desktop\newsgroups' 

# Empty list to hold text documents.
texts = []
stopwords = set(nltk.corpus.stopwords.words('english'))
def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        if item not in stopwords:
            stemmed.append(stemmer.stem(item))    
    return stemmed

# Iterate through the  directory and build the collection of texts for NLTK.
dict1 = {}
dict1 = defaultdict(lambda:0,dict1)
for subdir, dirs, files in os.walk(path):
    for file in files:
        url = subdir + os.path.sep + file
        f = open(url, 'r', encoding='ISO-8859-1')
        raw = f.read()
        f.close()
        tokens = nltk.word_tokenize(raw) 
        tokens = stem_tokens(tokens, stemmer)
        text = nltk.Text(tokens)
        for x in tokens:
            dict1[x]+=1
        texts.append(text)

print ("Prepared ", len(texts), " documents...")
print ("They can be accessed using texts[0] - texts[" + str(len(texts)-1) + "]")


#Load the list of texts into a TextCollection object.
collection = nltk.TextCollection(texts)
print ("Created a collection of") 
print (len(collection)) 
print ("terms.")

#get a list of unique terms
unique_terms = list(set(collection))
def cnt(x):
    return dict1[x]
unique_terms.sort(key=cnt,reverse=True)
print ("Unique terms found: ")
print (len(unique_terms))
newlist = []
for x in collection:
    if x in unique_terms[:3000]:
        newlist.append(x)

newcollection = nltk.TextCollection(newlist)

# Function to create a TFIDF vector for one document.  For each of
# our unique words, we have a feature which is the td*idf for that word
# in the current document
def TFIDF(document):
    word_tfidf = []
    for word in unique_terms[:3000]:
        word_tfidf.append(newcollection.tf_idf(word,document))
    return word_tfidf

### And here we actually call the function and create our array of vectors.
vectors = [numpy.array(TFIDF(f)) for f in texts if len(f) != 0]
#vector1 = [set(v) for v in vectors]  # Convert vectors to sets
print ("Vectors created.")

import sys
#from util import manhattan_distance, chebyschev_distance,pearson_distance,jaccard_distance

from nltk.cluster import cosine_distance, euclidean_distance
from scipy.spatial.distance import euclidean,cityblock,chebyshev,correlation
from nltk.metrics.distance import jaccard_distance

# mets = ['cosine','euclidean','manhattan','chebyschev','pearson','jaccard']
mets=['chebyschev']
import os
f=open(r'C:\Users\vaibh\OneDrive\Desktop\out.txt','w')
for met in mets:

    clusters1 = []
    clusters2 = []
    clusters3 = []
    # if(met=='cosine'):
    #     metric = cosine_distance
    # elif(met=='euclidean'):
    #     metric = euclidean
    # elif(met=='manhattan'):
    #     metric = cityblock
    # elif(met=='chebyschev'):
    metric = chebyshev
    # elif(met=='pearson'):
    #     metric = correlation
   # elif(met=='jaccard'):
    #    metric = jaccard_distance

    a=0.0
    b=0.0
    for x in range(30):#30 denotes no of times algorithm will run
        clusterer = KMeansClusterer(3,metric,avoid_empty_clusters=True)#3 is the number of clusters
        clusters = clusterer.cluster(vectors,assign_clusters=True, trace=False)
        #print(clusters)
        #means = clusterer.means()
        #print (means)
        a+=entropy(clusters)#entropy means less variation of data in one cluster a means entropy & b means purity calcula

        labels=[]

        cnt = 0
        lcnt=[]
        for root,dirs,files in os.walk(path):
            if(len(files) > 0):
                cnt+=1
                lcnt.append(0)
                for i in range(len(files)):
                    labels.append(cnt-1)
        ans=0
        for i in range(0,3):
            for x in range(0,cnt):
                lcnt[x]=0
            for j in range(0,len(labels)):
                if(clusters[j] == i):
                    lcnt[labels[j]]+=1
            ans += max(lcnt)
        b += (float(ans)/len(set(labels)))#calculation of purity
    
    for k in range(0, len(texts)):
        if(clusters[k]==0):
           clusters1.append(k)
        if(clusters[k]==1):
           clusters2.append(k)
        if(clusters[k]==2):
           clusters3.append(k)
    print("First cluster: ")
    print(clusters1)
    print('\nNumber of documents in first cluster: ', len(clusters1))
    print("\n Second cluster: ")
    print(clusters2)
    print('\nNumber of documents in second cluster: ', len(clusters2))
    print("\n Third cluster: ")
    print(clusters3)
    print('\nNumber of documents in third cluster: ', len(clusters3))
    
    a=a/30.0
    b=b/30.0
    print("entropy")
    print(a)
    print(b)
    '''
    f.write(path+'\n'+ met+ '\n' +'Purity:'+str(b)+ '\n' +'Entropy:'+str(a)+ '\n' '--------------------'+ '\n')
    print ("Entropy:")
    print (a)
    print ("Purity:")
    print (b)
    plt.plot(clusters)
    plt.show()'''
    f.write('DIFFERENT CLUSTERS: \n\n\n')
    #c1
    f.write('cluster 1: \n')
    f.write('[')
    for items in clusters1:
        f.write("%s, " % items)
    f.write(']')
    f.write('\n')
    #c2
    f.write('cluster 2: \n')
    f.write('[')
    for items in clusters2:
        f.write("%s, " % items)
    f.write(']')
    f.write('\n')
    #c3
    f.write('cluster 3: \n')
    f.write('[')
    for items in clusters3:
        f.write("%s, " % items)
    f.write(']')
    f.write('\n\n\n')
    f.write("genereted report: \n")
    f.write(path+'\n'+ met+ '\n' +'Purity:'+str(b)+ '\n' +'Entropy:'+str(a)+ '\n' '--------------------'+ '\n')

f.close()
