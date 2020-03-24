#from __future__ import print_function
# importing modules
import numpy as np
import gensim
from nltk.tokenize import word_tokenize
import json

# Load Google's pre-trained Word2Vec model.
model = gensim.models.KeyedVectors.load_word2vec_format('./model/GoogleNews-vectors-negative300.bin', binary=True)
#model.get_vector('Get')
#model.distance('get','Get')
#model.similar_by_vector(model.get_vector('Get'))


from autocorrect import Speller
spell = Speller(lang='en')
remove_words = []
def read_a_caption_file(name):
    max_length_sentences = 0
    with open(name) as json_file:
        text = json.load(json_file)
        for i in range(len(text)):
            text[i]['caption'] = text[i]['caption'].lower()
            text[i]['caption'] = [spell(word) for word in word_tokenize(text[i]['caption'])\
                                  if word.isalpha() and len(word)>1 and word not in remove_words]
            text[i]['caption'] = ['$'] + text[i]['caption'] + ['E']
            if (max_length_sentences<len(text[i]['caption'])):
                max_length_sentences = len(text[i]['caption'])
    return max_length_sentences, text

max_length_sentences1, text1 = read_a_caption_file('captions.json')
max_length_sentences2, text2 = read_a_caption_file('additional_captions.json')

text = text1 + text2
max_length_sentences = max(max_length_sentences1, max_length_sentences2)




def remove_bad_word(text):
    temp = []
    for i in range(len(text)):
        flag = 0
        for word in text[i]['caption']:
            if word not in model.vocab and word.upper() not in model.vocab:
                flag = 1
                break
        if (flag == 0):
            temp = temp + [text[i].copy()]
    return temp
text = remove_bad_word(text)


#index_not_exist = [i for i in range(len(is_exist)) if is_exist[i]==0]
#word_not_exist = [dictionary[i] for i in index_not_exist]


end_word_padd = '#'
for i in range(len(text)):
    text[i]['caption'] = text[i]['caption']\
                         + [end_word_padd for i in range(max_length_sentences-len(text[i]['caption']))]

import csv
with open('edited caption.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # header of file
    writer.writerow(['image_id'] + [str(i+1) for i in range(max_length_sentences)])
    for i in range(len(text)):
        writer.writerow([str(text[i]['image_id'])] +text[i]['caption'])


dictionary = set([text[i]['caption'][j] for i in range(len(text)) for j in range(len(text[i]['caption']))])
dictionary = list(dictionary)
"""
encoded = text.copy()
for i in range(len(text)):
    if i%100==0:
        print(str(i) +'/' + str(len(text)))
    for j in range(max_length_sentences):
        if encoded[i]['caption'][j] in model.vocab:
            encoded[i]['caption'][j] = model.get_vector(encoded[i]['caption'][j])
        else:
            encoded[i]['caption'][j] = model.get_vector(encoded[i]['caption'][j].upper())
"""

encoded = dictionary.copy()
for i in range(len(dictionary)):
    if i%1000==0:
        print(str(i) +'/' + str(len(dictionary)))
    if encoded[i] in model.vocab:
        encoded[i] = model.get_vector(encoded[i])
    else:
        encoded[i] = model.get_vector(encoded[i].upper())

import csv
with open('encoded dictionary.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # header of file
    writer.writerow(['word'] + [str(i+1) for i in range(300)])
    for i in range(len(dictionary)):
        if i%1000==0:
            print(str(i) + '/' + str(len(dictionary)))
        writer.writerow([dictionary[i]] +[str(encoded[i][j]) for j in range(len(encoded[i]))])
