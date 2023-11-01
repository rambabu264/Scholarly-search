import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy import spatial
import re
import pickle

tokenizer = nltk.tokenize.toktok.ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words('english')


def remove_stopwords(text, is_lower_case=False):
    pattern = r'[^a-zA-z0-9s]'
    text = re.sub(pattern," ",''.join(text))
    tokens = tokenizer.tokenize(text)
    tokens = [tok.strip() for tok in tokens]
    if is_lower_case:
        cleaned_token = [tok for tok in tokens if tok not in stopword_list]
    else:
        cleaned_tokens = [tok for tok in tokens if tok.lower() not in stopword_list]
    filtered_text = ' '.join(cleaned_tokens)
    return filtered_text


def glove_vectors_():
    glove_vectors = dict()
    glove_file = 'F://IR/glove6B/glove.6B.300d.txt'
    file = open(glove_file, encoding = 'utf-8')
    for line in file:
        values = line.split()
        word = values[0]
        vectors = np.asarray(values[1:])
        glove_vectors[word] = vectors
    file.close()
    return glove_vectors

vec_dimension = 300
def get_embedding(x):
    arr  = np.zeros(vec_dimension)
    text = str(x).split()
    
    with open('F://IR/glove_vectors.pkl', 'rb') as file:
        glove_vectors = pickle.load(file)

    for t in text:
        try:
            vec = glove_vectors.get(t).astype(float)
            arr = arr + vec
        except:
            pass
    arr = arr.reshape(1,-1)[0]
    return (arr/len(text))

out_dict = {}

# if __name__ == "__main__":
#     csv_file = 'F://IR/Data/Data.csv'
#     df = pd.read_csv(csv_file)

#     for sen in zip(df["PMID"], df["Abstract"]):
#         if sen[1]:
#             average_vector = (np.mean(np.array([get_embedding(x) for x in nltk.word_tokenize(remove_stopwords(sen[1]))]), axis=0))
#             dict = { sen[0] : (average_vector) }
#             out_dict.update(dict)

#     df.set_index('PMID', inplace=True)
#     df['Values'] = df.index.map(out_dict)
#     df.to_csv("F://IR/Data/EMbedded_Data.csv")