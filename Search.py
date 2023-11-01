import nltk
import scipy
import numpy as np
import pandas as pd
import Word_Embedding as WE
import ast
import math


out_dict = {}

def get_sim(query_embedding, average_vector_doc):
    sim = [(1 - scipy.spatial.distance.cosine(query_embedding, average_vector_doc))]
    return sim

def Ranked_documents(query):
    query_word_vectors = (np.mean(np.array([WE.get_embedding(x) for x in nltk.word_tokenize(query.lower())],dtype=float), axis=0))
    rank = []
    for k,v in out_dict.items():
        rank.append((k, get_sim(query_word_vectors, np.array(v))))
        rank = sorted(rank,key=lambda t: t[1], reverse=True)
    return rank

def search(query):
    df = pd.read_csv("F://IR/Data/Embedded_Data.csv")
    df['Values'] = df['Values'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df.set_index('PMID', inplace=True)

    for index, row in df.iterrows():
        PMID = index
        values = row['Values']
        out_dict[PMID] = values

    ranks = Ranked_documents(query)
    return ranks

def and_operation(left_operand, right_operand):
    result = []                                 
    l_index = 0                              
    r_index = 0                                 
    l_skip = int(math.sqrt(len(left_operand)))  
    r_skip = int(math.sqrt(len(right_operand)))

    while (l_index < len(left_operand) and r_index < len(right_operand)):
        l_item = left_operand[l_index]  
        r_item = right_operand[r_index] 
        
        if (l_item == r_item):
            result.append(l_item)   
            l_index += 1            
            r_index += 1           
        
        elif (l_item > r_item):
            if (r_index + r_skip < len(right_operand)) and right_operand[r_index + r_skip] <= l_item:
                r_index += r_skip
            else:
                r_index += 1

        else:
            if (l_index + l_skip < len(left_operand)) and left_operand[l_index + l_skip] <= r_item:
                l_index += l_skip
            else:
                l_index += 1

    return result

def or_operation(left_operand, right_operand):
    result = []     
    l_index = 0    
    r_index = 0     

    while (l_index < len(left_operand) or r_index < len(right_operand)):
        if (l_index < len(left_operand) and r_index < len(right_operand)):
            l_item = left_operand[l_index]  
            r_item = right_operand[r_index]
            
            if (l_item == r_item):
                result.append(l_item)
                l_index += 1
                r_index += 1

            elif (l_item > r_item):
                result.append(r_item)
                r_index += 1

            else:
                result.append(l_item)
                l_index += 1

        elif (l_index >= len(left_operand)):
            r_item = right_operand[r_index]
            result.append(r_item)
            r_index += 1

        else:
            l_item = left_operand[l_index]
            result.append(l_item)
            l_index += 1

    return result

def not_operation(right_operand, indexed_docIDs):
    if (not right_operand):
        return indexed_docIDs
    
    result = []
    r_index = 0 
    for item in indexed_docIDs:
        if (item != right_operand[r_index]):
            result.append(item)
        elif (r_index + 1 < len(right_operand)):
            r_index += 1
    return result