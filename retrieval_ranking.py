import tweet_tokenizer as tweet
import indexer
import numpy as np
import math

def retrieval_ranking(query_ls, inverted_index=indexer.indexing_tokens()):
    
    query_index, doc_list = query_indexing(inverted_index, query_ls)
    # Log tf-idf weighting in query matrix
    matrix_q = np.zeros(shape=(len(query_ls), len(doc_list)+1))
    for x in range(len(query_ls)):
        term_i = query_ls[x]
        matrix_q[x][0] = weighting_query(tf_iq_weight(query_ls, term_i), idf_weight(query_index, term_i))
        for y in range(len(doc_list)):
            docID = doc_list[y]
            matrix_q[x][y+1] = weighting_doc(tf_ij_weight(query_index, term_i, docID), idf_weight(query_index, term_i))

    # Log length normalization in normalization matrix
    matrix_n = np.zeros(shape=(len(query_ls), len(doc_list)+1))
    for y in range(len(doc_list)+1):
        tmp_list = []
        square_sum = 0
        for x in range(len(query_ls)):
            tmp_list.append(matrix_q[x][y])
            square_sum += (matrix_q[x][y])**2
        for z in range(len(tmp_list)):
            matrix_n[z][y] = (tmp_list[z]) / (math.sqrt(square_sum))

    # Compute similarity between each document and query
    similarity_dict = {}
    similarity_sorted = []
    ranked_docs = []
    for y in range(0, len(doc_list)):
        tmp_cos = 0
        for x in range(len(query_ls)):
            tmp_cos += matrix_n[x][0]*matrix_n[x][y+1]
        similarity_dict[doc_list[y]] = tmp_cos

    similarity_sorted = sorted(similarity_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

    for z in range(len(similarity_sorted)):
        ranked_docs.append(similarity_sorted[z][0])

    return similarity_sorted, ranked_docs


#{"bbc":{"123456":1,"234567":1}, "cuts":{"467903":1}}

def query_indexing(inverted_index, query_ls):
    query_index =  {}
    doc_num = 0
    doc_list = []
    for i in range(len(query_ls)):
        keyword = query_ls[i]
        if keyword in inverted_index:
            query_index[keyword] = inverted_index[keyword]
            doc_list = doc_list + list(query_index[keyword].keys())
    
    return query_index, doc_list

def tf_ij_weight(query_index, i, j):
    '''
    Inputs:
      query_index
      i -- index term
      j -- document ID

    Formula:
      tf(i,j) = 1 + log(f(i,j)), if f(i,j) > 0
      tf(i,j) = 0, otherwise
      where the log is taken in base 10
    '''
    if j in query_index[i]:
        f_ij = query_index[i][j]
    else:
        return 0

    if f_ij > 0:
        tf_ij = 1 + math.log(f_ij, 10)
    else:
        tf_ij = 0

    return tf_ij

def tf_iq_weight(query_ls, i):
    return query_ls.count(i)

def idf_weight(query_index, i):
    '''Formula: idf_i = log(N/df_i)'''

    N = 45899
    df_i = len(query_index[i])
    idf_i = math.log((N/df_i), 10)

    return idf_i

def weighting_doc(tf_ij, idf_i):
    w_ij = tf_ij * idf_i
    return w_ij

def weighting_query(tf_iq, idf_i):
    w_iq = (0.5 + 0.5 * tf_iq) * idf_i
    return w_iq



















    
