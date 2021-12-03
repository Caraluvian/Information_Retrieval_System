import gensim
import test_query
from test_query import query_list
from bs4 import BeautifulSoup
from collections import Counter
from gensim.models import Word2Vec

def word2vec_reranking (file1="docID_1000.txt", file2="wordlist_1000.txt", file3="topics_MB1-49.txt"):

    # Read test_query file
    test_file = open(file3, "r").read()
    soup = BeautifulSoup(test_file, "html.parser")

    # Extract topic id and title from test_query file
    topicId_ls = []
    for topic_ids in soup.find_all('num'):
        topic_id = topic_ids.string.split(" ")[2]
        if topic_id[3] == "0":
            topicId_ls.append(topic_id[4:])
        else:
            topicId_ls.append(topic_id[3:])

    title_ls = []
    for titles in soup.find_all('title'):
        title = titles.string.strip()
        title_ls.append(title)

    # Get docID_ls and word_ls
    docID_ls = eval(open(file1, "r", encoding='UTF-8-sig').read())
    word_ls = eval(open(file2, "r", encoding='UTF-8-sig').read())

    synonym_words = []
    for i in range(len(title_ls)):

        # initialize an empty dict for every query with the docID as key and similarities as values after reranking
        synonym_keys = []

        # initialize the sentences and add the query as the first sentence
        sentences = word_ls[i]

        #print(sentences)
        #sentences = [['first', 'sentence'], ['second', 'sentence']]
        
        model = Word2Vec(sentences, min_count=2)
        
        # Convert query string to query list (tokenized)
        query_ls = query_list(title_ls[i])
        synonym_words_tmp = []
        try:
            for query_word in query_ls:
                # type(sim_words) is list
                sim_words = model.wv.most_similar(query_word, topn=5)
                
                for tuple in sim_words:
                    synonym_words_tmp.append(tuple[0])
            # type(synonym_words) is list ["abc","cde"]
            synonym_keys = list(dict(filter(lambda a:a[1]>1, Counter(synonym_words_tmp).items())).keys())
            
            synonym_words.append(synonym_keys)
        except:
            synonym_words.append([])

    # Add synonym_words to the original query and write Top1000 ranked docs into word2vec_results.
    test_query.test_query(synonym_words)




if __name__ == '__main__':
    word2vec_reranking()