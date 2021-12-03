import os
import tweet_tokenizer as tweet
from bs4 import BeautifulSoup
import retrieval_ranking as rr

def dict_convertor(trec_ls=tweet.tweet_tokenizer()):

    # Create a doc dictionary with docID as key and word list as value
    doc_dict = {}
    for i in range(len(trec_ls)):
        docID = trec_ls[i][0]
        doc_dict[docID] = trec_ls[i][1:]
    return doc_dict

def query_list(query):
    # Stopword list
    stopword_lines = open("StopWords.txt", "r", encoding='UTF-8-sig').readlines()
    stopword_ls = []
    for line in stopword_lines:
        stopword_ls.append(line.strip())
    # Query tokenization
    query_ls = query.split()
    for i in range(len(query_ls)):
        query_ls[i] = tweet.lowercase_text(query_ls[i])
        query_ls[i] = tweet.remove_punctuations(query_ls[i])
        query_ls[i] = tweet.remove_number(query_ls[i])
        query_ls[i] = tweet.remove_stopword(query_ls[i], stopword_ls)
    query_ls = [string for string in query_ls if string != ""]

    return query_ls

def test_query(synonym_words=[], file="topics_MB1-49.txt"):

    # Create a doc dictionary with docID as key and word list as value
    doc_dict = dict_convertor()

    # Read test_query file
    test_file = open(file, "r").read()
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

    # Search 49 test queries, compute similarity and get top1000 ranked docs
    docID_ls = []
    word_ls = []
    
    if os.path.exists("word2vec_results.txt"):
        os.remove("word2vec_results.txt")
    if os.path.exists("docID_1000.txt"):
        os.remove("docID_1000.txt")
    if os.path.exists("wordlist_1000.txt"):
        os.remove("wordlist_1000.txt")
    print("Start creating word2vec_results...")


    # Search 49 test queries, compute similarity and get top1000 ranked docs
    if os.path.exists("Results.txt"):
        os.remove("Results.txt")
    print("Starting inserting top1000 ranked docs...")

    for x in range(len(title_ls)):
        query_ls = query_list(title_ls[x])
        similarity_sorted, ranked_docs = rr.retrieval_ranking(query_ls)

        for id in ranked_docs[:5]:    
            new_query = query_ls + doc_dict[id]

        new_query = list(dict.fromkeys(new_query))
        similarity_sorted, ranked_docs = rr.retrieval_ranking(new_query)

        # Write top1000 ranked docs into Results.txt
        limit = 1
        for y in range(len(ranked_docs)):

            with open("Results.txt", "a") as file:
                file.write(f"{topicId_ls[x]} Q0 {ranked_docs[y]} {limit} {int(similarity_sorted[y][1]*1000)/1000} myRun\n")
            if limit == 1000:
                break
            limit += 1

    # Search 49 test queries, compute similarity and get top1000 ranked docs


    for x in range(len(title_ls)):
        query_ls = query_list(title_ls[x])

        #add synonyms to query if we use word2vec
        if synonym_words != []:
            query_ls = query_ls + synonym_words[x]
        similarity_sorted, ranked_docs = rr.retrieval_ranking(query_ls)

        # Write top1000 ranked docs into word2vec_results.txt
        limit = 1
        docID_ls_tmp = []
        word_ls_tmp = []
        for y in range(len(ranked_docs)):
            docID_ls_tmp.append(ranked_docs[y])
            word_ls_tmp.append(doc_dict[ranked_docs[y]])

            #wirte result to word2vec files 
            if synonym_words != []:
                with open("word2vec_results.txt", "a") as file:
                    file.write(f"{topicId_ls[x]} Q0 {ranked_docs[y]} {limit} {int(similarity_sorted[y][1]*1000)/1000} myRun\n")
            if limit == 1000:
                break
            limit += 1
        
        docID_ls.append(docID_ls_tmp)
        word_ls.append(word_ls_tmp)

    if synonym_words != []:
        print("Finished writing word2vec_results.txt!")

    print("Start creating docID_1000.txt and wordlist_1000.txt...")
    
    # Created docID_1000.txt
    with open("docID_1000.txt", "a+", encoding='UTF-8-sig') as file1:
        file1.write(f"{docID_ls}")
    # Created wordlist_1000.txt
    with open("wordlist_1000.txt", "a+", encoding='UTF-8-sig') as file2:
        file2.write(f"{word_ls}")

    print("Finished writing docID_1000.txt and wordlist_1000.txt!")

if __name__ == '__main__':
    test_query()
