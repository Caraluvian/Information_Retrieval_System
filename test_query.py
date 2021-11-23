import os
from bs4 import BeautifulSoup
import retrieval_ranking as rr
import tweet_tokenizer as tweet


def dict_convertor(trec_ls=tweet.tweet_tokenizer()):

    # Create a doc dictionary with docID as key and word list as value
    doc_dict = {}
    for i in range(len(trec_ls)):
        docID = trec_ls[i][0]
        doc_dict[docID] = trec_ls[i][1:-1]
    return doc_dict

def test_query(file="topics_MB1-49.txt"):
    # Create a doc dictionary with docID as key and word list as value
    doc_dict = dict_convertor()
    
    # Read test_query file
    test_file = open(file, "r").read()
    soup = BeautifulSoup(test_file, "html.parser")

    # Extract topic id and title from test_query file
    topicId_ls = []
    for topic_ids in soup.find_all('num'):
        topic_id = topic_ids.string.split(" ")[2]
        topicId_ls.append(topic_id)

    title_ls = []
    for titles in soup.find_all('title'):
        title = titles.string.strip()
        title_ls.append(title)

    # Search 49 test queries, compute similarity and get top1000 ranked docs
    docID_ls = []
    word_ls = []
    if os.path.exists("docID_1000.txt"):
        os.remove("docID_1000.txt")
    if os.path.exists("wordlist_1000.txt"):
        os.remove("wordlist_1000.txt")
    print("Start creating docID_1000.txt and wordlist_1000.txt...")
    for x in range(len(title_ls)):
        similarity_sorted, ranked_docs = rr.retrieval_ranking(title_ls[x])
        # Write top1000 ranked docs into Results.txt
        limit = 1
        for y in range(len(ranked_docs)):
            docID_ls.append(ranked_docs[y])
            word_ls.append(doc_dict[ranked_docs[y]])
            if limit == 1000:
                break
            limit += 1
    # Created docID_1000.txt
    with open("docID_1000.txt", "a+", encoding='UTF-8-sig') as file1:
        file1.write(docID_ls)
    # Created wordlist_1000.txt
    with open("wordlist_1000.txt", "a+", encoding='UTF-8-sig') as file2:
        file2.write(word_ls)

    print("Finished!")
