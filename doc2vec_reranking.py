import json
def doc2vec_reranking(file1="docID_1000.txt", file2="wordlist_1000.txt"):
    # Using readlines()
    docID_file = open(file1, "r", encoding='UTF-8-sig').read()
    wordlist_file = open(file2, "r", encoding='UTF-8-sig').read()
    docID_ls = [i.strip()[1:-1] for i in docID_file[1:-1].split(',')]
    word_ls = [i.strip()[1:-1] for i in wordlist_file[1:-1].split(',')]

    return word_ls

