from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def doc2vec_reranking(file1="docID_1000.txt", file2="wordlist_1000.txt"):
    # Get docID_ls and word_ls
    docID_ls = eval(open(file1, "r", encoding='UTF-8-sig').read())
    word_ls = eval(open(file2, "r", encoding='UTF-8-sig').read())

    tagged_data = [TaggedDocument(d, [i]) for i, d in enumerate(word_ls)]

    return tagged_data
