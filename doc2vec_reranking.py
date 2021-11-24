from gensim.models.doc2vec import Doc2Vec, TaggedDocument
import nltk
from nltk.tokenize import word_tokenize

def doc2vec_reranking(file1="docID_1000.txt", file2="wordlist_1000.txt"):
    # Get docID_ls and word_ls
    docID_ls = eval(open(file1, "r", encoding='UTF-8-sig').read())
    word_ls = eval(open(file2, "r", encoding='UTF-8-sig').read())

    tagged_doc = [TaggedDocument(d, [i]) for i, d in enumerate(word_ls)]

    ## Train doc2vec model
    model = Doc2Vec(tagged_doc, vector_size=100, window=2, min_count=1, workers=4)
    # Save trained doc2vec model
    model.save("test_doc2vec.model")
    ## Load saved doc2vec model
    model= Doc2Vec.load("test_doc2vec.model")

    # find most similar doc 
    test_doc = word_tokenize("BBC World Service staff cuts".lower())
    print(model.docvecs.most_similar(positive=[model.infer_vector(test_doc)]))
    #print(model.dv.most_similar(positive=[model.infer_vector(test_doc)]))

    #print(docID_ls[3075])
