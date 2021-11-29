import os
import nltk
nltk.download('punkt')
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from gensim.test.utils import get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def doc2vec_reranking(file1="docID_1000.txt", file2="wordlist_1000.txt", file3="topics_MB1-49.txt"):
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


    # Search 49 test queries, compute similarity and get top1000 ranked docs
    if os.path.exists("doc2vec_results.txt"):
        os.remove("doc2vec_results.txt")
    print("Starting inserting top1000 ranked docs...")

    for i in range(len(title_ls)):

        document = word_ls[i]
        print(len(document))
        tagged_doc = [TaggedDocument(d, [i]) for i, d in enumerate(document)]

        ## Train doc2vec model
        model = Doc2Vec(tagged_doc, vector_size=5, window=2, min_count=20, negative=0, workers=6, dm=1, epochs=10, alpha=0.025)
        # Save trained doc2vec model
        model.save("test_doc2vec.model")
        ## Load saved doc2vec model
        model= Doc2Vec.load("test_doc2vec.model")

        # find most similar doc
        test_doc = word_tokenize(title_ls[i].lower())
        reranked_list = model.dv.most_similar(positive=[model.infer_vector(test_doc)], topn=1000)

        for j in range(len(reranked_list)):
            with open("doc2vec_results.txt", "a") as file:
                file.write(f"{topicId_ls[i]} Q0 {docID_ls[i][reranked_list[j][0]]} {j+1} {int(reranked_list[j][1]*1000)/1000} myRun\n")
    print("Finished!")

if __name__ == '__main__':
    doc2vec_reranking()