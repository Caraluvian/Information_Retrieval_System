import os
from bs4 import BeautifulSoup
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

def bert_reranking (file1="docID_1000.txt", file2="wordlist_1000.txt", file3="topics_MB1-49.txt"):

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

    if os.path.exists("bert_results.txt"):
        os.remove("bert_results.txt")
    print("Start inserting top1000 ranked docs...")

    model = SentenceTransformer('bert-base-nli-mean-tokens')

    for i in range(len(title_ls)):

        # initialize an empty dict for every query with the docID as key and similarities as values after reranking
        reranked_dict = {}

        # initialize the sentences and add the query as the first sentence
        sentences = []
        sentences.append(title_ls[i])
        for j in range(len(word_ls[i])):
            sentences.append( ' '.join(word_ls[i][j]) )
        
        sentence_embeddings = model.encode(sentences)

        #compute the cosine_similarity
        reranked_similarity = cosine_similarity([sentence_embeddings[0]], sentence_embeddings[1:])

        # add docID as keys and similarities as value to the reranked dictionary for each query
        for x in range(len(reranked_similarity[0])):
            reranked_dict[docID_ls[i][x]] = reranked_similarity[0][x]

        # sort the reranked dictionary by similarity as list of tuples
        sorted_list = sorted(reranked_dict.items(), key = lambda kv:(kv[1], kv[0]), reverse=True)

        # write the results into files
        for y in range(len(sorted_list)):
            with open("bert_results.txt", "a") as file:
                    file.write(f"{topicId_ls[i]} Q0 {sorted_list[y][0]} {y+1} {int(sorted_list[y][1]*1000)/1000} myRun\n")
        
    print("Finished!")
if __name__ == '__main__':
    bert_reranking()