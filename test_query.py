import os
from bs4 import BeautifulSoup
import retrieval_ranking as rr

def test_query(file="topics_MB1-49.txt"):
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
    if os.path.exists("Results.txt"):
        os.remove("Results.txt")
    print("Starting inserting top1000 ranked docs...")
    for x in range(len(title_ls)):
        similarity_sorted, ranked_docs = rr.retrieval_ranking(title_ls[x])
        # Write top1000 ranked docs into Results.txt
        limit = 1
        for y in range(len(ranked_docs)):
            
            with open("Results.txt", "a") as file:
                file.write(f"{topicId_ls[x]} Q0 {ranked_docs[y]} {limit} {int(similarity_sorted[y][1]*1000)/1000} myRun\n")
            if limit == 1000:
                break
            limit += 1

    print("File writing finished!")
