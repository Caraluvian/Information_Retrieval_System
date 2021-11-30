import os
import torch
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, AutoModel
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

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')

    tokens = {'input_ids': [], 'attention_mask': []}

    for i in range(len(title_ls)):
        sentences = []
        sentences.append(title_ls[i])
        for j in range(len(word_ls[i])):
            sentences.append( ' '.join(word_ls[i][j]) )

        for sentence in sentences:
            
            new_tokens = tokenizer.encode_plus(sentence, max_length=128, truncation=True, padding='max_length', return_tensors='pt')
            tokens['input_ids'].append(new_tokens['input_ids'][0])
            tokens['attention_mask'].append(new_tokens['attention_mask'][0])
            print(sentence)

        tokens['input_ids'] = torch.stack(tokens['input_ids'])
        tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

        outputs = model(**tokens)
        #print(outputs.keys())

        embeddings = outputs.last_hidden_state
        print(embeddings)
    
    # with open("bert_results.txt", "a") as file:
    #     file.write(f"{embeddings}\n")
        
    print("Finished!")
if __name__ == '__main__':
    bert_reranking()