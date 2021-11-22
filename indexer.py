import tweet_tokenizer as tweet

def indexing_tokens(trec_ls=tweet.tweet_tokenizer()):
    inverted_index = {}
    # i refers to each line
    # j refers to each word
    for i in range(len(trec_ls)):
        for j in range(1, len(trec_ls[i])):
            keyword = trec_ls[i][j]
            docID = trec_ls[i][0]
            # If the keyword is not one of the keys in inverted_index dict,
            # then create a new keyword as key and empty dict as value.
            # Also, add new docID as key and add term frequency to 1 as value.
            if keyword not in inverted_index:
                # example: {"bbc":{}}
                inverted_index[keyword] = {}
                inverted_index[keyword][docID] = 1
            # If both keyword and docID exist, then term frequency increases by 1.
            elif keyword in inverted_index and docID in inverted_index[keyword]:
                inverted_index[keyword][docID] += 1
            # If only keyword exists but docID doesn't exist, then add new docID as key
            # and add term frequency to 1 as value.
            elif keyword in inverted_index and docID not in inverted_index[keyword]:
                inverted_index[keyword][docID] = 1

    return inverted_index


