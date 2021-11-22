import nltk
from nltk.stem.porter import PorterStemmer

def tweet_tokenizer(file1="Trec_microblog11.txt", file2="StopWords.txt"):
    # Using readlines()
    trec_file = open(file1, "r", encoding='UTF-8-sig')
    stopword_file = open(file2, "r", encoding='UTF-8-sig')
    trec_lines = trec_file.readlines()
    stopword_lines = stopword_file.readlines()

    # 1-D list for stopword_file
    stopword_ls = []
    for line in stopword_lines:
        stopword_ls.append(line.strip())

    # 2-D list for trec_file

    trec_ls = []
    for line in trec_lines:
        trec_ls.append(line.strip().split())

    for line in range(len(trec_ls)):
        for i in range(len(trec_ls[line])):
            # Lowercase all the words
            trec_ls[line][i] = lowercase_text(trec_ls[line][i])
            # Remove punctuations
            trec_ls[line][i] = remove_punctuations(trec_ls[line][i])
            # Remove numbers
            if i != 0:
                trec_ls[line][i] = remove_number(trec_ls[line][i])
            # Remove stopword
            trec_ls[line][i] = remove_stopword(trec_ls[line][i], stopword_ls)
            # Stem index words
            #trec_ls[line][i] = porter_stemming(trec_ls[line][i])
        trec_ls[line] = [string for string in trec_ls[line] if string != ""]

    return trec_ls

def lowercase_text(word):
    new_word = word.lower()
    return new_word

def remove_punctuations(word):
    new_word = ""
    punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~…=|`+♥【】►ˇ「」♫♪┌П┐‘’“”。→！？«»◆£€☹☑•ﾟ∀★´`█▬▀•＞¿♦▽♺☮❥☃ ☛†®『』～♡②⇒☉④①®◣◢☀￼™☒✈➙➔▪✔✿ⓂⓄⓀⒺⒹⒾⓉ√❤⇒ⒶⓇⓄ▷☞✦✖ⓄⒿⒺⒻⓇⒺⓎⒼⓄⓁⒹⒷⒺⓇⒼ♻ⓌⓄⓇⓁⒹⒿⒺⓌⒾⓈⒽⒸⓄⓃⓇⒺⓈⓥ☆✘✍━▸'''
    for i in range(len(word)):
        if word[i] not in punctuations:
            new_word = new_word + word[i]
    return new_word

def remove_number(word):
    new_word = ""
    for i in range(len(word)):
        if word[i] not in "0123456789":
            new_word = new_word + word[i]
    return new_word

def remove_stopword(word, stopword_ls):
    if word in stopword_ls:
        return ""
    else:
        return word

def porter_stemming(word):
    porter_stemmer = PorterStemmer()
    new_word = porter_stemmer.stem(word)
    return new_word
