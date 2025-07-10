import nltk
import os
import re
import stanza
import numpy as np
import math
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import unicodedata

# modules
import picklejar
import textNormalization

#Config.
OUTPUT_DIR = "SA-Output/"
CORPUS_DIR = "C:/Users/emith/Downloads/corpusCriticasCine/"
TEST_SAMPLE_RATIO = 0.8
STOPWORD_LIST = nltk.corpus.stopwords.words("spanish")
#Eliminar stopwords de la lista
STOPWORD_LIST.remove("estado")
STOPWORD_LIST.extend(["ser", "hacer", "haber"])
SKIP_TEXT_NORMALIZATION = True
SKIP_POLARITY_ANALYSIS = True
nlp = stanza.Pipeline(lang="es", processors='tokenize, lemma, mwt, pos') if not SKIP_TEXT_NORMALIZATION else None

output = picklejar.Jar(OUTPUT_DIR)

if SKIP_TEXT_NORMALIZATION:
    documents = output.load_pickle("documents.pkl")
    tags = output.load_pickle("tags.pkl")
    unique_tokens = output.load_pickle("unique_tokens.pkl")
else:
    sample_size = 4381
    tags = []
    documents = []
    unique_tokens = set()
    for i in range(1, sample_size + 1):
        print("Processing Document %d" % i)
        if not os.path.exists("%s%d.xml" % (CORPUS_DIR, i)) or not os.path.exists("%s%d.review.pos" % (CORPUS_DIR, i)): 
            continue
        tag = None
        doc = []
        with open("%s%d.xml" % (CORPUS_DIR, i), 'r', encoding="ANSI") as file:
            content = file.read()
            tag = re.search(r"rank=\"(\d)\"", content)
            #doc = re.search(r"<body>(.*)</body>", content, re.DOTALL)
        #if not tag or not doc:
        #    continue
        #doc = nlp(doc.group(1))
        #normalized_doc = []
        #for sentence in doc.sentences:
        #    tokens = [word.lemma for word in sentence.words]
        #    for token in tokens:
        #        token = re.sub(r"-", "", token)
        #        if not token.isalpha():
        #            continue
        #        token = token.lower()
        #        token = re.sub(r"[\W\d_\s]", "", token)
        #        if re.match(r"^\s$", token) or len(token) < 2:
        #            continue            
        #        if token in STOPWORD_LIST:
        #            continue
        #        normalized_doc.append(token)
        #        unique_tokens.add(token)   
        with open("%s%d.review.pos" % (CORPUS_DIR, i), 'r', encoding="ANSI", newline='') as file:
            for line in file:
                line = line.rstrip()
                word_info = line.split()
                if not word_info or len(word_info) < 4:
                    continue
                if not word_info[1].isalpha() or len(word_info[1]) < 2:
                    continue
                if word_info[1] in STOPWORD_LIST:
                    continue
                doc.append(word_info[1])
                unique_tokens.add(word_info[1])
        if not tag or not doc:
            continue      
        tags.append(int(tag.group(1)))
        documents.append(doc)
    unique_tokens = sorted(unique_tokens)
    output.save_list_2_txt("unique_tokens.txt", unique_tokens)    
    output.save_pickle("unique_tokens.pkl", unique_tokens)
    output.save_pickle("documents.pkl", documents)
    output.save_pickle("tags.pkl", tags)

print("Vocabulary Length:", len(unique_tokens))
print("Sample Size:", len(documents))
print("Size of y:", len(tags))

#Frequencies:
if SKIP_TEXT_NORMALIZATION:
    token_raw_freq = output.load_pickle("token_raw_freq.pkl")
else:
    token_raw_freq = []
    for i,doc in enumerate(documents):
        print("Getting frequencies of document ", i)
        freq = 0
        raw_freq = []
        for token in unique_tokens:
            raw_freq.append(sum(1 if token == word else 0 for word in doc))
        token_raw_freq.append(raw_freq)
    output.save_pickle("token_raw_freq.pkl", token_raw_freq)

#Train/Test Split
#X_train, X_test, y_train, y_test = train_test_split(token_raw_freq, tags, test_size=TEST_SAMPLE_RATIO, random_state=999)
#
##Model Training
#clasifier = make_pipeline(StandardScaler(), PCA(n_components=min(len(X_train), len(X_train[0]))), SGDClassifier(max_iter=50000, tol=1e-3))
#clasifier.fit(X_train, y_train)
#y_predicted = clasifier.predict(X_test)
#report = classification_report(y_test, y_predicted) + '\n' + str(confusion_matrix(y_test, y_predicted)) + '\n'
#print(report)
#output.save_text(report, "classification_report.txt")

#Senticon analysis
doc_polarity_scores = []
doc_polarity_avg_full = []
doc_polarity_avg_medium = []

if SKIP_POLARITY_ANALYSIS:
    doc_polarity_scores = output.load_pickle("doc_polarity_scores.pkl")
    doc_polarity_avg_full = output.load_pickle("doc_polarity_avg_full.pkl")
    doc_polarity_avg_medium = output.load_pickle("doc_polarity_avg_medium.pkl")
    print("Polarity scores loaded.")
    print("Length of polarity scores:", len(doc_polarity_scores))
    print("Length of polarity avg full:", len(doc_polarity_avg_full))
    print("Length of polarity avg medium:", len(doc_polarity_avg_medium))
else:
    senticon_continious = output.load_pickle("senticon_continious.pkl")
    senticon_discrete_full = output.load_pickle("senticon_discrete.pkl")
    senticon_discrete_medium = output.load_pickle("senticon_discrete_medium.pkl")

    print("Length of continous senticon:", len(senticon_continious))
    print("Length of discrete senticon (full strength):", len(senticon_discrete_full))
    print("Discrete values:", set(senticon_discrete_full.values()))
    print("Length of discrete senticon (medium strength):", len(senticon_discrete_medium))
    print("Discrete medium values:", set(senticon_discrete_medium.values()))

    for i, doc in enumerate(documents):
        print("Getting polarity of document ", i)
        polarity_score = 0
        polarity_avg_full = {}
        polarity_avg_med = {}
        score_count = 0
        avg_med_count = 0
        avg_full_count = 0
        for token in doc:
            normalized = unicodedata.normalize('NFKD', token)  
            token_no_diacritic = ''.join([c for c in normalized if not unicodedata.combining(c)])
            if token in senticon_continious:
                polarity_score += senticon_continious[token]
                score_count += 1
            if token in senticon_discrete_full:
                polarity = senticon_discrete_full[token_no_diacritic]
                polarity_avg_full[polarity] = polarity_avg_full.get(polarity, 0) + 1
                avg_full_count += 1
            if token in senticon_discrete_medium:
                polarity = senticon_discrete_medium[token_no_diacritic]
                polarity_avg_med[polarity] = polarity_avg_med.get(polarity, 0) + 1
                avg_med_count += 1    
        #Real polarity score
        doc_polarity_scores.append(polarity_score / score_count if score_count > 0 else 0)

        #Discrete full
        positive = polarity_avg_full.get("pos", 0)
        negative = polarity_avg_full.get("neg", 0)
        positive = positive/avg_full_count if avg_full_count > 0 else 0
        negative = negative/avg_full_count if avg_full_count > 0 else 0
        doc_polarity_avg_full.append([positive, negative])

        #Discrete medium
        positive = polarity_avg_med.get("pos", 0)
        negative = polarity_avg_med.get("neg", 0)
        positive = positive/avg_med_count if avg_med_count > 0 else 0
        negative = negative/avg_med_count if avg_med_count > 0 else 0
        doc_polarity_avg_medium.append([positive, negative])
    output.save_pickle("doc_polarity_scores.pkl", doc_polarity_scores)
    output.save_pickle("doc_polarity_avg_full.pkl", doc_polarity_avg_full)
    output.save_pickle("doc_polarity_avg_medium.pkl", doc_polarity_avg_medium)     
    print("Polarity scores saved.")

#Average polarity scores [count, continous, discrete full pos, discrete full neg, discrete medium por, discrete medium neg]
avg_1 = [0, 0, 0, 0, 0, 0]
avg_2 = [0, 0, 0, 0, 0, 0]
avg_3 = [0, 0, 0, 0, 0, 0]
avg_4 = [0, 0, 0, 0, 0, 0]
avg_5 = [0, 0, 0, 0, 0, 0]

for i, rank in enumerate(tags):
    match rank:
        case 1:
            avg_1[0] += 1
            avg_1[1] = math.fsum([avg_1[1], doc_polarity_scores[i]])
            avg_1[2] = math.fsum([avg_1[2], doc_polarity_avg_full[i][0]])
            avg_1[3] = math.fsum([avg_1[3], doc_polarity_avg_full[i][1]])
            avg_1[4] = math.fsum([avg_1[4], doc_polarity_avg_medium[i][0]])
            avg_1[5] = math.fsum([avg_1[5], doc_polarity_avg_medium[i][1]])
        case 2:
            avg_2[0] += 1
            avg_2[1] = math.fsum([avg_2[1], doc_polarity_scores[i]])
            avg_2[2] = math.fsum([avg_2[2], doc_polarity_avg_full[i][0]])
            avg_2[3] = math.fsum([avg_2[3], doc_polarity_avg_full[i][1]])
            avg_2[4] = math.fsum([avg_2[4], doc_polarity_avg_medium[i][0]])
            avg_2[5] = math.fsum([avg_2[5], doc_polarity_avg_medium[i][1]])
        case 3:
            avg_3[0] += 1
            avg_3[1] = math.fsum([avg_3[1], doc_polarity_scores[i]])
            avg_3[2] = math.fsum([avg_3[2], doc_polarity_avg_full[i][0]])
            avg_3[3] = math.fsum([avg_3[3], doc_polarity_avg_full[i][1]])
            avg_3[4] = math.fsum([avg_3[4], doc_polarity_avg_medium[i][0]])
            avg_3[5] = math.fsum([avg_3[5], doc_polarity_avg_medium[i][1]])
        case 4:
            avg_4[0] += 1
            avg_4[1] = math.fsum([avg_4[1], doc_polarity_scores[i]])
            avg_4[2] = math.fsum([avg_4[2], doc_polarity_avg_full[i][0]])
            avg_4[3] = math.fsum([avg_4[3], doc_polarity_avg_full[i][1]])
            avg_4[4] = math.fsum([avg_4[4], doc_polarity_avg_medium[i][0]])
            avg_4[5] = math.fsum([avg_4[5], doc_polarity_avg_medium[i][1]])
        case 5:
            avg_5[0] += 1
            avg_5[1] = math.fsum([avg_5[1], doc_polarity_scores[i]])
            avg_5[2] = math.fsum([avg_5[2], doc_polarity_avg_full[i][0]])
            avg_5[3] = math.fsum([avg_5[3], doc_polarity_avg_full[i][1]])
            avg_5[4] = math.fsum([avg_5[4], doc_polarity_avg_medium[i][0]])
            avg_5[5] = math.fsum([avg_5[5], doc_polarity_avg_medium[i][1]])
        case _:
            print("Document %d has an invalid rank: %d" % (i, rank))
    
    for i in range(1, 6):
        if not avg_1[0] > 0:
            break
        avg_1[i] = np.divide(avg_1[i], avg_1[0])
    for i in range(1, 6):
        if not avg_2[0] > 0:
            break
        avg_2[i] = np.divide(avg_2[i], avg_2[0])
    for i in range(1, 6):
        if not avg_3[0] > 0:
            break
        avg_3[i] = np.divide(avg_3[i], avg_3[0])
    for i in range(1, 6):
        if not avg_4[0] > 0:
            break
        avg_4[i] = np.divide(avg_4[i], avg_4[0])
    for i in range(1, 6):
        if not avg_5[0] > 0:
            break
        avg_5[i] = np.divide(avg_5[i], avg_5[0])

scores = "Average Polarity Scores:\n"
scores += "Continous:\n"
scores += "1 Star:%.20f\n" % avg_1[1]
scores += "2 Stars:%.20f\n" % avg_2[1]
scores += "3 Stars:%.20f\n" % avg_3[1]
scores += "4 Stars:%.20f\n" % avg_4[1]
scores += "5 Stars:%.20f\n" % avg_5[1]
scores += "\nDiscrete Full Stength (Positive | Negative):\n"
scores += "1 Star: %.20f | %.20f\n" % (avg_1[2], avg_1[3])
scores += "2 Stars: %.20f | %.20f\n" % (avg_2[2], avg_2[3])
scores += "3 Stars: %.20f | %.20f\n" % (avg_3[2], avg_3[3])
scores += "4 Stars: %.20f | %.20f\n" % (avg_4[2], avg_4[3])
scores += "5 Stars: %.20f | %.20f\n" % (avg_5[2], avg_5[3])
scores += "\nDiscrete Medium Stength (Positive | Negative):\n"
scores += "1 Star: %.20f | %.20f\n" % (avg_1[4], avg_1[5])
scores += "2 Stars: %.20f | %.20f\n" % (avg_2[4], avg_2[5])
scores += "3 Stars: %.20f | %.20f\n" % (avg_3[4], avg_3[5])
scores += "4 Stars: %.20f | %.20f\n" % (avg_4[4], avg_4[5])
scores += "5 Stars: %.20f | %.20f\n" % (avg_5[4], avg_5[5])
print(scores)
output.save_text(scores, "polarity_scores.txt")
print("Report Saved")