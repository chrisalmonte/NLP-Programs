#Módulo que clasifica textos como spam o ham usando regresión logística
import nltk
import numpy as np
import re
from matplotlib import pyplot as plt

#Módulos propios
import picklejar
import textNormalization

#Config.
CORPUS = "../../Corpora/SMS_Spam_Corpus_big.txt"
OUTPUT_DIR = "Spam_Classification/"
WORD_TOKENIZER = nltk.tokenize.TweetTokenizer()
SKIP_TXT_NORMALIZATION = False
MAX_EPOCHS = 500000
MAX_MEAN_ERROR = 0.01
ALPHA = 1.3
lemmatizer = nltk.stem.WordNetLemmatizer()

output = picklejar.Jar(OUTPUT_DIR)
corpus = textNormalization.Operations.str_from_file(CORPUS, file_encoding="ANSI", parse_HTML=False)

document = corpus.split("\n")
document_normalized = []
y = []
vocabulary = set()
frequencies = []

if SKIP_TXT_NORMALIZATION:
     document_normalized = output.load_pickle("document_normalized.pkl")
     y = output.load_pickle("document_y.pkl")
     vocabulary = output.load_pickle("document_vocabulary.pkl")
     frequencies = output.load_pickle("document_frequencies.pkl")
else:
    for i, sms in enumerate(document):
        if sms.endswith(",spam"):
            y.append(1)
            document[i] = sms[:-5]
        else:
            y.append(0)
            document[i] = sms[:-4]

        doc = WORD_TOKENIZER.tokenize(document[i])
        sms_normalized = []
        for token in doc:
            if re.match(r"^\s$", token):
                    continue
            if token[0].isnumeric() and token[-1].isnumeric():
                token = token.replace(',', '')
                token = token.replace(' ', '')
            if token.isnumeric():
                    token = "#SMALL_NUMBER#" if len(token) < 3 else "#LARGE_NUMBER#"
                    vocabulary.add(token)
                    sms_normalized.append(token)
                    continue
            token = lemmatizer.lemmatize(token.lower())
            vocabulary.add(token)
            sms_normalized.append(token)
        document_normalized.append(sms_normalized)
    output.save_text("\n\n".join(" ".join(sentence) for sentence in document_normalized), "document_normalized.txt", "Texto normalizado: ")
    output.save_pickle("document_normalized.pkl", document_normalized)
    output.save_pickle("document_vocabulary.pkl", vocabulary)
    output.save_pickle("document_y.pkl", y)

    for word in vocabulary:
        features = []
        for sms in document_normalized:
            features.append(sms.count(word))
        frequencies.append(features)
    output.save_pickle("document_frequencies.pkl", frequencies)

#Regresión Logística
W = np.zeros(len(vocabulary))
W.reshape(len(vocabulary), 1)
X = np.array(frequencies)
y = np.array(y)
y_gorrito = []
J = []

#para evitar underflow en lugar de sigmoid(logit) es sigmoid(logit - maximo de logic)

for i in range(MAX_EPOCHS):
    logit = np.dot(W.T, X)
    y_gorrito = np.exp(logit) / (1 + np.exp(logit))
    W = W - ALPHA * 1/len(document_normalized) * np.dot(X, np.transpose(y_gorrito - y))
    j = y * np.log(y_gorrito) - (1 - y) * np.log(1 - y_gorrito)
    J.append(np.sum(j))
    print(i, J[i])
    if np.mean(j) < MAX_MEAN_ERROR:
        break
output.save_list_2_txt("J.txt", J, "Costos de la regresión logística en cada íteración.", separator="\n")
output.save_list_2_txt("y_circumfleja.txt", y_gorrito, "Valores finales de y gorrito.", separator="\n")
output.save_pickle("W.pkl", W)
output.save_pickle("J.pkl", J)

#Graficar
t = np.linspace(0, len(J), len(J))
plt.plot(t, J)
plt.xlabel("Iteración")
plt.ylabel("Costo")
plt.title("Curva de Aprendizaje")
plt.savefig(OUTPUT_DIR + "curva_aprendizaje.png")
#plt.show()

#Métricas:
FP = 0
FN = 0
TP = 0
TN = 0
for i in range(len(y)):
    if y[i] == 1 and y_gorrito[i] >= 0.8:
        TP += 1
    elif y[i] == 1 and y_gorrito[i] < 0.8:
        FN += 1
    elif y[i] == 0 and y_gorrito[i] >= 0.8:
        FP += 1
    else:
        TN += 1
#Imprimir Tabla
print("Matriz de Confusión:")
metrics = "".ljust(10) + "SPAM".ljust(10) + "HAM".ljust(10) + "\n"
metrics += "SPAM".ljust(10) + str(TP).ljust(10) + str(FN).ljust(10) + "\n"
metrics += "HAM".ljust(10) + str(FP).ljust(10) + str(TN).ljust(10) + "\n\n"
metrics += "Specificity: %.6f\n" % (TN / (TN + FP))
metrics += "Precision: %.6f\n" % (TP / (TP + FP))
metrics += "Recall: %.6f\n" % (TP / (TP + FN))
metrics += "F1: %.6f\n" % (2 * TP / (2 * TP + FP + FN))

print(metrics)
output.save_text(metrics, "matriz_confusion.txt", "Métricas: ")