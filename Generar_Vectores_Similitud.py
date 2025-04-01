#Módulo que genera los vectores de similitud de las palabras en un corpus de texto. Crea el archivo term_document_data.pkl con los vectores de similitud de las palabras en el corpus.

import nltk
import stanza
import os
import pickle
import re
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA

#Config.
OUTPUT_DIR = "NLP_Output/"
CORPUS = "../../Corpora/e990505_mod.htm"
PARSE_HTML = True
REMOVE_STOPWORDS = True
STOPWORD_LIST = nltk.corpus.stopwords.words("spanish")
#Eliminar stopwords de la lista
STOPWORD_LIST.remove("estado")
STOPWORD_LIST.extend(["ser", "hacer"])
LEMMATIZE = True
SENTENCE_TOKENIZER = nltk.data.load('tokenizers/punkt_tab/spanish.pickle')
WORD_TOKENIZER = nltk.tokenize.ToktokTokenizer()
PCA_COMPONENTS = 500
context_window = 6
nlp = stanza.Pipeline(lang="es", processors='tokenize, lemma')

#Función para extraer texto de un archivo
def str_from_file(file_path: str, file_encoding: str="utf-8", parse_HTML=True):
    with open(file_path, encoding=file_encoding) as file:
        raw = file.read()
    #Limpiar etiquetas HTML
    if parse_HTML:
        soup = BeautifulSoup(raw, "lxml")
        return soup.get_text()
    return raw

#Función para generar tokens únicos y enunciados procesados.
def normalize_text(text: str, remove_stopwords=False, stopwords: list=None):
    if remove_stopwords and stopwords is None:
        raise ValueError("No se proporcionó lista de stopwords.")
    #Tokenizar oraciones.
    #sentence_tokens = sentence_tokenizer.tokenize(text)
    #sentence_tokens = list(set(sentence_tokens))    
    #Tokenizar palabras
    doc = nlp(text)
    word_tokens = set()
    sentences_processed = []
    for sentence in doc.sentences:
        processed_sentence = []
        #tokens = word_tokenizer.tokenize(sentence)        
        tokens = [(word.lemma if LEMMATIZE else word.text) for word in sentence.words]
        #Filtrar tokens.
        for token in tokens:
            token = re.sub(r"-", "", token)
            if not token.isalpha():
                continue
            #Ignorar numeros romanos
            if re.match(r"^M{0,3}(CM|CD|D?C{0,3})?(XC|XL|L?X{0,3})?(IX|IV|V?I{0,3})?$", token):
                continue
            #Pasar a minúsculas.
            token = token.lower()
            token = re.sub(r"[\W\d_\s]", "", token)
            #Ignorar tokens vacios
            if re.match(r"^\s$", token) or len(token) < 2:
                continue 
            #Ignorar stopwords            
            if remove_stopwords and (token in stopwords):
                continue
            processed_sentence.append(token)
            word_tokens.add(token)
        if processed_sentence not in sentences_processed:
            sentences_processed.append(processed_sentence)
    return (sorted(word_tokens), sentences_processed)

#Función para guardar la salida en un archivo
def save(content, filename: str, description: str = ""):    
    path = OUTPUT_DIR + filename
    with open(path, 'w', encoding='utf-8') as file:
        file.write("%s\n\n" % description)
        file.write(content)
        file.write("\n")

#Validar directorio de salida
if not os.path.exists(OUTPUT_DIR):
    try:
        os.mkdir(OUTPUT_DIR)
    except:
        raise OSError("No se pudo crear el directorio de salida.")

#Extraer texto del archivo
text = str_from_file(CORPUS, parse_HTML=PARSE_HTML)
save(text, "texto_extraido.txt", "Texto sin etiquetas extraído del archivo \"%s\":" % CORPUS)
doc = normalize_text(text, remove_stopwords=REMOVE_STOPWORDS, stopwords=STOPWORD_LIST)
save("\n".join(sorted(STOPWORD_LIST)), "stopwords.txt", "Lista de stopwords usadas (NLTK Corpus):")
save("\n".join(doc[0]), "vocabulario.txt", "Vocabulario del corpus:")
save("\n\n".join(" ".join(sentence) for sentence in doc[1]), "oraciones.txt", "Oraciones normalizadas del corpus:")
#Extraer contexto de las palabras
context = []
for word in doc[0]:
    word_context = {}
    for sentence in doc[1]:
        if word in sentence:
            index = sentence.index(word)
            for i in range(max(0, index - int(context_window/2)), min(index + int(context_window/2) + 1, len(sentence))):
                if i != index:
                    word_context[sentence[i]] = (word_context.get(sentence[i], 0) + 1)
    context.append(word_context)
#Guardar contexto en un archivo
with open(OUTPUT_DIR + "contexto.txt", 'w', encoding='utf-8') as file:
    file.write("Contexto de las palabras en el vocabulario:\n\n")
    for i, word in enumerate(doc[0]):
        file.write("Palabra: %s\n" % word)
        file.write("Contexto:\n")
        for value in context[i]:
            file.write("\t%s: %d\n" % (str(value), context[i][value]))
        file.write("\n")
    file.write("\n")

#Generar vectores de contexto
context_vector = []
for i in range(len(doc[0])):
    vector = []
    for j in range(len(context)):
        vector.append(context[j].get(doc[0][i], 0))
    context_vector.append(vector)
context_vector_pca = PCA(n_components=PCA_COMPONENTS).fit_transform(context_vector)
context_vector_pca_data = {}
for i, word in enumerate(doc[0]):
    context_vector_pca_data[word] = context_vector_pca[i].tolist()

#Guardar vectores de contexto en un archivo
with open(OUTPUT_DIR + "term_document_data.pkl", "wb") as file:
    pickle.dump(context_vector_pca_data, file)
