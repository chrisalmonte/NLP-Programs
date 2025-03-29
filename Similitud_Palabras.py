#Palabras similares
import nltk
import os
import re
from bs4 import BeautifulSoup

#Config.
corpus = "../../Corpora/e990505_mod_lemmatized_spacy.txt"
stopwords_list = nltk.corpus.stopwords.words("spanish")
sentence_tokenizer = nltk.data.load('tokenizers/punkt_tab/spanish.pickle')
word_tokenizer = nltk.tokenize.ToktokTokenizer()

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
    sentence_tokens = sentence_tokenizer.tokenize(text)
    sentence_tokens = list(set(sentence_tokens))    
    #Tokenizar palabras
    word_tokens = set()
    sentences_processed = []
    for sentence in sentence_tokens:
        processed_sentence = []
        tokens = word_tokenizer.tokenize(sentence)
        #Filtrar tokens.
        for token in tokens:
            #Ignorar tokens sin letras
            if re.match(r"^\W$", token):
                continue
            #Ignorar tokens numéricos
            if re.match(r"^(([-+/*])*([0-9])*([,.])*([0-9])*)*$", token):
                continue            
            if re.match(r"^\d+[aoº°]$", token):
                continue            
            #Ignorar links
            if re.match(r"^(.*)\.(.*)$", token):
                continue
            #Agregar en minúsculas.
            token = token.lower()
            token = re.sub(r"[\W\-]", "", token)
            #Ignorar stopwords
            if remove_stopwords and (token in stopwords):
                continue
            processed_sentence.append(token)
            word_tokens.add(token)
        sentences_processed.append(processed_sentence)
    return (sorted(word_tokens), sentences_processed)

#Función para guardar la salida en un archivo
def save(content, filename: str, description: str = ""):
    try:
        os.mkdir("NLP_Output")
    except FileExistsError:
        pass
    except:
        return
    path = "NLP_Output/" + filename
    with open(path, 'w', encoding='utf-8') as file:
        file.write("%s\n\n" % description)
        file.write(content)
        file.write("\n")

#Generar vectores de palabras
text = str_from_file(corpus, parse_HTML=False)
#save(text, "texto_extraido.txt", "Texto sin etiquetas extraído del archivo \"%s\":" % corpus)
doc = normalize_text(text, remove_stopwords=True, stopwords=stopwords_list)
save("\n".join(sorted(stopwords_list)), "stopwords.txt", "Lista de stopwords usadas (NLTK Corpus):")
save("\n".join(doc[0]), "vocabulario.txt", "Vocabulario del corpus:")
save("\n\n".join(" ".join(sentence) for sentence in doc[1]), "oraciones.txt", "Oraciones normalizadas del corpus:")

