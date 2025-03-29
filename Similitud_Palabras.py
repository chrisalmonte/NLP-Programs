#Palabras similares
import nltk
import os
import re
from bs4 import BeautifulSoup
#nltk.download('punkt_tab')

#Config.
corpus = "../Corpora/e990505_mod_lemmatized_spacy.txt"
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

#Función para generar tokens únicos
def gen_vocab(text: str):
    #Tokenizar oraciones.
    sentence_tokens = sentence_tokenizer.tokenize(text)
    sentence_tokens = list(set(sentence_tokens))
    #Tokenizar palabras
    word_tokens = set()
    for sentence in sentence_tokens:
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
            word_tokens.add(token)
    return sorted(list(word_tokens))

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
        #for obj in content:
        #    file.write(str(obj) + "\n")
        file.write("\n")

#Generar vectores de palabras
text = str_from_file(corpus, parse_HTML=False)
#save(text, "texto_extraido.txt", "Texto sin etiquetas extraído del archivo \"%s\":" % corpus) 
vocab = gen_vocab(text)
save("\n".join(vocab), "vocabulario.txt", "Vocabulario del corpus:")

#Recopilar Vocabulario
#Tokenizar (Existen varios tokenizers, incluso dentro de nltk)
#words = nltk.word_tokenize(text)