#Módulo que genera los vectores de similitud de las palabras en un corpus de texto.

import math
import nltk
import stanza
import re
from bs4 import BeautifulSoup
from sklearn.decomposition import PCA

#Módulos propios
import nlpmanager
import picklejar

#Re-analyse text?
process_text = False

#Config.
OUTPUT_DIR = "NLP_Output/"
CORPUS = "../../Corpora/e990505_mod.htm"
PARSE_HTML = True
REMOVE_STOPWORDS = True
STOPWORD_LIST = nltk.corpus.stopwords.words("spanish")
#Eliminar stopwords de la lista
STOPWORD_LIST.remove("estado")
STOPWORD_LIST.extend(["ser", "hacer", "haber"])
LEMMATIZE = True
SENTENCE_TOKENIZER = nltk.data.load('tokenizers/punkt_tab/spanish.pickle')
WORD_TOKENIZER = nltk.tokenize.ToktokTokenizer()
PCA_COMPONENTS = 500
context_window = 8
nlp = stanza.Pipeline(lang="es", processors='tokenize, lemma, mwt, pos') if process_text else None

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
    word_frequencies = {}
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
            if re.match(r"^m{0,3}(cm|cd|d?c{0,3})?(xc|xl|l?X{0,3})?(ix|iv|v?i{0,3})?$", token):
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
            word_frequencies[token] = word_frequencies.get(token, 0) + 1
        if processed_sentence not in sentences_processed:
            sentences_processed.append(processed_sentence)
    return (sorted(word_tokens), sentences_processed, word_frequencies)

#Función para aplicar PCA
def apply_PCA(data: list, componets: int = PCA_COMPONENTS):
    matrix_pca = PCA(n_components=componets).fit_transform(freq_raw)
    dict_pca = {}
    for i, word in enumerate(doc_normalized.unique_tokens):
        dict_pca[word] = matrix_pca[i].tolist()
    return dict_pca

#Función IDF_BM25
def idf_bm25(b=0.75, k=1.2):
    bm25_list = []
    avg_doc_len = sum(word_context.length for word_context in context_objs) / len(context_objs)
    for word in doc_normalized.unique_tokens:
        idf_k = sum(1 if word in word_context.context_bag else 0 for word_context in context_objs)
        vector = []
        for word_context in context_objs:
            c = word_context.getCount(word)
            bm25_word_doc = ((k+1)*c)/(c + k*(1-b+b*(word_context.length/avg_doc_len)))
            vector.append(bm25_word_doc * math.log(1 if idf_k == 0 else (1 + len(context_objs))/idf_k))
        bm25_list.append(vector)
    return bm25_list

def log(num):
    return 0 if num == 0 else math.log(num, 2)

#Inicio del Programa

#Indicar un directorio de salida
output = picklejar.Jar(OUTPUT_DIR)

if process_text:
    #Extraer texto del archivo
    text = str_from_file(CORPUS, parse_HTML=PARSE_HTML)
    doc = normalize_text(text, remove_stopwords=REMOVE_STOPWORDS, stopwords=STOPWORD_LIST)
    doc_normalized = nlpmanager.DocProperties(doc[0], doc[1], doc[2])
    output.save_text(text, "texto_extraido.txt", "Texto sin etiquetas extraído del archivo \"%s\":" % CORPUS)
    output.save_text("\n".join(sorted(STOPWORD_LIST)), "stopwords.txt", "Lista de stopwords usadas (NLTK Corpus):")
    output.save_text("\n".join(doc_normalized.unique_tokens), "vocabulario.txt", "Vocabulario del corpus:")
    output.save_text("\n\n".join(" ".join(sentence) for sentence in doc_normalized.sentences), "oraciones.txt", "Oraciones normalizadas del corpus:")

    #Extraer contexto de las palabras
    context = []
    context_objs = []
    for word in doc_normalized.unique_tokens:
        word_context = {}
        for sentence in doc_normalized.sentences:
            if word in sentence:
                index = sentence.index(word)
                for i in range(max(0, index - int(context_window/2)), min(index + int(context_window/2) + 1, len(sentence))):
                    if i != index:
                        word_context[sentence[i]] = (word_context.get(sentence[i], 0) + 1)
        context_objs.append(nlpmanager.WordProperties(word, word_context, corpus_frequency=doc_normalized.word_frequencies[word]))
        context.append(word_context)
    doc_normalized.set_contexts(context)
    output.save_pickle("doc_normalized.pkl", doc_normalized)
    output.save_pickle("context_objs.pkl", context_objs)

    #Guardar frecuencias en un archivo
    with open(OUTPUT_DIR + "frecuencias.txt", 'w', encoding='utf-8') as file:
        file.write("Frecuencias de las palabras en el vocabulario:\n\n")
        for token in doc_normalized.unique_tokens:
            file.write(token.ljust(20) + ": " + str(doc_normalized.word_frequencies[token]) + "\n")

    #Guardar contexto en un archivo
    with open(OUTPUT_DIR + "contexto.txt", 'w', encoding='utf-8') as file:
        file.write("Contexto de las palabras en el vocabulario:\n\n")
        for i, word in enumerate(doc_normalized.unique_tokens):
            file.write("Palabra: %s\n" % word)
            file.write("Contexto:\n")
            for value in context[i]:
                file.write("\t%s: %d\n" % (str(value), context[i][value]))
            file.write("\n")
        file.write("\n")
else:
    doc_normalized = output.load_pickle("doc_normalized.pkl")
    context_objs = output.load_pickle("context_objs.pkl")
    context = doc_normalized.contexts

#Generar vectores de contexto con Raw frequency
#freq_raw = []
#for i in range(len(doc_normalized.unique_tokens)):
#    vector = []
#    for j in range(len(context)):
#        vector.append(context[j].get(doc_normalized.unique_tokens[i], 0))
#    freq_raw.append(vector)
#    context_objs[i].setFrequencyVector(vector)
#output.save_pickle("term_frequency_raw.pkl", apply_PCA(freq_raw))
#
##Generar vectores de contexto con Frecuencia relativa
#freq_relative = []
#for i in range(len(doc_normalized.unique_tokens)):
#    vector = []
#    for j in range(len(context)):
#        vector.append(0 if len(context[j]) == 0 else (context[j].get(doc_normalized.unique_tokens[i], 0)/len(context[j])))
#    freq_relative.append(vector)
#output.save_pickle("term_frequency_relative.pkl", apply_PCA(freq_relative))
#
##Generar vectores de contexto con TF Sublineal
#freq_sublin = []
#for i in range(len(doc_normalized.unique_tokens)):
#    vector = []
#    for j in range(len(context)):
#        vector.append(math.log(1 + context[j].get(doc_normalized.unique_tokens[i], 0), 2))
#    freq_sublin.append(vector)
#output.save_pickle("term_frequency_sublin.pkl", apply_PCA(freq_sublin))
#freq_idf_bm25 = idf_bm25()
#output.save_pickle("term_frequency_idfbm25.pkl", apply_PCA(freq_idf_bm25))
#
##Generar entropías 
for sentence in doc_normalized.sentences:
    sentence_word_set = set(sentence)
    for word in sentence_word_set:
        word_properties = context_objs[doc_normalized.index_of(word)]
        word_properties.addSentenceCount()
        for other_word in sentence_word_set:
            word_properties.sentence_crossings[other_word] = word_properties.sentence_crossings.get(other_word, 0) + 1
cond_entropies = []
mutual_infos = []
for word_1 in context_objs:
    prob_w1 = (word_1.sentence_count/doc_normalized.get_sentence_count())
    word_cond_entropies = []
    word_mutual_infos = []
    for word_2 in context_objs:
        prob_w2 = (word_2.sentence_count/doc_normalized.get_sentence_count())
        
        #probabilidades conjuntas:
        count_both = word_1.sentence_crossings.get(word_2.word, 0)
        count_just_w1 = (word_1.sentence_count - count_both)
        count_just_w2 = (word_2.sentence_count - count_both)
        count_none = (doc_normalized.get_sentence_count() - count_both - count_just_w1 - count_just_w2)

        both = count_both / doc_normalized.get_sentence_count()
        just_w1 = count_just_w1 / doc_normalized.get_sentence_count()
        just_w2 = count_just_w2 / doc_normalized.get_sentence_count()
        none = count_none / doc_normalized.get_sentence_count()

        #Entropía condicional
        term_1 = ((prob_w2) * (-(both/prob_w2) * log(both/prob_w2)))
        term_2 = ((prob_w2) * (-(just_w2/prob_w2) * log(just_w2/prob_w2)))
        term_3 = ((1 - prob_w2) * (-(none/(1 - prob_w2)) * log(none/(1 - prob_w2))))
        term_4 = ((1 - prob_w2) * (-(just_w1/(1 - prob_w2)) * log(just_w1/(1 - prob_w2))))

        cond_entropy =  math.fsum([term_1, term_2, term_3, term_4])
        word_cond_entropies.append(cond_entropy)

        #Información mutua
        mutual_info = (0 if (both/(prob_w1 * prob_w2)) == 0 else both * math.log(both/(prob_w1 * prob_w2), 2)) 
        + (0 if (just_w1/(prob_w1 * (1 - prob_w2))) == 0 else just_w1 * math.log(just_w1/(prob_w1 * (1 - prob_w2)), 2))
        + (0 if (just_w2/((1 - prob_w1) * prob_w2)) == 0 else just_w2 * math.log(just_w2/((1 - prob_w1) * prob_w2), 2)) 
        + (0 if (none/((1 - prob_w1) * (1 - prob_w2))) == 0 else none * math.log(none/((1 - prob_w1) * (1 - prob_w2)), 2))
        word_mutual_infos.append(mutual_info) 

    cond_entropies.append(word_cond_entropies)
    mutual_infos.append(word_mutual_infos)
output.save_pickle("term_conditional_entropy.pkl", cond_entropies)
output.save_pickle("term_mutual_info.pkl", mutual_infos)

print("Análisis terminado con éxito.")
