#Módulo que compara los vectores de las palabras, una vez que se han generado con el módulo Vectores_Frecuencia.py

import numpy as np

#Módulos propios
import picklejar

#Palabra a comparar
WORD = "problema"
#Donde se guardaron los pickles del análisis de texto
OUTPUT_DIR = "NLP_Output/"

#Función que regresa el ángulo coseno entre dos vectores de palabra
def angle_between_vectors(vector1, vector2,):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)    
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return max(cos_theta, 0)

def simmilarity_values(word: str, vector_dictionary: dict):
    if word not in vector_dictionary.keys():
        raise ValueError("No se encontró \"%s\" en los vectores" % word)
    similitud = []
    for token in vector_dictionary:
        similitud.append((token, angle_between_vectors(vector_dictionary[word], vector_dictionary[token])))
    return sorted(similitud, key=lambda x: x[1], reverse=True)

def write_dict_columns(file_name: str, header:str, **kwargs):
    with open(OUTPUT_DIR + file_name, 'w', encoding="utf-8") as file:
        file.write("%s\n\n" % header)
        for key in kwargs.keys():
            file.write(str(key).ljust(27) + "| ")
        file.write("\n\n")
        for i in range(len(list(kwargs.values())[0])):
            for value  in kwargs.values():
                file.write(str(value[i][0]).ljust(20) + ": " + "{:.3f}".format(value[i][1]).ljust(5) + "| ")
            file.write("\n")

def order_entropy_values(word: str, vectors: list, descending: bool = False):
    pairs = []
    vector = vectors[doc_normalized.index_of(word)]
    for i, value in enumerate(vector):
        pairs.append((doc_normalized.unique_tokens[i], value))
    return sorted(pairs, key=lambda x: x[1], reverse=descending)


#Programa
#Cargar los vectores
output = picklejar.Jar(OUTPUT_DIR)
vecs_fraw = output.load_pickle("term_frequency_raw.pkl")
vecs_frel = output.load_pickle("term_frequency_relative.pkl")
vecs_fsub = output.load_pickle("term_frequency_sublin.pkl")
vecs_idfbm25 = output.load_pickle("term_frequency_idfbm25.pkl")
vecs_entropy = output.load_pickle("term_conditional_entropy.pkl")
vecs_mutual_info = output.load_pickle("term_mutual_info.pkl")
doc_normalized = output.load_pickle("doc_normalized.pkl")

#sim_word_raw = simmilarity_values(WORD, vecs_fraw)
#sim_word_rel = simmilarity_values(WORD, vecs_frel)
#sim_word_sub = simmilarity_values(WORD, vecs_fsub)
#sim_word_idfbm25 = simmilarity_values(WORD, vecs_idfbm25)
#write_dict_columns("similitud_%s.txt" % WORD, "Similitud de la palabra \"%s\"" % WORD, Raw_Frequency=sim_word_raw, 
#                 Relative_Frequency=sim_word_rel, Sublineal=sim_word_sub, IDF_BM25=sim_word_idfbm25)

cond_entropy = order_entropy_values(WORD, vecs_entropy)
mutual_info = order_entropy_values(WORD, vecs_mutual_info, descending=True)
write_dict_columns("entropia_%s.txt" % WORD, "Valores de entropía de la palabra \"%s\"" % WORD, Conditional_Entropy=cond_entropy,
                 Mutual_Information=mutual_info)
