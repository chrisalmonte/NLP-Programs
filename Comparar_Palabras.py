#Módulo que compara los vectores de las palabras, una vez que se han generado con el módulo Vectores_Frecuencia.py

import pickle
import numpy as np

OUTPUT_DIR = "NLP_Output/"

#Función para cargar un archivo de pickle
def open_data(path:str):
    data = None
    with open(path, 'rb') as file:
        data = pickle.load(file)
    return data

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

def write_comparison(file_name: str, header:str, **kwargs):
    with open(OUTPUT_DIR + file_name, 'w', encoding="utf-8") as file:
        file.write("%s\n\n" % header)
        for key in kwargs.keys():
            file.write(str(key).ljust(27) + "| ")
        file.write("\n\n")
        for i in range(len(list(kwargs.values())[0])):
            for word_data  in kwargs.values():
                file.write(str(word_data[i][0]).ljust(20) + ": " + "{:.3f}".format(word_data[i][1]).ljust(5) + "| ")
            file.write("\n")
            

word = "partido"
vecs_fraw = open_data(OUTPUT_DIR + "term_frequency_raw.pkl")
vecs_frel = open_data(OUTPUT_DIR + "term_frequency_relative.pkl")
vecs_fsub = open_data(OUTPUT_DIR + "term_frequency_sublin.pkl")
sim_word_raw = simmilarity_values(word, vecs_fraw)
sim_word_rel = simmilarity_values(word, vecs_frel)
sim_word_sub = simmilarity_values(word, vecs_fsub)

write_comparison("similitud_%s_comparacion.txt" % word, "Similitud de la palabra \"%s\"" % word, Raw_Frequency=sim_word_raw, Relative_Frequency=sim_word_rel, Sublineal=sim_word_sub)
