import pickle
import numpy as np

dictionary_data = "NLP_Output/term_document_data.pkl"

#Cargar vectores
with open(dictionary_data, 'rb') as file:
    vectors = pickle.load(file)

def simmilarity_value(word_a, word_b):
    # Get vectors from the dictionary
    if word_a not in vectors:
        raise ValueError("No se encontró \"%s\" en el diccionario" % word_a)
    if word_b not in vectors:     
        raise ValueError("No se encontró \"%s\" en el diccionario" % word_b)
    
    vector1 = np.array(vectors[word_a])
    vector2 = np.array(vectors[word_b])
    
    cos_theta = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    return max(cos_theta, 0)

#Comparar palabra
word = "pri"

similitud = []
for token in vectors:
    similitud.append((token, simmilarity_value(word, token)))
similitud = sorted(similitud, key=lambda x: x[1], reverse=True)

with open("NLP_Output/similitud_%s.txt" % word, 'w', encoding="utf-8") as file:
    file.write("Similitud de la palabra \"%s\" con las demás palabras del vocabulario:\n\n" % word)
    for valor in similitud:
        file.write("%s: %f\n" % (valor[0], valor[1]))
