#Clase para guardar y cargar objetos.
import pickle
import os

class Jar:
    def __init__(self, path: str):
        self.path = ""
        #Validar directorio de salida
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except:
                raise OSError("No se pudo crear el directorio de salida.")
        else:
            self.path = path

    #Función para guardar texto en un archivo
    def save_text(self, content, filename: str, description: str = ""):
        path = self.path + filename
        with open(path, 'w', encoding='utf-8') as file:
            file.write("%s\n\n" % description)
            file.write(content)
            file.write("\n")

    #Función para cargar un pickle
    def load_pickle(self, pickle_filename):
        data = None
        path = self.path + pickle_filename
        with open(path, 'rb') as file:
            data = pickle.load(file)
        return data
    
    #Función para guardar un pickle
    def save_pickle(self, pickle_filename, data):
        path = self.path + pickle_filename
        with open(path, "wb") as file:
            pickle.dump(data, file)
