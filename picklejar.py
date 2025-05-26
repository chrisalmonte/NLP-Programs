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
    def save_text(self, content, filename: str, description: str = "", file_encoding: str = "utf-8"):
        path = self.path + filename
        with open(path, 'w', encoding=file_encoding) as file:
            file.write("%s\n\n" % description)
            file.write(content)
            file.write("\n")

    #Función para guardar un diccionario en un archivo de texto
    def save_dict_2_txt(self, filename, dictionary: dict, description: str = "", file_encoding: str = "utf-8"):
        path = self.path + filename
        with open(path, 'w', encoding=file_encoding) as file:
            file.write(description + "\n\n")
            for value in dictionary:
                file.write("%s: %d\n" % (str(value), dictionary[value]))
                file.write("\n")
    
    def save_list_2_txt(self, filename, list: list, description: str = "", file_encoding: str = "utf-8", separator: str = '\n'):
        path = self.path + filename
        with open(path, 'w', encoding=file_encoding) as file:
            file.write(description + "\n\n")
            for value in list:
                file.write(str(value))
                file.write(separator)
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

    
