from bs4 import BeautifulSoup

class Operations:
    @staticmethod
    #Función para extraer texto de un archivo
    def str_from_file(file_path: str, file_encoding: str="utf-8", parse_HTML=True):
        with open(file_path, encoding=file_encoding) as file:
            raw = file.read()
        #Limpiar etiquetas HTML
        if parse_HTML:
            soup = BeautifulSoup(raw, "lxml")
            return soup.get_text()
        return raw