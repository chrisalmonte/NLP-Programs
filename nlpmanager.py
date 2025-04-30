#Clase para guardar las propiedades individuales de cada palabra.
class WordProperties:
    def __init__(self, word: str, context_counts: dict, corpus_frequency: int = 0):
        self.word = word
        self.word_corpus_freq = corpus_frequency
        self.context_bag = set()
        self.context_counts = []
        self.frequency_vector = []
        self.contexts = []
        self.length = 0
        self.entropy = 1
        self.sentence_count = 0
        for word in context_counts.keys():
            self.context_bag.add(word)
            self.context_counts.append((word, context_counts[word]))
            self.length += context_counts[word]
        self.context_counts_dic = context_counts
    
    def addCorpusCount(self):
        self.word_corpus_freq += 1
    
    def setFrequencyVector(self, vector: list):
        self.frequency_vector = vector

    def getCount(self, word):
        return self.context_counts_dic.get(word, 0)
    
    def addSentenceCount(self):
        self.sentence_count += 1

#Clase para guardar el documento normalizado.
class DocProperties:
    def __init__(self, unique_tokens, normalized_sentences, word_frequencies):
        self.unique_tokens = unique_tokens
        self.sentences = normalized_sentences
        self.word_frequencies = word_frequencies
        self.token_index = {}
        for i, token in enumerate(unique_tokens):
            self.token_index[token] = i

    def get_word_count(self):
        return len(self.unique_tokens)
    
    def get_sentence_count(self):
        return len(self.sentences)
        