import picklejar
import xml.etree.ElementTree as ET

OUTPUT_DIR = "SA-Output/"
SENTIMENT_DIC_DIR = "../../Corpora/Sentiment_dictionaries-Diccionarios_de_sentimientos/"

senticon_continious = {}
senticon_discrete_full = {}
senticon_discrete_medium = {}
output = picklejar.Jar(OUTPUT_DIR)

#Read senticon polarity file
senticon_xml_file = SENTIMENT_DIC_DIR + "Senticon_es/senticon.es.xml"
tree = ET.parse(senticon_xml_file)
root = tree.getroot()
for lemma_element in root.findall('.//lemma'): # Finds all 'lemma' elements anywhere in the tree
    lemma_text = lemma_element.text.strip()
    pol_value = lemma_element.get('pol')
    senticon_continious[lemma_text] = float(pol_value)
print("Length of continous senticon:", len(senticon_continious))

#Read senticon strength file
senticon_discrete_file = SENTIMENT_DIC_DIR + "Spanish_sentiment_lexicon/fullStrengthLexicon.txt"
with open(senticon_discrete_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = line.split('\t')
        if len(entry) >= 3:
            senticon_discrete_full[entry[0]] = entry[-1].rstrip()
print("Length of discrete senticon:", len(senticon_discrete_full))
print("Discrete values:", set(senticon_discrete_full.values()))

#Read senticon medium strength file
senticon_discrete_file = SENTIMENT_DIC_DIR + "Spanish_sentiment_lexicon/mediumStrengthLexicon.txt"
with open(senticon_discrete_file, 'r', encoding='utf-8') as file:
    for line in file:
        entry = line.split('\t')
        if len(entry) >= 3:
            senticon_discrete_medium[entry[0]] = entry[-1].rstrip()
print("Length of discrete senticon:", len(senticon_discrete_medium))
print("Discrete medium values:", set(senticon_discrete_medium.values()))

# Save the dictionaries
output.save_pickle("senticon_continious.pkl", senticon_continious)
output.save_pickle("senticon_discrete.pkl", senticon_discrete_full)
output.save_pickle("senticon_discrete_medium.pkl", senticon_discrete_medium)