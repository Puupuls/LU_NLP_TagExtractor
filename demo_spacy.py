import spacy
from collections import Counter
from string import punctuation

spacy.prefer_gpu()
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_md")

def get_hotwords(text):
    result = []
    pos_tag = ['PROPN', 'ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if(token.text in nlp.Defaults.stop_words or token.text in punctuation):
            continue
        if(token.pos_ in pos_tag):
            result.append(token.text)
    return result


new_text = "Elon Musk has shared a photo of the spacesuit designed by SpaceX. This is the second image shared of the new design and the first to feature the spacesuit's full-body look."

output = get_hotwords(new_text)
most_common_list = Counter(output).most_common(10)
for item in most_common_list:
  print(item[0])
