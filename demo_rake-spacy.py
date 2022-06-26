import spacy
from collections import Counter
from string import punctuation
from rake_spacy import Rake

spacy.prefer_gpu()
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_md
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_md")

r = Rake(
    nlp=nlp
)

text = "This is a test text that should have some tags in it."

ranklist = r.apply(text)

print(ranklist)
