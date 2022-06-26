from keybert import KeyBERT

doc = """
Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned 
with the interactions between computers and human language, in particular how to program computers to process and analyze 
large amounts of natural language data. The goal is a computer capable of "understanding" the contents of documents, 
including the contextual nuances of the language within them. The technology can then accurately extract information 
and insights contained in the documents as well as categorize and organize the documents themselves. Challenges in 
natural language processing frequently involve speech recognition, natural-language understanding, 
and natural-language generation.
"""
kw_model = KeyBERT()
# kw_model = KeyBERT(model='all-MiniLM-L6-v2')
# kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

print(kw_model.extract_keywords(doc))
print(kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 1), stop_words=None))
print(kw_model.extract_keywords(doc, keyphrase_ngram_range=(1, 2), stop_words=None))
print(kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', use_maxsum=True, nr_candidates=20, top_n=5))
# print(kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', use_mmr=True, diversity=0.7))
# print(kw_model.extract_keywords(doc, keyphrase_ngram_range=(3, 3), stop_words='english', use_mmr=True, diversity=0.2))

