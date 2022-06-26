from rake_nltk import Rake
import nltk

nltk.download('stopwords')
nltk.download('punkt')

# Uses stopwords for english from NLTK, and all puntuation characters by default
r = Rake()

r.extract_keywords_from_text("This is a test text that should have some tags in it.")
# r.extract_keywords_from_sentences(<list of sentences>)

print(r.get_ranked_phrases())
print(r.get_ranked_phrases_with_scores())
