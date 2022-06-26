from __future__ import annotations

import os.path
from collections import Counter
from string import punctuation
import flask
import gensim.downloader
import nltk
import spacy
from flask import render_template, request
from gensim.models._fasttext_bin import Model
from keybert import KeyBERT
from loguru import logger
from rake_nltk import Rake as NLTKRake
from rake_spacy import Rake as SpacyRake
from spacy import Language
from torch import cosine_similarity
from torchtext.vocab import GloVe, FastText
from yake import KeywordExtractor

app = flask.Flask(__name__)
nltk.download('stopwords')
nltk.download('punkt')
spacy.download('en_core_web_lg')
glove: None | GloVe = None
nlp: None | Language = None
keybert: None | KeyBERT = None
word2vec: None = None
fast_text: None = None
yake_model: None | KeywordExtractor = None


def get_hotwords(text: str, limit: int = 10):
    """
    Get most common adjectives and Nouns in text and mark them as tags.
    """
    result = []
    pos_tag = ['ADJ', 'NOUN']
    doc = nlp(text.lower())
    for token in doc:
        if token.text in nlp.Defaults.stop_words or token.text in punctuation:
            continue
        if token.pos_ in pos_tag:
            result.append(token.text)
    return Counter(result).most_common(limit)


def get_similar(model, word, limit=10, similarity_threshold=0.75):
    """
    Get the closest words by meaning using GloVe embeddings.
    """
    if word not in model.stoi and word.lower() not in model.stoi:
        return []
    # Get word embedding
    vec = model.get_vecs_by_tokens([word], True)
    # Get all cosine similarities
    distances = cosine_similarity(vec, model.vectors)
    # Sort by similarity
    shortest_distances = distances.flatten().argsort()[-limit:]
    # Filter by similarity threshold
    filtered_shortest_distances = shortest_distances[distances[shortest_distances] > similarity_threshold]
    # Return the closest words and their similarity
    return [(glove.itos[i], distances[i]) for i in filtered_shortest_distances][::-1]


def text2camel(text):
    """
    Convert text to camel case.
    """
    return text.title().replace(" ", "").replace("_", "")


@app.route('/')
def index():
    """
    Show the web page.
    """
    return render_template('index.html')


@app.route('/get-tags', methods=['POST'])
def get_tags():
    """
    Get the tags from the text using 4 different methods/models.
    """
    # Do not load models on startup, instead load them on demand and cache them to speed up subsequent requests
    global glove, nlp, fast_text, word2vec, keybert, yake_model
    if not glove:
        logger.info('Loading GloVe')
        glove = GloVe(name='840B', dim=300)
    if not fast_text:
        logger.info('Loading FastText')
        if not os.path.exists('fast_text.model'):
            fast_text = gensim.downloader.load('fasttext-wiki-news-subwords-300')
            fast_text.save('fast_text.model')
        else:
            fast_text = gensim.models.KeyedVectors.load('fast_text.model')
    if not nlp:
        logger.info('Loading spacy')
        nlp = spacy.load("en_core_web_lg", exclude=['parser', 'ner', "lemmatizer"])
    if not word2vec:
        logger.info('Loading Word2Vec')
        if not os.path.exists('word2vec.model'):
            word2vec = gensim.downloader.load('word2vec-google-news-300')
            word2vec.save('word2vec.model')
        else:
            word2vec = gensim.models.KeyedVectors.load('word2vec.model')
    if not keybert:
        logger.info('Loading KeyBERT')
        keybert = KeyBERT(model='all-mpnet-base-v2')
    if not yake_model:
        logger.info('Loading YAKE')
        yake_model = KeywordExtractor()

    # Get the text from the web page and create response object
    text = request.form['text']
    result = {
        "text": text
    }

    # Get spacy tags that are based on Part of Speech (POS) tagging
    logger.info("Processing SpaCy")
    spacy_hotwords = {}
    try:
        most_common_list = get_hotwords(text)
        spacy_hotwords = {text2camel(tag[0]): float(tag[1]) for tag in most_common_list}
    except Exception as e:
        logger.exception(e)
    result['spacy_hotwords'] = spacy_hotwords

    # Get tags using spacy Rake
    logger.info("Processing spacy_rake")
    rake_spacy_tags = {}
    try:
        r = SpacyRake(
            nlp=nlp,
            min_length=1,
            max_length=3,
        )
        extracted = [(float(v), str(t)) for v, t in r.apply(text)]
        rake_spacy_tags = {text2camel(str(t)): v for v, t in extracted}
    except Exception as e:
        logger.exception(e)
    result['rake_spacy_tags'] = rake_spacy_tags

    # Get tags using NLTK Rake these usually are almost the same as above ones as they use the same algorithm, but
    # sometimes they are different
    logger.info("Processing rake_nltk")
    rake_nltk_tags = {}
    try:
        # Use Rake 3 times with different length keywords
        r = NLTKRake(
            min_length=1,
            max_length=3,
            include_repeated_phrases=False,
        )
        r.extract_keywords_from_text(text)
        ranklist = r.get_ranked_phrases_with_scores()
        rake_nltk_tags = {text2camel(tag[1]): float(tag[0]) for tag in ranklist}
    except Exception as e:
        logger.exception(e)
    result['rake_nltk_tags'] = rake_nltk_tags

    # Get tags using KeyBERT
    logger.info("Processing KeyBERT")
    keybert_tags = {}
    try:
        # Use different n-gram lengths to find more tags
        kb_tags = keybert.extract_keywords(text, keyphrase_ngram_range=(3, 3), top_n=10)
        keybert_tags = {text2camel(tag[0]): float(tag[1]) for tag in kb_tags}
        kb_tags = keybert.extract_keywords(text, keyphrase_ngram_range=(2, 2), top_n=10)
        keybert_tags.update({text2camel(tag[0]): float(tag[1]) for tag in kb_tags})
        kb_tags = keybert.extract_keywords(text, keyphrase_ngram_range=(1, 1), top_n=10)
        keybert_tags.update({text2camel(tag[0]): float(tag[1]) for tag in kb_tags})
        keybert_tags = {k: v for k, v in keybert_tags.items() if v > 0.5}
    except Exception as e:
        logger.exception(e)
    result['keybert_tags'] = keybert_tags

    # Get tags using YAKE
    logger.info("Processing YAKE")
    yake_tags = {}
    try:
        yake_tags = yake_model.extract_keywords(text)
        yake_tags = {text2camel(tag[0]): float(tag[1]) for tag in yake_tags}
    except Exception as e:
        logger.exception(e)
    result['yake_tags'] = yake_tags

    # Merge tags from above 4 methods (treat both RAKE implementations as one)
    tags = list(spacy_hotwords.keys()) + list(set(list(rake_spacy_tags.keys()) + list(rake_nltk_tags.keys()))) + \
           list(keybert_tags.keys()) + list(yake_tags.keys())

    # Get tags that are mentioned more than once
    counts = Counter(tags)
    tags = {tag: count for tag, count in counts.most_common() if count > 1}
    result['tags'] = tags

    # Get similar words based on meaning using GloVe
    logger.info("Processing GloVe")
    glove_tags = {}
    tags_lower = [tag.lower() for tag in tags]
    for tag in tags:
        try:
            for similar_word, sim_conf in get_similar(glove, tag):
                # If the similar word is not in the list of tags, add it to the list of tags
                if similar_word.lower() not in tags_lower:
                    glove_tags[text2camel(similar_word)] = max(float(sim_conf),
                                                               glove_tags.setdefault(
                                                                   text2camel(similar_word), 0))
        except Exception as e:
            logger.exception(e)
    result['glove'] = glove_tags

    # Get similar words based on meaning using FastText
    logger.info("Processing FastText")
    fast_text_tags = {}
    for tag in tags:
        try:
            for similar_word, sim_conf in fast_text.most_similar(tag.lower(), topn=10):
                # If the similar word is not in the list of tags, add it to the list of tags
                if similar_word.lower() not in tags_lower:
                    fast_text_tags[text2camel(similar_word)] = max(float(sim_conf),
                                                                   fast_text_tags.setdefault(
                                                                       text2camel(similar_word), 0))
        except KeyError:
            logger.warning(f"FastText does not have the word: {tag}")
        except Exception as e:
            logger.exception(e)
    result['fasttext'] = fast_text_tags

    # Get similar words based on meaning using Word2Vec
    logger.info("Processing Word2Vec")
    word2vec_tags = {}
    for tag in tags:
        try:
            for similar_word, sim_conf in word2vec.most_similar(tag.lower(), topn=10):
                # If the similar word is not in the list of tags, add it to the list of tags
                if similar_word.lower() not in tags_lower and sim_conf > 0.75:
                    word2vec_tags[text2camel(similar_word)] = max(float(sim_conf),
                                                                  word2vec_tags.setdefault(
                                                                      text2camel(similar_word), 0))
        except KeyError:
            logger.warning(f"Word2Vec does not have the word: {tag}")
        except Exception as e:
            logger.exception(e)
    result['word2vec'] = word2vec_tags

    # Merge previously found tags with the tags found using GloVe
    result['tags'].update(glove_tags)
    # result['tags'].update(fast_text_tags)  # Disable this as it generates tags with typos
    result['tags'].update(word2vec_tags)

    logger.info(result)
    return result


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
