import random

import gensim
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import regex
import seaborn as sns
import spacy
import streamlit as st
import tensorflow as tf
from PIL import Image
from gensim import corpora
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
from wordcloud import WordCloud

nltk.download("stopwords")

#
# NLP
#

DATASETS = {
    'Commentaires Yelp en français': {
        'path': './data/yelp_reviews.csv.zip',
        'column': 'text',
        'url': 'https://github.com/Abdess/avis_restau',
        'description': (
            'Une série de commentaires de restaurants français référencés sur Yelp. '
            'Le but est de comprendre les sujets des clients. '
        )
    },
    'Commentaires négatifs sur Yelp en français': {
        'path': './data/yelp_bad_reviews.csv.zip',
        'column': 'text',
        'url': 'https://github.com/Abdess/avis_restau',
        'description': (
            'Une série de commentaires négatifs de restaurants français référencés sur Yelp. '
            'Le but est de comprendre les sujets de mécontentement des clients. '
        )
    }
}

MODELS = {
    'lda': {
        'display_name': 'Latent Dirichlet Allocation (LDA)',
    },
    'nmf': {
        'display_name': 'Non-Negative Matrix Factorization (NMF)'
    }
}

COLORS = [color for color in mcolors.XKCD_COLORS.values()]

WORDCLOUD_FONT_PATH = r'./data/Inkfree.ttf'

EMAIL_REGEX_STR = '\S*@\S*'
MENTION_REGEX_STR = '@\S*'
HASHTAG_REGEX_STR = '#\S+'
URL_REGEX_STR = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'


@st.cache()
def generate_texts_df(selected_dataset: str):
    dataset = DATASETS[selected_dataset]
    return pd.read_csv(f'{dataset["path"]}')


@st.cache()
def denoise_docs(texts_df: pd.DataFrame, text_column: str):
    texts = texts_df[text_column].values.tolist()
    remove_regex = regex.compile(
        f'({EMAIL_REGEX_STR}|{MENTION_REGEX_STR}|{HASHTAG_REGEX_STR}|{URL_REGEX_STR})')
    texts = [regex.sub(remove_regex, '', text) for text in texts]
    docs = [[w for w in simple_preprocess(
        doc, deacc=True) if w not in stopwords.words('french')] for doc in texts]
    return docs


@st.cache()
def create_bigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    docs = [bigram_phraser[doc] for doc in docs]
    return docs


@st.cache()
def create_trigrams(docs):
    bigram_phrases = gensim.models.Phrases(docs)
    bigram_phraser = gensim.models.phrases.Phraser(bigram_phrases)
    trigram_phrases = gensim.models.Phrases(bigram_phrases[docs])
    trigram_phraser = gensim.models.phrases.Phraser(trigram_phrases)
    docs = [trigram_phraser[bigram_phraser[doc]] for doc in docs]
    return docs


@st.cache()
def generate_docs(texts_df: pd.DataFrame, text_column: str, ngrams: str = None):
    docs = denoise_docs(texts_df, text_column)

    if ngrams == 'bigrams':
        docs = create_bigrams(docs)
    if ngrams == 'trigrams':
        docs = create_trigrams(docs)

    lemmantized_docs = []
    nlp = spacy.load('fr_core_news_lg', disable=['parser', 'ner'])
    for doc in docs:
        doc = nlp(' '.join(doc))
        lemmantized_docs.append([token.lemma_ for token in doc])

    return lemmantized_docs


@st.cache()
def generate_wordcloud(docs, collocations: bool = False):
    wordcloud_text = (' '.join(' '.join(doc) for doc in docs))
    wordcloud = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                          background_color='white', collocations=collocations).generate(wordcloud_text)
    return wordcloud


# @st.cache()
def prepare_training_data(docs):
    id2word = corpora.Dictionary(docs)
    corpus = [id2word.doc2bow(doc) for doc in docs]
    return id2word, corpus


def train_model(docs, num_topics: int = 5, per_word_topics: bool = True):
    id2word, corpus = prepare_training_data(docs)
    model = gensim.models.LdaModel(corpus=corpus, id2word=id2word, num_topics=num_topics,
                                   per_word_topics=per_word_topics)
    return model


if __name__ == '__main__':
    st.set_page_config(page_title='Avis Restau',
                       page_icon='./data/favicon.png')

    ngrams = None
    with st.sidebar:
        st.header('Paramètres')
        st.subheader('WordCloud')
        collocations = st.checkbox('Activer les collocations')
        st.markdown(
            "<sup>Les collocations dans les nuages de mots permettent d'afficher des phrases.</sup>",
            unsafe_allow_html=True)

        st.subheader('Prétraitement')
        if st.checkbox('Utiliser les N-grammes'):
            ngrams = st.selectbox('N-grammes', ['bigrams', 'trigams'])

        st.subheader('Topic Modeling')
        st.selectbox('Modèle de base', [
            model['display_name'] for model in MODELS.values()])
        num_topics = st.number_input(
            'Nombre de sujets', min_value=1, max_value=200, value=5)
        per_word_topics = st.checkbox('Sujets par mot', value=True)

    st.title('Avis Restau')
    st.header('Analyse des commentaires')
    st.markdown(
        'La modélisation de sujets est un terme général. Il englobe un certain nombre de méthodes d\'apprentissage statistique spécifiques. '
        'Ces méthodes procèdent de la manière suivante : elles expliquent les documents en fonction d\'un ensemble '
        'de sujets, et ces sujets en fonction des éléments suivants un ensemble de mots.'
    )
    st.markdown(
        'Deux méthodes très couramment utilisées sont l\'allocation latente de Dirichlet (LDA) '
        'et la factorisation matricielle non négative. (NMF), par exemple.'
    )
    st.markdown(
        'Utilisée sans qualificatifs supplémentaires, l\'approche est généralement supposée être '
        'non supervisé, bien qu\'il existe des variantes semi-supervisées et supervisées.'
    )

    with st.expander('Détails supplémentaires'):
        st.markdown(
            'L\'objectif peut être considéré comme une factorisation matricielle.')
        st.image('./data/mf.png', use_column_width=True)
        st.markdown('Cette factorisation rend les méthodes beaucoup plus efficaces '
                    ' que la caractérisation directe des documents in term of words.')
        # st.markdown('LDA est ... TODO ...')  # TODO
        # st.markdown('NMF est ... TODO ...')  # TODO

    st.header('Jeux de données')
    st.markdown(
        'Quelques petits ensembles de données d\'exemple ont été préchargés pour illustrer la situation.')
    selected_dataset = st.selectbox(
        'Jeu de données', sorted(list(DATASETS.keys())))
    with st.expander('Description du jeu de données', expanded=True):
        st.markdown(DATASETS[selected_dataset]['description'])
        st.markdown(DATASETS[selected_dataset]['url'])

    text_column = DATASETS[selected_dataset]['column']
    texts_df = generate_texts_df(selected_dataset)
    docs = generate_docs(texts_df, text_column, ngrams=ngrams)

    with st.expander('Documents de référence', expanded=True):
        sample_texts = texts_df[text_column].sample(5).values.tolist()
        for index, text in enumerate(sample_texts):
            st.markdown(f'**{index + 1}**: _{text}_')

    with st.expander('Corpus de mots en fréquence', expanded=True):
        wc = generate_wordcloud(docs)
        st.image(wc.to_image(
        ), caption='Jeu de données Wordcloud (pas un modèle de sujet)', use_column_width=True)
        st.markdown(
            'Ce sont les mots restants après le prétraitement du document.')

    with st.expander('Distribution du nombre de mots du document', expanded=True):
        len_docs = [len(doc) for doc in docs]
        fig, ax = plt.subplots()
        sns.histplot(pd.DataFrame(len_docs, columns=[
            'Mots dans le document']), ax=ax)
        st.pyplot(fig)

    model = None
    with st.sidebar:
        if st.button('Entraîner'):
            model = train_model(docs, num_topics, per_word_topics)
            st.balloons()
            st.success('Entraînement terminé !')

    st.header('Modèle')
    if model:
        st.write(model)
    else:
        st.markdown(
            'Aucun modèle n\'a encore été formé : utilisez la barre latérale pour configurer et entraîner un modèle de modélisation de sujets.')

    st.header('Résultats du modèle')
    if model:
        topics = model.print_topics()
        st.subheader('Résumés des sujets pondérés par les mots')
        for topic in topics:
            st.markdown(f'**Topic #{topic[0]}**: _{topic[1]}_')

        st.subheader('Top N des mots-clés des sujets Wordclouds')
        topics = model.show_topics(formatted=False, num_topics=num_topics)
        cols = st.columns(3)
        colors = random.sample(COLORS, k=len(topics))
        for index, topic in enumerate(topics):
            wc = WordCloud(font_path=WORDCLOUD_FONT_PATH, width=700, height=600,
                           background_color='white', collocations=collocations, prefer_horizontal=1.0,
                           color_func=lambda *args, **kwargs: colors[index])
            with cols[index % 3]:
                wc.generate_from_frequencies(dict(topic[1]))
                st.image(wc.to_image(),
                         caption=f'Topic #{index}', use_column_width=True)
    else:
        st.markdown(
            'Aucun modèle n\'a encore été entraîné : utilisez la barre latérale pour configurer et entraîner un modèle de modélisation de sujets.')

# ### Désactivé : Inutile d'en faire trop, mieux vaut se limiter à l'essentiel, au productif

# import streamlit as st
# import time
# import argparse
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from plotly.graph_objs import *
# from sklearn.decomposition import PCA
# import numpy as np
# from PIL import Image

# def display_props():
# 	# en-tête
# 	st.markdown("#### Explorer avec différents schémas de réduction et fonctionnalité de recherche")
# 	return
# display_props()

# ## options d'embeddings
# embeddings = ("Word2Vec 1k", "GloVe 1k")
# options = list(range(len(embeddings)))
# embedding_type = st.sidebar.selectbox("Sélectionner l'embeddings", options, format_func=lambda x: embeddings[x])
# st.sidebar.text('OU')
# uploaded_file = st.sidebar.file_uploader("Téléverser un fichier (facultatif)", type="txt")

# def load_data(embedding_type):
# 	if embedding_type==0:
# 		file = "./data/reviews2vec.csv"

# 	df = pd.read_csv(file)
# 	data = df.values.tolist()
# 	labels = [d[0] for d in data]
# 	data = np.array([d[1:] for d in data])
# 	return data, labels

# if not uploaded_file:
# 	data, labels = load_data(embedding_type)
# else:
# 	df = pd.read_table(uploaded_file, sep='\s')
# 	data = df.values.tolist()
# 	labels = [d[0] for d in data]
# 	data = np.array([d[1:] for d in data])

# ## réductions dimensionnelles
# def display_reductions():
# 	reductions = ("ACP", "T-SNE")
# 	options = list(range(len(reductions)))
# 	reductions_type = st.sidebar.selectbox("Sélectionner la réduction dim.", options, format_func=lambda x: reductions[x])
# 	return reductions_type
# reductions_type = display_reductions()

# # no. dimensions
# def display_dimensions():
# 	dims = ("2-D", "3-D")
# 	dim = st.sidebar.radio("Dimensions", dims)
# 	return dim
# dim = display_dimensions()

# def plot_2D(data, labels, need_labels, search=None):
# 	sizes = [5]*len(labels)
# 	colors = ['rgb(93, 164, 214)']*len(labels)
# 	if search:
# 		sizes[search] = 25
# 		colors[search] = 'rgb(243, 14, 114)'

# 	if not need_labels:
# 		labels=None

# 	fig = go.Figure(data=[go.Scatter(
# 		    x=data[:,0], y=data[:,1],
# 		    mode='markers+text',
# 		    text=labels,
# 		    marker=dict(
# 		        color=colors,
# 		        size=sizes
# 		    )
# 		)],layout=Layout(
#     paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)'))
# 	return fig

# def plot_3D(data, labels, need_labels, search=None):
# 	sizes = [5]*len(labels)
# 	colors = ['rgb(93, 164, 214)']*len(labels)

# 	if search:
# 		sizes[search] = 25
# 		colors[search] = 'rgb(243, 14, 114)'

# 	if not need_labels:
# 		labels=None

# 	fig = go.Figure(data=[go.Scatter3d(
# 		    x=data[:,0], y=data[:,1], z=data[:,2],
# 		    mode='markers+text',
# 		    text=labels,
# 		    marker=dict(
# 		        color=colors,
# 		        size=sizes
# 		    )
# 		)], layout=Layout(
#     paper_bgcolor='rgba(0,0,0,0)',
#     plot_bgcolor='rgba(0,0,0,0)'))
# 	return fig

# # recherche
# def display_search():
# 	search_for = st.sidebar.text_input("Recherche de mots", "")
# 	return search_for
# search_for = display_search()

# # contrôle des étiquettes
# def display_labels():
# 	need_labels = st.sidebar.checkbox("Affichage des étiquettes", value=True)
# 	return need_labels
# need_labels = display_labels()

# def render_plot(fig):
# 	fig.update_layout(margin={"r":50,"t":100,"l":0,"b":0}, height=750, width=850)
# 	st.plotly_chart(fig)

# def plot_for_D(data, labels, need_labels, search_idx=None):
# 	if dim=='2-D':
# 		fig = plot_2D(data, labels, need_labels, search_idx)
# 		render_plot(fig)
# 	elif dim=='3-D':
# 		fig = plot_3D(data, labels, need_labels, search_idx)
# 		render_plot(fig)

# button = st.sidebar.button('Visualiser')
# if button:
# 	if dim=='2-D':
# 		pca = PCA(n_components=2)
# 		data = pca.fit_transform(data)
# 	else:
# 		pca = PCA(n_components=3)
# 		data = pca.fit_transform(data)

# 	if search_for:
# 		search_idx = labels.index(search_for)
# 		plot_for_D(data, labels, need_labels, search_idx)
# 	else:
# 		plot_for_D(data, labels, need_labels)

#
# CV
#

st.header("Analyse des photos")


@st.cache(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('./data/yelp_images.h5')
    return model


def predict_class(image, model):
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, [224, 224])

    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)

    return prediction


model = load_model()

file = st.file_uploader(
    "Preuve de concept de la labellisation automatique des photos. Identification des photos relatives à la nourriture, au décor dans le restaurant ou à l’extérieur du restaurant",
    type=["jpg", "png"])

if file is None:
    st.text("Veuillez charger une image provenant d'un restaurant Yelp")

else:
    slot = st.empty()
    slot.text('Inférence en cours...')

    test_image = Image.open(file)

    st.image(test_image, caption="Image d'entrée", width=400)

    pred = predict_class(np.asarray(test_image), model)

    class_names = ['Boisson', 'Nourriture', 'Intérieur', 'Menu', 'Extérieur']

    result = class_names[np.argmax(pred)]

    output = 'Cette image est classifié comme : ' + result

    slot.text('Terminé')

    st.success(output)
