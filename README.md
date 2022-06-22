# Measuring Gender Bias in Word Embeddings

## Author: Bakbergen Ryskulov

This is a supplementary code for my master thesis **"Do economic shocks affect the gender bias in language?".**

 **_NOTE:_**  The code is not optimized, since the analysis is supposed to run only once and for preliminary results.


## Acknowledgements

For pretrained words vectors on Google Ngram Dataset: [Zi Yin](https://github.com/ziyin-dl/ngram-word2vec) \
For aligning method of word vectors: [Hamilton et. al (2016)](https://github.com/williamleif/histwords) \
For gender bias measurement methods: [Garg et. al (2018)](https://github.com/nikhgarg/EmbeddingDynamicStereotypes) and [Caliskan et al. 2017](https://arxiv.org/abs/1608.07187) \
For preprocessing and training word vectors: [Pierre Megret](https://www.kaggle.com/code/pierremegret/gensim-word2vec-tutorial/notebook)


## Usage

The folder ```notebooks``` contains Jupyter notebooks used for: 
1. Prototyping of gender bias measurement for word vectors in GloVe format.
2. Training Word2Vec model on Alice Wu [gendered posts sample](https://www.aeaweb.org/articles?id=10.1257/pandp.20181101).

A step-by-step explanation of the analysis is described in Jupyter notebooks.

Sample code to measure gender bias for pretrained word vectors:

```bash
python -m analyze_hist_books data/pretrained_vectors/ data/aligned_vectors/ --start-year 1900 --year-inc 50 --end-year 1951
```

The folder ```svar_matlab``` contains MatLab scripts to run Structural Vector Autoregression analysis used in my thesis.

## License
[MIT](https://choosealicense.com/licenses/mit/)