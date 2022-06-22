import os
import re
import gc

import nltk
import numpy as np
import pandas as pd

from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec


def load_KeyedVectors(path):
    tmp_file = get_tmpfile("tmp_word2vec_file.txt")
    _ = glove2word2vec(path, tmp_file)
    embed = KeyedVectors.load_word2vec_format(tmp_file)
    return embed


def average_attr_words(vectors, word_list1, word_list2):
    words_to_average1 = list()
    words_to_average2 = list()
    
    for word1 in word_list1:
        try:
            words_to_average1.append(vectors[word1])
        except:
            pass
            
    for word2 in word_list2:
        try:
            words_to_average2.append(vectors[word2])
        except:
            pass
    
    averaged_words1 = np.array(words_to_average1).mean(axis=0)
    averaged_words2 = np.array(words_to_average2).mean(axis=0)
    
    return averaged_words1, averaged_words2


def cossim(v1, v2, signed = True):
    c = np.dot(v1, v2)/np.linalg.norm(v1)/np.linalg.norm(v2)
    if not signed:
        return abs(c)
    return c


def calc_distance_between_vectors(vec1, vec2, distype = 'norm'):
    if distype == 'norm':
        return np.linalg.norm(np.subtract(vec1, vec2))
    else:
        return cossim(vec1, vec2)


def calc_relative_norm_distance(vectors, male_word_list, female_word_list, neutral_words):
    
    male_avg_vec, female_avg_vec = average_attr_words(vectors, male_word_list, female_word_list)
    
    list_rel_norm_dist = []
    for word in neutral_words:
        try:
            rel_norm_dist = calc_distance_between_vectors(vectors[word], male_avg_vec) - \
                            calc_distance_between_vectors(vectors[word], female_avg_vec)
            list_rel_norm_dist.append(rel_norm_dist)
        except:
            pass
            # print("Word is not present: ", word)
    return np.array(list_rel_norm_dist).mean()


def calc_dist_in_directory(path, male_word_list, female_word_list, list_of_word_groups, files):  
    computed_biases = pd.DataFrame(['career', 'family', 'maths', 'arts', 'science', 'intelligence', 
              'appearance', 'strength', 'weakness', 'professions', 'professions2'], columns=['group'])
    
    for file in files:
        print('Analyzing file: ', file)
        
        file_path = path + file
        word2vec_dict = load_KeyedVectors(file_path)
        
        year = re.findall(r'\d+', file)[0]
        
        results = []
        for word_group in list_of_word_groups:
            # print('\t Analyzing word_group: ', word_group)
            try:
                measure = calc_relative_norm_distance(word2vec_dict, male_word_list, female_word_list, word_group)
                results.append(measure)
            except:
                results.append('NA')
            
        results = pd.DataFrame(results, columns=[year])
        computed_biases = pd.concat([computed_biases, results], axis=1)
        
        del word2vec_dict
        gc.collect()

    return computed_biases


def intersection_align_gensim(m1, m2, words=None):
    """
    Intersect two gensim word2vec models, m1 and m2.
    Only the shared vocabulary between them is kept.
    If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
    Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
    These indices correspond to the new syn0 and syn0norm objects in both gensim models:
        -- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
        -- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
    The .vocab dictionary is also updated for each model, preserving the count but updating the index.
    """

    # Get the vocab for each model
    vocab_m1 = set(m1.index_to_key)
    vocab_m2 = set(m2.index_to_key)

    # Find the common vocabulary
    common_vocab = vocab_m1 & vocab_m2
    if words: common_vocab &= set(words)

    # If no alignment necessary because vocab is identical...
    if not vocab_m1 - common_vocab and not vocab_m2 - common_vocab:
        return (m1,m2)

    # Otherwise sort by frequency (summed for both)
    common_vocab = list(common_vocab)
    common_vocab.sort(key=lambda w: m1.get_vecattr(w, "count") + m2.get_vecattr(w, "count"), reverse=True)
    # print(len(common_vocab))

    # Then for each model...
    for m in [m1, m2]:
        # Replace old syn0norm array with new one (with common vocab)
        indices = [m.key_to_index[w] for w in common_vocab]
        old_arr = m.vectors
        new_arr = np.array([old_arr[index] for index in indices])
        m.vectors = new_arr

        # Replace old vocab dictionary with new one (with common vocab)
        # and old index2word with new one
        new_key_to_index = {}
        new_index_to_key = []
        for new_index, key in enumerate(common_vocab):
            new_key_to_index[key] = new_index
            new_index_to_key.append(key)
        m.key_to_index = new_key_to_index
        m.index_to_key = new_index_to_key
        
        print(len(m.key_to_index), len(m.vectors))
        
    return (m1,m2)


def smart_procrustes_align_gensim(base_embed, other_embed, words=None):
    """
    Original script: https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf
    Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
    Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
        
    First, intersect the vocabularies (see `intersection_align_gensim` documentation).
    Then do the alignment on the other_embed model.
    Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
    Return other_embed.
    If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
    """

    # patch by Richard So [https://twitter.com/richardjeanso) (thanks!) to update this code for new version of gensim
    # base_embed.init_sims(replace=True)
    # other_embed.init_sims(replace=True)

    # make sure vocabulary and indices are aligned
    in_base_embed, in_other_embed = intersection_align_gensim(base_embed, other_embed, words=words)
    
    # re-filling the normed vectors
    in_base_embed.fill_norms(force=True)
    in_other_embed.fill_norms(force=True)
    
    # get the (normalized) embedding matrices
    base_vecs = in_base_embed.get_normed_vectors()
    other_vecs = in_other_embed.get_normed_vectors()
    
    # just a matrix dot product with numpy
    m = other_vecs.T.dot(base_vecs) 
    # SVD method from numpy
    u, _, v = np.linalg.svd(m)
    # another matrix operation
    ortho = u.dot(v) 
    # Replace original array with modified one, i.e. multiplying the embedding matrix by "ortho"
    other_embed.vectors = (other_embed.vectors).dot(ortho)    
    
    return other_embed


def calc_dist_in_directory_aligned(path, male_word_list, female_word_list, list_of_word_groups, files):
    
    computed_biases = pd.DataFrame(['career', 'family', 'maths', 'arts', 'science', 'intelligence', 
              'appearance', 'strength', 'weakness', 'professions', 'professions2'], columns=['group'])
    
    for file in files:
        print('Analyzing file: ', file)
        
        file_path = path + file
        word2vec_dict = KeyedVectors.load_word2vec_format(file_path)
        
        year = re.findall(r'\d+', file)[0]
        
        results = []
        for word_group in list_of_word_groups:
            # print('\t Analyzing word_group: ', word_group)
            try:
                measure = calc_relative_norm_distance(word2vec_dict, male_word_list, female_word_list, word_group)
                results.append(measure)
            except:
                results.append('NA')
            
        results = pd.DataFrame(results, columns=[year])
        computed_biases = pd.concat([computed_biases, results], axis=1)
        
        del word2vec_dict
        gc.collect()

    return computed_biases
