import os
import re
import gc
import argparse

import pandas as pd
from gensim.models import KeyedVectors

from calc_utils import load_KeyedVectors
from calc_utils import smart_procrustes_align_gensim
from calc_utils import calc_relative_norm_distance

from word_groups.caliscan_words import *



def align_vectors(years, in_dir, out_dir):
    first_iter = True
    base_embed = None

    for year in years:
        print("Loading year: ", year)
        year_embed = load_KeyedVectors(in_dir + f'vectors-{year}-ngram.txt')
        
        print("Aligning year: ", year)
        if first_iter:
            aligned_embed = year_embed
            first_iter = False
        else:
            aligned_embed = smart_procrustes_align_gensim(base_embed, year_embed)
        base_embed = aligned_embed
        
        print("Writing year: ", year)
        aligned_embed.save_word2vec_format(out_dir + f'vectors-{year}-ngram-aligned.txt')


def calc_dist_in_directory(aligned_dir, male_word_list, female_word_list, list_of_word_groups):
    computed_biases = pd.DataFrame(['career', 'family', 'maths', 'arts', 'science', 'intelligence', 
              'appearance', 'strength', 'weakness', 'professions', 'professions2'], columns=['group'])

    files = os.listdir(aligned_dir)[1:] #remove .getkeep file
    
    for file in files:
        print('Analyzing file: ', file)
        
        file_path = aligned_dir + file
        word2vec_dict = KeyedVectors.load_word2vec_format(file_path)
        
        year = re.findall(r'\d+', file)[0]
        
        results = []
        for word_group in list_of_word_groups:
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
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Computes gender bias for word embeddings and writes excel file to the current directory.")
    parser.add_argument("pretrained_dir", help="path to word vectors", default='data/pretrained_vectors/')
    parser.add_argument("aligned_dir", help="output path", default='data/aligned_vectors/')
    parser.add_argument("--start-year", type=int, help="start year (inclusive)", default=1900)
    parser.add_argument("--year-inc", type=int, help="year increment", default=1)
    parser.add_argument("--end-year", type=int, help="end year (inclusive)", default=2000)
    parser.add_argument("--disp-year", type=int, help="year to measure displacement from", default=1900)
    args = parser.parse_args()
    years = [num for num in range(args.start_year, args.end_year + 1, args.year_inc)]
    print(years)

    align_vectors(years, args.pretrained_dir, args.aligned_dir)

    list_of_word_groups = [career, family, maths, arts, science, intelligence, appearance, strength, 
                       weakness, professions, professions2]

    computed_biases  = calc_dist_in_directory(args.aligned_dir, male_words, female_words, list_of_word_groups)
    computed_biases = computed_biases.reindex(sorted(computed_biases.columns), axis=1)
    computed_biases.to_excel('gender_bias_1900s_aligned.xlsx')