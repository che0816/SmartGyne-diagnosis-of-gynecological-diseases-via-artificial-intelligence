import argparse
import json
import os
import sys
print(os.getcwd())
sys.path.append(os.getcwd())
from TopWORDS_MEPA.CollocationDiscovery import CollocationDiscovery
from TopWORDS_MEPA.utils import mkdir_chdir

def run_topwords_mepa(args):
    # 从args获取参数

    with open(args.punctuation_dir, 'r', encoding='utf-8') as f:
        punctuations = set([word.strip() for word in f.readlines()] + ['\n'])

    prior_info = {
        "prior_word_info": {},
        "prior_word_category_setting": "add",
        "prior_collocation_info": []
    }

    if args.background_words_dir:
        with open(args.background_words_dir, 'r', encoding='utf-8') as f:
            background_words = [word.strip() for word in f.readlines()]
        # print(background_words)
        for word in background_words:
            prior_info['prior_word_info'][word] = []

    with open(args.technical_terms_dir, encoding='utf-8') as jf:
        technical_terms = json.load(jf)

    if not os.path.isabs(args.input_text_dir):
        input_text_dir = os.path.abspath(args.input_text_dir)
    else:
        input_text_dir = args.input_text_dir
    print(input_text_dir)

    mkdir_chdir(args.output_dir)

    # 使用从args获取的参数初始化模型
    model = CollocationDiscovery(
        text_file=input_text_dir,
        technical_terms=technical_terms,
        prior_info=prior_info,
        punctuations=punctuations,
        word_max_len_for_screening=args.word_max_len_for_screening,
        word_min_freq_for_screening=args.word_min_freq_for_screening,
        word_max_len_for_screening_tt=args.word_max_len_for_screening_tt,
        word_min_freq_for_screening_tt=args.word_min_freq_for_screening_tt,
        screen_tt_threshold=args.screen_tt_threshold,
        collocation_max_len_for_screening=args.collocation_max_len_for_screening,
        collocation_min_freq_for_screening=args.collocation_min_freq_for_screening,
        num_of_open_categories_of_a_word=args.num_of_open_categories_of_a_word,
        min_prob_in_nb_smooth=args.min_prob_in_nb_smooth,
        num_of_processes=args.num_of_processes,
        alpha=args.alpha
    )

    # 使用从args获取的参数调用em_update
    model.em_update(
        em_iteration_num=args.em_iteration_num,
        prune_by_count_threshold_collocation=args.prune_by_count_threshold_collocation,
        prune_by_count_threshold_word=args.prune_by_count_threshold_word,
        prune_by_para_threshold_collocation=args.prune_by_para_threshold_collocation,
        prune_by_para_threshold_word=args.prune_by_para_threshold_word,
        prune_by_score_iteration_num=args.prune_by_score_iteration_num,
        em_iteration_num_in_prune_by_score=args.em_iteration_num_in_prune_by_score
    )

    model.output_decoded_result(mode='mle+pc')
    model.output_dictionary_result(is_compute_posterior=True, is_compute_score=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run TopWORDS-MEPA algorithm with configurable parameters.')
    # 输入/输出参数
    parser.add_argument('--input_text_dir', type=str,
                        default=r'.\examples\data\HSD_tiny\HSD_tiny.txt',
                        help='Directory of input text file')
    parser.add_argument('--technical_terms_dir', type=str,
                        default=r'.\examples\data\HSD_tiny\technical_terms.json',
                        help='Directory of technical terms JSON file')
    parser.add_argument('--output_dir', type=str,
                        default=r'.\examples\output\HSD_tiny',
                        help='Output directory')
    parser.add_argument('--background_words_dir', type=str,
                        default=r'.\examples\data\HSD_tiny\background_words.txt',
                        help='Directory containing TopWORDS dictionary')
    parser.add_argument('--punctuation_dir', type=str,
                        default=r'.\examples\data\HSD_tiny\punctuations.txt',
                        help='Directory punctuations text file')
    parser.add_argument('--num_of_processes', type=int, default=4,
                        help='Number of parallel processes')

    # CollocationDiscovery参数
    parser.add_argument('--word_max_len_for_screening', type=int, default=1,
                        help='Max word length for screening')
    parser.add_argument('--word_min_freq_for_screening', type=int, default=1,
                        help='Min word frequency for screening')
    parser.add_argument('--word_max_len_for_screening_tt', type=int, default=8,
                        help='Max word length for technical terms screening')
    parser.add_argument('--word_min_freq_for_screening_tt', type=int, default=5,
                        help='Min word frequency for technical terms screening')
    parser.add_argument('--screen_tt_threshold', type=float, default=0.5,
                        help='Threshold for technical terms screening')
    parser.add_argument('--collocation_max_len_for_screening', type=int, default=5,
                        help='Max collocation length for screening')
    parser.add_argument('--collocation_min_freq_for_screening', type=int, default=100,
                        help='Min collocation frequency for screening')
    parser.add_argument('--num_of_open_categories_of_a_word', type=int, default=None,
                        help='Number of open categories per word')
    parser.add_argument('--min_prob_in_nb_smooth', type=float, default=0.01,
                        help='Minimum probability in Naive Bayes smoothing')
    parser.add_argument('--alpha', type=float, default=10.0,
                        help='Alpha parameter for Dirichlet prior')

    # EM参数
    parser.add_argument('--em_iteration_num', type=int, default=100,
                        help='Number of EM iterations')
    parser.add_argument('--prune_by_count_threshold_collocation', type=float, default=0.1,
                        help='Pruning threshold for collocation counts')
    parser.add_argument('--prune_by_count_threshold_word', type=float, default=0.1,
                        help='Pruning threshold for word counts')
    parser.add_argument('--prune_by_para_threshold_collocation', type=float, default=1e-6,
                        help='Parameter threshold for collocation pruning')
    parser.add_argument('--prune_by_para_threshold_word', type=float, default=1e-6,
                        help='Parameter threshold for word pruning')
    parser.add_argument('--prune_by_score_iteration_num', type=int, default=3,
                        help='Iterations for score-based pruning')
    parser.add_argument('--em_iteration_num_in_prune_by_score', type=int, default=100,
                        help='EM iterations during score-based pruning')

    args = parser.parse_args()
    run_topwords_mepa(args)