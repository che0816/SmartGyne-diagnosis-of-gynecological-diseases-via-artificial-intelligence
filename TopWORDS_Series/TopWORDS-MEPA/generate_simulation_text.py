from os.path import join
import json
import argparse
from TopWORDS_MEPA.utils import *

# 前置变量
category_list = ['background', 'address', 'name', 'office']
entity_category_list = category_list[1:]
category_num = len(category_list)
category2ix = {category: ix for ix, category in enumerate(category_list)}
select_mp_num, select_w_num = 100, 1000


# 命名函数
def mark_meta_data_fun(select_mp_num, select_w_num):
    return f'mp_{select_mp_num}__w_{select_w_num}'

def mark_data_fun(select_mp_num, select_w_num, sentence_num, seed):
    return f'mp_{select_mp_num}__w_{select_w_num}__s_{sentence_num}__seed_{seed}'


def mark_task_fun(task_ix, select_mp_num, select_w_num, sentence_num, seed):
    return f'task_ix_{task_ix}__mp_{select_mp_num}__w_{select_w_num}__s_{sentence_num}__seed_{seed}'


def set_dict(input_dir, output_dir):
    # meta pattern
    collocation_dictionary = pd.read_excel(join(input_dir, f'collocation_dictionary_manual_{select_mp_num}.xlsx'))[:select_mp_num]
    print('load mp dictionary size', len(collocation_dictionary))
    char_set_in_mp = {item
                      for mp in collocation_dictionary['collocation_string']
                      for item in mp.split('+')
                      if len(item) == 1}
    print(f'num of chars in mp: {len(char_set_in_mp)}')

    # words
    word_dictionary = pd.read_excel(join(input_dir, f'word_dictionary_manual_{select_w_num}.xlsx'), header=[0, 1])[:select_w_num]
    print('load word dictionary size', len(word_dictionary))
    word_set = set(word_dictionary[('word', 'word')])
    print('num of non-single-char words selected:', sum(len(w) > 1 for w in word_set))

    char_set = {c for w in word_set for c in w}
    print(f'num of chars in words selected: {len(char_set)}')

    word_set = word_set | char_set | char_set_in_mp
    print(f'num of words selected: {len(word_set)}')
    word_list = sorted(word_set)
    word_num = len(word_list)
    word2ix = {word: ix for ix, word in enumerate(word_list)}

    columns = word_dictionary.columns.copy()
    word_dictionary = pd.concat([
        word_dictionary,
        pd.DataFrame({('word', 'word'): sorted(char_set | char_set_in_mp)})
    ], ignore_index=True)
    word_dictionary = word_dictionary[columns]
    word_dictionary = word_dictionary.set_index(('word', 'word'))

    word_dictionary = word_dictionary.reindex(word_list)
    word_dictionary[pd.isna(word_dictionary)] = 0

    for category in category_list:
        print(word_dictionary[('theta', category)].sum())

    category = category_list[0]
    word_dictionary[('theta', category)][word_dictionary[('theta', category)] == 0] = 0.02 / (word_num - 700)
    for category in entity_category_list:
        word_dictionary[('theta', category)][word_dictionary[('theta', category)] == 0] = 0.01 / (word_num - 100)

    for category in category_list:
        print(word_dictionary[('theta', category)].sum())


    # parameter
    collocation_list = [string_to_collocation(s, category_num, category2ix, word2ix)
                        for s in collocation_dictionary['collocation_string']]
    collocation_num = len(collocation_list)
    print(collocation_dictionary['rho_value'].min())
    print(collocation_dictionary['rho_value'].sum())
    rho_value = collocation_dictionary['rho_value'].values
    theta_value = word_dictionary['theta'].values.T[:-1, :]

    mark_meta_data = mark_meta_data_fun(select_mp_num, select_w_num)
    collocation_dictionary.to_excel(join(output_dir, f'collocation_dictionary__{mark_meta_data}.xlsx'))
    word_dictionary.to_excel(join(output_dir, f'word_dictionary__{mark_meta_data}.xlsx'))

    with open(join(output_dir, f'meta_data__{mark_meta_data}.json'), 'w', encoding='utf8') as jf:
        json.dump({'category_list': category_list,
                   'collocation_list': collocation_list,
                   'collocation_str_list': list(collocation_dictionary['collocation_string']),
                   'rho_value': rho_value.tolist(),
                   'word_list': word_list,
                   'theta_value': theta_value.tolist()
                   }
                  , jf, ensure_ascii=False)


def generate_data(output_dir, sentence_num, seed):

    mark_meta_data = mark_meta_data_fun(select_mp_num, select_w_num)

    with open(join(output_dir, f'meta_data__{mark_meta_data}.json'), encoding='utf8') as jf:
        meta_data = json.load(jf)
        category_list = meta_data['category_list']

        word_list = meta_data['word_list']
        word_num = len(word_list)
        theta_value = np.array(meta_data['theta_value'])

        rho_value = np.array(meta_data['rho_value'])
        collocation_str_list = meta_data['collocation_str_list']
        collocation_list = meta_data['collocation_list']
        collocation_num = len(collocation_list)

    multi_char_word_ix_s = [word_ix for word_ix, word in enumerate(word_list) if len(word) > 1]

    np.random.seed(seed)
    mark_data = mark_data_fun(select_mp_num, select_w_num, sentence_num, seed)  # 生成文件名
    file_name = join(output_dir, f'text__{mark_data}.txt')

    # 生成文件
    collocation_count, word_count = generate_file(collocation_list, collocation_num, rho_value,
                                                  category_num,
                                                  word_list, word_num, theta_value,
                                                  file_name, sentence_num)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sentence_num", type=int, default=100000)
    parser.add_argument("--input_dir", type=str, default=r'./simulation/input')
    parser.add_argument("--output_dir", type=str, default=r'./simulation/output')

    args = parser.parse_args()
    seed = args.seed
    sentence_num = args.sentence_num
    output_dir = args.output_dir
    input_dir = args.input_dir

    set_dict(input_dir, output_dir)
    generate_data(output_dir, sentence_num, seed)
