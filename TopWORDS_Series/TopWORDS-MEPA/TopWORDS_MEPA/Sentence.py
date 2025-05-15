
from decimal import Decimal
from collections import defaultdict
from typing import List, Tuple, Dict, Generator, Set
import numpy as np

from TopWORDS_MEPA.utils import Char, Word, Collocation
from TopWORDS_MEPA.utils import get_uint_dtype
from TopWORDS_MEPA.utils import pad_sequences
from TopWORDS_MEPA.utils import ngrams
from TopWORDS_MEPA.utils import print_error
from TopWORDS_MEPA.Dictionary import Dictionary

class Sentence:
    __slots__ = ['sent_string', 'len', 'bool', 'word_list', 'collocation_list', 'likelihood', 'ner_pos_list']

    def __init__(self, sent_string: str, punctuations: Set[Char], ner_pos_list: List = None):

        self.sent_string = sent_string
        self.len = len(sent_string)

        if self.len == 0:
            self.bool = False
        elif self.len == 1 and sent_string in punctuations:
            self.bool = False
        else:
            self.bool = True

        if self.bool:
            self.word_list = None
            self.collocation_list = None
            self.likelihood = 0.0

        if ner_pos_list:
            self.ner_pos_list = ner_pos_list
        else:
            self.ner_pos_list = []

    def set_word_list(self, word_max_len: int, word2ix: Dict[Word, int]) -> None:
        """
        Returns
        -------
        self.word_list is a 2-D numpy array
        shape: word_num_in_sent * 3
        row: (w_s_p = word_start_position, w_e_p = word_end_position, word_ix)
        word_start_position + word_len = word_end_position
        """

        out = []
        for w_s_p in range(self.len):
            for w_e_p in range(w_s_p + 1, min(w_s_p + word_max_len, self.len) + 1):
                word = self.sent_string[w_s_p:w_e_p]
                word_ix = word2ix.get(word, 0)
                if word_ix:
                    out.append([w_s_p, w_e_p, word_ix])
        out = np.array(out, dtype=np.uint64)
        dtype = get_uint_dtype(out.max())
        self.word_list = out.astype(dtype)
        del out, dtype


    def set_word_list_ner(self, word_max_len: int, word2ix: Dict[Word, int], category2ix) -> None:
        """
        Returns
        -------
        self.word_list is a 2-D numpy array
        shape: word_num_in_sent * 3
        row: (w_s_p = word_start_position, w_e_p = word_end_position, word_ix)
        word_start_position + word_len = word_end_position
        """

        out = []
        self.ner_pos_list = [(word2ix[entity[0]], category2ix[entity[1]], entity[2], entity[3])
                                 for entity in self.ner_pos_list if entity[0] in word2ix]
        pos_list = [(entity[2], entity[3]) for entity in self.ner_pos_list]
        for w_s_p in range(self.len):
            for w_e_p in range(w_s_p + 1, min(w_s_p + word_max_len, self.len) + 1):
                word = self.sent_string[w_s_p:w_e_p]
                word_ix = word2ix.get(word, 0)
                if word_ix:
                    if w_e_p - w_s_p == 1 or (w_s_p, w_e_p) in pos_list:  # 长度为1 或 出现在entity列表中
                        out.append([w_s_p, w_e_p, word_ix])
        out = np.array(out, dtype=np.uint64)
        dtype = get_uint_dtype(out.max())
        self.word_list = out.astype(dtype)
        del out, dtype


    @staticmethod
    def is_valid_collocation(collocation: Collocation, category_num: int) -> bool:
        if all(tag_ix > category_num for tag_ix in collocation):
            return False
        for tag_ix_s in ngrams(collocation, 3):
            if all(tag_ix > category_num for tag_ix in tag_ix_s):
                return False
            if all(tag_ix < category_num for tag_ix in tag_ix_s):
                return False
        for tag_ix_1, tag_ix_2 in ngrams(collocation, 2):
            if tag_ix_1 < category_num and tag_ix_2 < category_num:
                if tag_ix_1 == tag_ix_2:
                    return False
        # if len(collocation) > 1:
        #     if 2 in collocation:
        #         return 1 in collocation
        
        # if len(collocation) == 1:
        #     return False
        # if all(tag_ix < category_num for tag_ix in collocation):
        #     return False
        # for tag_ix_1, tag_ix_2 in bigrams(collocation):
        #     if tag_ix_1 > category_num and tag_ix_2 > category_num:
        #         return False
        return True

    def generate_collocation(self,
                             collocation_max_len: int,
                             category_num: int,
                             category_ixs_of_word: List[List[int]],
                             is_prune_by_prior_entity: bool = False) -> Generator[Collocation, None, None]:
        """
        Returns
        -------
        Generator of collocations
        collocation is 1-D list
        # 如果ner_pos_list为[]， 就自由生成，否则按照ner_pos_list生成
        """
        out = [[] for _ in range(self.len + 1)]
        # 3-D list
        # sent.len * (num of collocations which start at the position) * (shape of one collocation)
        # collocation is 1-D list

        if is_prune_by_prior_entity:
            for entity in self.ner_pos_list:
                category_ixs_of_word[entity[0]] = [entity[1]]

        # backward algorithm
        for w_s_p, w_e_p, word_ix in reversed(self.word_list):
            collocation_current_s = []
            if w_e_p - w_s_p == 1:
                collocation_current_s.append([word_ix + category_num + 1])
            for category_ix in category_ixs_of_word[word_ix]:
                collocation_current_s.append([category_ix])

            if collocation_current_s:
                out[w_s_p].extend(collocation_current_s)

                for collocation_next in out[w_e_p]:
                    if len(collocation_next) < collocation_max_len:
                        out[w_s_p].extend([collocation_current + collocation_next
                                           for collocation_current in collocation_current_s])

        return (tuple(collocation)
                for c_s_p in range(self.len)  # c_s_p = collocation_start_position
                for collocation in out[c_s_p]
                if self.is_valid_collocation(collocation, category_num))

    def set_collocation_list(self,
                             category_num: int,
                             collocation_max_len: int,
                             collocation2ix: Dict[Collocation, int],
                             category_ixs_of_word: List[List[int]],
                             collocation_fragment_set: Set[Collocation]) -> None:
        """
        Returns
        -------
        2-D list
        row:
        (c_s_p = collocation_start_position, c_e_p = collocation_end_position,
         collocation_ix, *word_list_in_collocation)
        padding with zeros
        collocation_string_len = collocation_end_position - collocation_start_position
        """

        out = [[] for _ in range(self.len + 1)]
        out[-1] = [[0, [], []]]
        # 3-D list
        # sent.len * (num of collocations which start at the position) * 3
        # (collocation_string_len, collocation, list of word_ix)
        # collocation is 1-D list

        # backward algorithm
        for w_s_p, w_e_p, word_ix in reversed(self.word_list):
            word_len = w_e_p - w_s_p
            collocation_current_s = []

            if word_len == 1:  # single char in collocation
                tag_ix = word_ix + category_num + 1
                if (tag_ix,) in collocation_fragment_set:
                    collocation_current_s.append([word_len, [tag_ix], []])

            for category_ix in category_ixs_of_word[word_ix]:
                collocation_current_s.append([word_len, [category_ix], [word_ix]])

            out[w_s_p].extend(collocation_current_s)

            for collocation_current in collocation_current_s:
                if collocation_current[1][0] > 0:  # 长的collocation不能包含background
                    for collocation_next in out[w_e_p]:
                        collocation = collocation_next[1]
                        if (0 not in collocation) and (0 < len(collocation) < collocation_max_len):
                            collocation_new = collocation_current[1] + collocation
                            if tuple(collocation_new) in collocation_fragment_set:
                                out[w_s_p].append(
                                    [collocation_current[0] + collocation_next[0],
                                     collocation_new,
                                     collocation_current[2] + collocation_next[2]]
                                )

        collocation_list = []
        for c_s_p in range(self.len):
            for collocation_current in out[c_s_p]:
                collocation = tuple(collocation_current[1])
                collocation_ix = collocation2ix.get(collocation, 0)
                if collocation_ix:
                    collocation_current = [c_s_p, c_s_p + collocation_current[0],
                                           collocation_ix, *collocation_current[2]]
                    collocation_list.append(collocation_current)

        self.collocation_list = pad_sequences(collocation_list)

    def forward_backward(self, dictionary: Dictionary) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute likelihood
        """
        a = np.zeros(self.len + 1, dtype=np.float64)
        b = np.zeros(self.len + 1, dtype=np.float64)

        a[0] = 1.0
        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in self.collocation_list:
            lik_temp = dictionary.rho_value[collocation_ix]
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                lik_temp *= dictionary.theta_value[category_ix, word_ix]

            a[c_e_p] += a[c_s_p] * lik_temp

        b[-1] = dictionary.rho_value[-1]
        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in reversed(self.collocation_list):
            lik_temp = dictionary.rho_value[collocation_ix]
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                lik_temp *= dictionary.theta_value[category_ix, word_ix]

            b[c_s_p] += lik_temp * b[c_e_p]

        self.likelihood = b[0]
        # print(self.lik)
        if self.likelihood < 1e-300:
            return self.forward_backward_use_decimal(dictionary)

        # with open('forward_backward.txt', 'a', encoding='utf-8') as f:
        #     f.write('{}\t{}\n'.format(a[-1] * b[-1], b[0]))

        return a, b

    def forward_backward_use_decimal(self, dictionary: Dictionary) -> Tuple[np.ndarray, np.ndarray]:
        """
        compute likelihood

        Parameters
        ----------
        dictionary

        Returns
        -------

        """
        a = np.array([Decimal() for _ in range(self.len + 1)])
        b = np.array([Decimal() for _ in range(self.len + 1)])

        a[0] = Decimal('1')
        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in self.collocation_list:
            lik_temp = dictionary.rho_value[collocation_ix]
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                lik_temp *= dictionary.theta_value[category_ix, word_ix]
            a[c_e_p] += a[c_s_p] * Decimal.from_float(lik_temp)

        b[-1] = Decimal.from_float(dictionary.rho_value[-1])
        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in reversed(self.collocation_list):
            lik_temp = dictionary.rho_value[collocation_ix]
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                lik_temp *= dictionary.theta_value[category_ix, word_ix]
            b[c_s_p] += Decimal.from_float(lik_temp) * b[c_e_p]

        self.likelihood = b[0]
        if self.likelihood.is_zero():
            print_error(self.sent_string + '\nThis sentence has likelihood zero.')
            # 如果这个句子的likelihood等于0，手动把单字词的词频调高到1e-300
            dictionary.rho_value += 1e-300
            dictionary.theta_value += 1e-300
            return self.forward_backward_use_decimal(dictionary)
            # print_error(self.sent_string + '\nThis sentence has likelihood zero.')

        return a, b

    def update_para(self, dictionary: Dictionary) -> None:
        a, b = self.forward_backward(dictionary)
        b /= self.likelihood

        if isinstance(self.likelihood, float):
            for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in self.collocation_list:
                lik_temp = dictionary.rho_value[collocation_ix]
                for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                                word_list_in_collocation):
                    lik_temp *= dictionary.theta_value[category_ix, word_ix]
                lik_temp = a[c_s_p] * b[c_e_p] * lik_temp

                dictionary.rho_new[collocation_ix] += lik_temp
                for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                                word_list_in_collocation):
                    dictionary.theta_new[category_ix, word_ix] += lik_temp
        else:
            for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in self.collocation_list:
                lik_temp = dictionary.rho_value[collocation_ix]
                for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                                word_list_in_collocation):
                    lik_temp *= dictionary.theta_value[category_ix, word_ix]
                lik_temp = float(
                    a[c_s_p] * b[c_e_p] * Decimal.from_float(lik_temp)
                )

                dictionary.rho_new[collocation_ix] += lik_temp
                for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                                word_list_in_collocation):
                    dictionary.theta_new[category_ix, word_ix] += lik_temp

    def prune(self, dictionary: Dictionary) -> None:
        mask = np.ones(len(self.collocation_list), dtype=np.bool_)
        for index, (_, _, collocation_ix, *word_list_in_collocation) in enumerate(self.collocation_list):
            if not dictionary.collocation_bool[collocation_ix]:
                mask[index] = False
                continue
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                if not dictionary.category_word_soft_bool[category_ix, word_ix]:
                    mask[index] = False
                    break
        # if len(set(index_bool)) != len(index_bool):
        #     raise ValueError(self.sent_string + '\nlen(set(index_bool)) != len(index_bool)')
        if not mask.all():
            self.collocation_list = self.collocation_list[mask]

            while all(self.collocation_list[:, -1] == 0):
                self.collocation_list = self.collocation_list[:, :-1]

            if self.collocation_list.dtype != np.uint8:
                dtype = get_uint_dtype(self.collocation_list.max())
                if self.collocation_list.dtype != dtype:
                    self.collocation_list = self.collocation_list.astype(dtype)

    def backward(self, dictionary: Dictionary, collocation_list: np.ndarray):
        b = [Decimal() for _ in range(self.len + 1)]
        b[-1] = Decimal.from_float(dictionary.rho_value[-1])
        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in reversed(collocation_list):
            lik_temp = dictionary.rho_value[collocation_ix]
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                lik_temp *= dictionary.theta_value[category_ix, word_ix]
            b[c_s_p] += Decimal.from_float(lik_temp) * b[c_e_p]
        return b[0]

    def compute_score(self, dictionary: Dictionary):
        large_likelihood = self.backward(dictionary, self.collocation_list)
        while large_likelihood.is_zero():
            # print_error(self.sent_string + '\nThis sentence has likelihood zero when computing scores.')
            dictionary.rho_value += 1e-300
            dictionary.theta_value += 1e-300
            large_likelihood = self.backward(dictionary, self.collocation_list)

        # sth is collocation, word, category word pair
        # sth = ('c', collocation_ix) or ('w', word_ix) or ('cw', (category_ix, word_ix))
        sth2mask_index = defaultdict(list)
        for index, (collocation_ix, *word_list_in_collocation) in enumerate(self.collocation_list[:, 2:]):
            sth2mask_index[('c', collocation_ix)].append(index)
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                sth2mask_index[('w', word_ix)].append(index)
                sth2mask_index[('cw', (category_ix, word_ix))].append(index)

        mask_index2sth_list = defaultdict(list)
        for sth, mask_index in sth2mask_index.items():
            mask_index = tuple(np.unique(mask_index))
            mask_index2sth_list[mask_index].append(sth)

        for mask_index, sth_list in mask_index2sth_list.items():
            collocation_list = np.delete(self.collocation_list, mask_index, axis=0)
            small_likelihood = self.backward(dictionary, collocation_list)
            if small_likelihood.is_zero():
                score = np.inf
            else:
                score = (large_likelihood / small_likelihood).ln().__float__()
            for item, item_ix in sth_list:
                if item == 'c':
                    dictionary.collocation_score[item_ix] += score
                elif item == 'w':
                    dictionary.word_score[item_ix] += score
                elif item == 'cw':
                    dictionary.category_word_score[item_ix] += score
                else:
                    raise ValueError

    def compute_approx_score(self, dictionary: Dictionary):
        a, b = self.forward_backward_use_decimal(dictionary)

        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in self.collocation_list:
            lik_temp = dictionary.rho_value[collocation_ix]
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                lik_temp *= dictionary.theta_value[category_ix, word_ix]

            lik_temp = a[c_s_p] * b[c_e_p] * Decimal.from_float(lik_temp)
            small_lik = self.likelihood - lik_temp
            if small_lik.is_signed() or small_lik.is_zero():
                score = np.PINF
            else:
                score = (self.likelihood / small_lik).ln().__float__()

            dictionary.collocation_approx_score[collocation_ix] += score
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                dictionary.category_word_approx_score[category_ix, word_ix] += score

    def decode_by_mle(self, dictionary: Dictionary):
        """
        Viterbi algorithm, backward algorithm
        forwardly compute probability, backwardly find best path

        Returns
        -------
        list of [collocation_ix, word_list_in_collocation]
        word_list_in_collocation is list of word_ix

        """

        prob = np.full(self.len + 1, -np.inf, dtype=np.float64)
        arg = np.zeros(self.len + 1, dtype=get_uint_dtype(len(self.collocation_list)))

        prob[0] = 0.0

        for ix, (c_s_p, c_e_p, collocation_ix, *word_list_in_collocation) in enumerate(self.collocation_list):
            lik_temp = np.log(dictionary.rho_value[collocation_ix])
            for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
                                            word_list_in_collocation):
                lik_temp += np.log(dictionary.theta_value[category_ix, word_ix])
            lik_temp += prob[c_s_p]

            if lik_temp > prob[c_e_p]:
                prob[c_e_p] = lik_temp
                arg[c_e_p] = ix  # ix is the index of self.collocation_list

        if np.isneginf(prob[-1]):
            print_error(self.sent_string + '\nThis sentence cannot be decoded by posterior mode.')

        position = self.len
        mask = []
        while position > 0:
            ix = arg[position]
            mask.append(ix)
            c_s_p, c_e_p, collocation_ix, *word_list_in_collocation = self.collocation_list[ix]
            # dictionary.collocation_seg_count[collocation_ix] += 1
            # for category_ix, word_ix in zip(dictionary.category_ixs_in_collocation[collocation_ix],
            #                                 word_list_in_collocation):
            #     dictionary.word_seg_count[category_ix, word_ix] += 1
            position = c_s_p
        mask.reverse()

        out = self.collocation_list[mask, 2:]
        return out

    def decode_to_string(self, dictionary: Dictionary, mode='mle+pc') -> str:
        out = []
        hidden_state = self.decode_by_mle(dictionary)
        category_num = dictionary.category_num
        for collocation_ix, *word_list_in_collocation in hidden_state:
            out_temp = []
            collocation = dictionary.collocation_list[collocation_ix]
            if len(collocation) == 1:
                word_ix = word_list_in_collocation[0]
                if mode == 'mle':
                    tag_ix = category_ix = collocation[0]
                elif mode == 'mle+pc':  # mle + posterior classification
                    tag_ix = category_ix = dictionary.theta_c_g_w[:, word_ix].argmax()
                    collocation_ix = tag_ix + 1
                else:
                    raise ValueError
                if tag_ix == 0:  # if category is background, do not output category
                    out_temp.append(dictionary.word_list[word_ix])
                else:
                    out_temp.append(f'{dictionary.word_list[word_ix]}\\{dictionary.category_list[category_ix]}')
                dictionary.collocation_seg_count[collocation_ix] += 1  # collocation_seg_count += 1
                dictionary.word_seg_count[category_ix, word_ix] += 1  # word_seg_count += 1

            else:
                word_order = 0
                dictionary.collocation_seg_count[collocation_ix] += 1  # collocation_seg_count += 1
                for tag_ix in collocation:
                    if tag_ix < category_num:
                        category_ix = tag_ix
                        word_ix = word_list_in_collocation[word_order]

                        dictionary.word_seg_count[category_ix, word_ix] += 1  # word_seg_count += 1

                        if tag_ix == 0:  # if category is background, do not output category
                            out_temp.append(dictionary.word_list[word_ix])
                        else:
                            out_temp.append(f'{dictionary.word_list[word_ix]}\\{dictionary.category_list[category_ix]}')

                        word_order += 1
                    elif tag_ix > category_num:
                        out_temp.append(dictionary.word_list[tag_ix - category_num - 1])
                    else:
                        print_error('end collocation appears')

            if len(out_temp) > 1:
                out.append('(' + ' '.join(out_temp) + ')')
            else:
                out.append(' '.join(out_temp))

        return ' '.join(out)

# @memory_profiler.profile
def e_step_in_one_sent(collocation_list_: np.ndarray,
                       rho_value_: np.ndarray,
                       theta_value_: np.ndarray,
                       category_ixs_in_collocation_: Tuple[Tuple[int, ...], ...]):
    sent_len = collocation_list_[-1, 1]  # important!  cannot prune single-char word
    a = np.array([Decimal() for _ in range(sent_len + 1)])
    b = np.array([Decimal() for _ in range(sent_len + 1)])

    a[0] = Decimal('1')
    for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in collocation_list_:
        lik_temp = rho_value_[collocation_ix]
        for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix], word_list_in_collocation):
            lik_temp *= theta_value_[category_ix, word_ix]
        a[c_e_p] += a[c_s_p] * Decimal.from_float(lik_temp)

    b[-1] = Decimal.from_float(rho_value_[-1])
    for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in reversed(collocation_list_):
        lik_temp = rho_value_[collocation_ix]
        for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix], word_list_in_collocation):
            lik_temp *= theta_value_[category_ix, word_ix]
        b[c_s_p] += Decimal.from_float(lik_temp) * b[c_e_p]

    likelihood = b[0]
    while likelihood.is_zero():
        rho_value_ += 1e-300
        theta_value_ += 1e-300

        a = np.array([Decimal() for _ in range(sent_len + 1)])
        b = np.array([Decimal() for _ in range(sent_len + 1)])

        a[0] = Decimal('1')
        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in collocation_list_:
            lik_temp = rho_value_[collocation_ix]
            for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix], word_list_in_collocation):
                lik_temp *= theta_value_[category_ix, word_ix]
            a[c_e_p] += a[c_s_p] * Decimal.from_float(lik_temp)

        b[-1] = Decimal.from_float(rho_value_[-1])
        for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in reversed(collocation_list_):
            lik_temp = rho_value_[collocation_ix]
            for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix], word_list_in_collocation):
                lik_temp *= theta_value_[category_ix, word_ix]
            b[c_s_p] += Decimal.from_float(lik_temp) * b[c_e_p]

        likelihood = b[0]

    b /= likelihood

    rho_new = defaultdict(float)
    theta_new = defaultdict(float)
    for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in collocation_list_:
        lik_temp = rho_value_[collocation_ix]
        for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix], word_list_in_collocation):
            lik_temp *= theta_value_[category_ix, word_ix]
        lik_temp = (a[c_s_p] * b[c_e_p] * Decimal.from_float(lik_temp)).__float__()

        rho_new[collocation_ix] += lik_temp
        for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix], word_list_in_collocation):
            theta_new[category_ix, word_ix] += lik_temp
    return likelihood, rho_new, theta_new


def backward(collocation_list_: np.ndarray,
             rho_value_: np.ndarray,
             theta_value_: np.ndarray,
             category_ixs_in_collocation_: Tuple[Tuple[int, ...], ...],
             sent_len: int):
    b = [Decimal() for _ in range(sent_len + 1)]
    b[-1] = Decimal.from_float(rho_value_[-1])
    for c_s_p, c_e_p, collocation_ix, *word_list_in_collocation in reversed(collocation_list_):
        lik_temp = rho_value_[collocation_ix]
        for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix],
                                        word_list_in_collocation):
            lik_temp *= theta_value_[category_ix, word_ix]
        b[c_s_p] += Decimal.from_float(lik_temp) * b[c_e_p]
    return b[0]

# @memory_profiler.profile
def compute_score_in_one_sent(collocation_list_: np.ndarray,
                              rho_value_: np.ndarray,
                              theta_value_: np.ndarray,
                              category_ixs_in_collocation_: Tuple[Tuple[int, ...], ...]):
    sent_len = collocation_list_[-1, 1]
    large_likelihood = backward(collocation_list_, rho_value_, theta_value_, category_ixs_in_collocation_, sent_len)
    while large_likelihood.is_zero():
        rho_value_ += 1e-300
        theta_value_ += 1e-300
        large_likelihood = backward(collocation_list_, rho_value_, theta_value_, category_ixs_in_collocation_, sent_len)

    # sth is collocation, word, category word pair
    # sth = ('c', collocation_ix) or ('w', word_ix) or ('cw', (category_ix, word_ix))
    sth2mask_index = defaultdict(list)
    for index, (collocation_ix, *word_list_in_collocation) in enumerate(collocation_list_[:, 2:]):
        sth2mask_index[('c', collocation_ix)].append(index)
        for category_ix, word_ix in zip(category_ixs_in_collocation_[collocation_ix],
                                        word_list_in_collocation):
            sth2mask_index[('w', word_ix)].append(index)
            sth2mask_index[('cw', (category_ix, word_ix))].append(index)

    mask_index2sth_list = defaultdict(list)
    for sth, mask_index in sth2mask_index.items():
        mask_index = tuple(np.unique(mask_index))
        mask_index2sth_list[mask_index].append(sth)

    collocation_score = defaultdict(float)
    word_score = defaultdict(float)
    category_word_score = defaultdict(float)
    for mask_index, sth_list in mask_index2sth_list.items():
        small_likelihood = backward(np.delete(collocation_list_, mask_index, axis=0),
                                    rho_value_, theta_value_, category_ixs_in_collocation_, sent_len)
        if small_likelihood.is_zero():
            score = float('inf')
        else:
            score = (large_likelihood / small_likelihood).ln().__float__()
        for item, item_ix in sth_list:
            if item == 'c':
                collocation_score[item_ix] += score
            elif item == 'w':
                word_score[item_ix] += score
            elif item == 'cw':
                category_word_score[item_ix] += score
            else:
                raise ValueError
    return collocation_score, word_score, category_word_score
