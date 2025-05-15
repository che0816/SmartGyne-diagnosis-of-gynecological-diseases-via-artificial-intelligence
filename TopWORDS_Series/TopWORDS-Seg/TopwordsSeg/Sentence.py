
from decimal import Decimal
from collections import defaultdict
from typing import List, Tuple, Dict
import numpy as np

from utils import Word, get_uint_dtype, ngrams, print_error


def prior_prob(sent_prior, i, m):  # [i:m], m > i
    prob = 1
    if m - i == 0:
        return prob
    else:
        for k in range(i, m - 1):
            prob *= (1 - sent_prior[k])
        prob *= sent_prior[m - 1]
    return prob

def prior_prob_with_decimal(sent_prior, i, m):  # [i: m], m > i
    prob = Decimal(1)
    if m - i == 0:
        return prob
    else:
        for k in range(i, m - 1):
            prob *= Decimal.from_float(1 - sent_prior[k])
        prob *= Decimal.from_float(sent_prior[m - 1])
    return prob


class Sentence:
    __slots__ = ["sent_string", "len", "bool", "word_list", "likelihood", "sent_prior"]

    def __init__(self, sent_string: str, punctuations: set):

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
            self.likelihood = 0.0
            self.sent_prior = np.array([1.0/2] * (self.len - 1) + [1.0])

    def set_sent_prior(self, sent_prior: np.ndarray = None):
        # self.sent_prior = None #  self.len 长度的list，每个元素属于(0,1)，代表了这个位置的先验概率
        if sent_prior is not None:
            assert len(sent_prior) == self.len, \
                f"length of sent prior {len(sent_prior)} and sentences {self.len} are not equal:" \
                f" {self.sent_string}, {sent_prior}"
            self.sent_prior = sent_prior

    def set_word_list(self, word_max_len: int, word2ix: Dict[Word, int]) -> None:
        """
        Returns
        -------
        2-D list
        shape: word_num_in_sent * 3
        row: (word_start_position, word_end_position, word_ix)
        word_start_position + word_len = word_end_position
        """
        out = []
        for w_s_p in range(self.len):
            for w_e_p in range(w_s_p + 1, min(w_s_p + word_max_len, self.len) + 1):
                word = self.sent_string[w_s_p:w_e_p]
                if word in word2ix:
                    word_ix = word2ix[word]
                    out.append([w_s_p, w_e_p, word_ix])
        out = np.array(out, dtype=np.uint64)
        dtype = get_uint_dtype(out.max())
        self.word_list = out.astype(dtype)


    def forward_backward(self, theta_value: np.ndarray, sent_prior: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        a = np.zeros(self.len + 1, dtype=np.float64)
        b = np.zeros(self.len + 1, dtype=np.float64)

        a[0] = 1
        for w_s_p, w_e_p, word_ix in self.word_list:  # w_s_p = word_start_position, w_e_p = word_end_position
            a[w_e_p] += a[w_s_p] * theta_value[word_ix] * prior_prob(sent_prior, w_s_p, w_e_p)

        b[-1] = theta_value[-1]
        for w_s_p, w_e_p, word_ix in reversed(self.word_list):
            b[w_s_p] += theta_value[word_ix] * b[w_e_p] * prior_prob(sent_prior, w_s_p, w_e_p)

        self.likelihood = b[0]
        if self.likelihood < 1e-300:
            print(self.sent_string + 'need Decimal')
            return self.forward_backward_use_decimal(theta_value, sent_prior)
        return a, b

    def forward_backward_use_decimal(self, theta_value: np.ndarray, sent_prior: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        a = np.array([Decimal() for _ in range(self.len + 1)])
        b = np.array([Decimal() for _ in range(self.len + 1)])

        a[0] = Decimal(1)
        for w_s_p, w_e_p, word_ix in self.word_list:  # w_s_p = word_start_position, w_e_p = word_end_position
            a[w_e_p] += a[w_s_p] * Decimal.from_float(theta_value[word_ix]) * \
                        prior_prob_with_decimal(sent_prior, w_s_p, w_e_p)

        b[-1] = Decimal.from_float(theta_value[-1])
        for w_s_p, w_e_p, word_ix in reversed(self.word_list):
            b[w_s_p] += Decimal.from_float(theta_value[word_ix]) * b[w_e_p] * \
                        prior_prob_with_decimal(sent_prior, w_s_p, w_e_p)

        self.likelihood = b[0]
        if self.likelihood.is_zero():
            print_error(self.sent_string + '\nThis sentence has likelihood zero when forward_backward.')
            # 如果这个句子的likelihood等于0，手动把单字词的词频调高1e-300
            for i, j, word_ix in self.word_list:
                if j - i == 1:
                    theta_value[word_ix] += 1e-300
            return self.forward_backward_use_decimal(theta_value, sent_prior)

        return a, b

    def update_para(self, theta_value: np.ndarray, theta_new: np.ndarray) -> None:
        a, b = self.forward_backward(theta_value, self.sent_prior)
        a /= self.likelihood

        if isinstance(self.likelihood, float):
            for w_s_p, w_e_p, word_ix in self.word_list:
                theta_new[word_ix] += a[w_s_p] * theta_value[word_ix] * b[w_e_p] * \
                                      prior_prob(self.sent_prior, w_s_p, w_e_p)
        else:
            for w_s_p, w_e_p, word_ix in self.word_list:
                lik_temp = a[w_s_p] * Decimal.from_float(theta_value[word_ix]) * b[w_e_p] * \
                           prior_prob_with_decimal(self.sent_prior, w_s_p, w_e_p)
                theta_new[word_ix] += lik_temp.__float__()


    def prune(self, word_bool: np.ndarray) -> None:
        mask = word_bool[self.word_list[:, 2]]
        if not mask.all():
            self.word_list = self.word_list[mask]
            if self.word_list.dtype != np.uint8:
                dtype = get_uint_dtype(self.word_list.max())
                if self.word_list.dtype != dtype:
                    self.word_list = self.word_list.astype(dtype)

    def backward(self, theta_value: np.ndarray, word_list: np.ndarray, sent_prior: np.ndarray) -> Decimal:
        b = [Decimal() for _ in range(self.len + 1)]

        b[-1] = Decimal.from_float(theta_value[-1])
        for w_s_p, w_e_p, word_ix in reversed(word_list):
            b[w_s_p] += Decimal.from_float(theta_value[word_ix]) * b[w_e_p] * \
                        prior_prob_with_decimal(sent_prior, w_s_p, w_e_p)

        return b[0]

    def compute_score(self, theta_value: np.ndarray, word_score: np.ndarray):
        large_likelihood = self.backward(theta_value, self.word_list, self.sent_prior)
        while large_likelihood.is_zero():
            print_error(self.sent_string + '\nThis sentence has likelihood zero when computing scores.')
            for i, j, word_ix in self.word_list:
                if j - i == 1:
                    theta_value[word_ix] += 1e-300
            large_likelihood = self.backward(theta_value, self.word_list, self.sent_prior)

        for word_ix in set(self.word_list[:, 2]):
            mask = (self.word_list[:, 2] != word_ix)
            word_list = self.word_list[mask]
            small_likelihood = self.backward(theta_value, word_list, self.sent_prior)
            if small_likelihood.is_zero():
                score = np.inf
            else:
                score = (large_likelihood / small_likelihood).ln().__float__()
            word_score[word_ix] += score

    def decode_by_post_mode(self, theta_value: np.ndarray, word_seg_count: np.ndarray) -> List[int]:
        """
        Viterbi algorithm
        forwardly compute probability, backwardly find best path

        Returns
        -------
        list of word_ix
        """
        prob = np.full(shape=self.len + 1, fill_value=np.NINF, dtype=np.float64)
        arg = np.zeros(self.len + 1, dtype=get_uint_dtype(len(self.word_list)))

        prob[0] = 0.0
        for ix, (i, j, word_ix) in enumerate(self.word_list):
            lik_temp = np.log(theta_value[word_ix]) + np.log(prior_prob(self.sent_prior, i, j)) + prob[i]
            if lik_temp > prob[j]:
                prob[j] = lik_temp
                arg[j] = ix  # ix is the index of self.word_list

        if np.isneginf(prob[-1]):
            print_error(self.sent_string + '\nThis sentence cannot be decoded by posterior mode.')
            out_word_ix_list = [word_ix
                                for i, j, word_ix in self.word_list
                                if j - i == 1]
            out_word_01_list = [1 for i in range(self.len)]
        else:
            position = self.len
            out_word_ix_list = []
            out_word_01_list = [0 for i in range(self.len)]
            while position > 0:
                i, j, word_ix = self.word_list[arg[position]]
                out_word_ix_list.append(word_ix)
                out_word_01_list[position - 1] = 1
                position = i
            out_word_ix_list.reverse()

        for word_ix in out_word_ix_list:
            word_seg_count[word_ix] += 1

        temp = np.where([1] + out_word_01_list)[0]
        out_word_list = [self.sent_string[word_start:word_end]
                         for word_start, word_end in ngrams(temp, 2)]
        # out_word_list = [self.dictionary.word_list[word_ix] for word_ix in out_word_ix_list]

        return out_word_01_list, out_word_list, None

    # 问题： 出现字典中没有的词
    def decode_by_post_mean(self, theta_value: np.ndarray, word_seg_count: np.ndarray,
                            word2ix: Dict[Word, int],
                            threshold: float = 0.5) -> List[int]:
        """
        Returns
        -------
        list of word_ix
        """
        a, b = self.forward_backward(theta_value, self.sent_prior)
        # prior_prob = np.array([1.0] + list(self.sent_prior))
        if isinstance(self.likelihood, float):
            post_mean_list = a * b / self.likelihood
            out = np.where(post_mean_list > threshold)[0]
        else:
            # prior_prob = np.array([Decimal(t) for t in prior_prob])
            post_mean_list = a * b / self.likelihood
            out = np.where(post_mean_list > Decimal(threshold))[0]

        out_word_list = [self.sent_string[word_start:word_end]
                         for word_start, word_end in ngrams(out, 2)]

        out_word_ix_list = [word2ix[word] for word in out_word_list if word in word2ix]
        # 有可能不在词典中

        for word_ix in out_word_ix_list:
            word_seg_count[word_ix] += 1
        
        out_word_01_list = []
        for word in out_word_list:
            out_word_01_list.extend([0] * (len(word) - 1) + [1])

        return out_word_01_list, out_word_list, post_mean_list

###############################################################
    
def forward_backward_use_decimal(word_list: np.ndarray, theta_value: np.ndarray, sent_prior: np.ndarray):
    sent_len = word_list[-1, 1]
    a = np.array([Decimal() for _ in range(sent_len + 1)])
    b = np.array([Decimal() for _ in range(sent_len + 1)])

    a[0] = Decimal(1)
    for w_s_p, w_e_p, word_ix in word_list:  # w_s_p = word_start_position, w_e_p = word_end_position
        a[w_e_p] += a[w_s_p] * Decimal.from_float(theta_value[word_ix]) * \
                    prior_prob_with_decimal(sent_prior, w_s_p, w_e_p)

    b[-1] = Decimal.from_float(theta_value[-1])
    for w_s_p, w_e_p, word_ix in reversed(word_list):
        b[w_s_p] += Decimal.from_float(theta_value[word_ix]) * b[w_e_p] * \
                    prior_prob_with_decimal(sent_prior, w_s_p, w_e_p)

    likelihood = b[0]

    while likelihood.is_zero():
        theta_value += 1e-300
        return forward_backward_use_decimal(word_list, theta_value, sent_prior)
    return likelihood, a, b


def update_para(word_list: np.ndarray, theta_value: np.ndarray, sent_prior: np.ndarray):
    likelihood, a, b = forward_backward_use_decimal(word_list, theta_value, sent_prior)
    a /= likelihood
    sent_len = word_list[-1, 1]
    theta_new = defaultdict(float)
    for w_s_p, w_e_p, word_ix in word_list:
        lik_temp = a[w_s_p] * Decimal.from_float(theta_value[word_ix]) * b[w_e_p] * \
                   prior_prob_with_decimal(sent_prior, w_s_p, w_e_p)
        theta_new[word_ix] += lik_temp.__float__()
    return likelihood.ln().__float__(), theta_new


def update_para_job(arg: Tuple[np.ndarray, np.ndarray], theta_value: np.ndarray):
    word_list = arg[0]
    sent_prior = arg[1]
    return update_para(word_list, theta_value, sent_prior)


def backward(word_list: np.ndarray, theta_value: np.ndarray, sent_len: int, sent_prior: np.ndarray) -> Decimal:
    b = [Decimal() for _ in range(sent_len + 1)]
    b[-1] = Decimal.from_float(theta_value[-1])
    for w_s_p, w_e_p, word_ix in reversed(word_list):
        b[w_s_p] += Decimal.from_float(theta_value[word_ix]) * b[w_e_p] * \
                    prior_prob_with_decimal(sent_prior, w_s_p, w_e_p)
    return b[0]


def compute_score(word_list: np.ndarray, theta_value: np.ndarray, sent_prior: np.ndarray):
    sent_len = word_list[-1, 1]
    large_likelihood = backward(word_list, theta_value, sent_len, sent_prior)
    while large_likelihood.is_zero():
        theta_value += 1e-300
        large_likelihood = backward(word_list, theta_value, sent_len, sent_prior)

    word_score = defaultdict(float)
    for word_ix in set(word_list[:, 2]):
        mask = (word_list[:, 2] != word_ix)
        word_list_temp = word_list[mask]
        small_likelihood = backward(word_list_temp, theta_value, sent_len, sent_prior)
        if small_likelihood.is_zero():
            score = np.inf
        else:
            score = (large_likelihood / small_likelihood).ln().__float__()
        word_score[word_ix] += score
    return word_score


def compute_score_job(arg: Tuple[np.ndarray, np.ndarray], theta_value: np.ndarray):
    word_list = arg[0]
    sent_prior = arg[1]
    return compute_score(word_list, theta_value, sent_prior)
