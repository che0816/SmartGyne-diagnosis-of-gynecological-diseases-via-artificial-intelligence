
from typing import List, Tuple
from collections import defaultdict
import numpy as np

from TopWORDS_MEPA.utils import Category, Word, Collocation
from TopWORDS_MEPA.utils import get_uint_dtype


class Dictionary:
    def initialize_word_dictionary(self, category_list: List[Category],
                                   word_list: List[Word], word_raw_count: List[int],
                                   category_word_hard_bool: np.ndarray, category_word_soft_bool: np.ndarray,
                                   sparse_posterior: np.ndarray):
        # category
        self.category_list = list(category_list.copy())
        self.category_num = len(self.category_list)
        self.category2ix = {category: ix for ix, category in enumerate(self.category_list)}
        # check
        assert (self.category_num == category_word_hard_bool.shape[0] == category_word_soft_bool.shape[0]
                == sparse_posterior.shape[0]), 'Input category_num arguments are not consistent.'
        assert (len(word_list)
                == len(word_raw_count) == category_word_hard_bool.shape[1] == category_word_soft_bool.shape[1]
                == sparse_posterior.shape[1]), 'Input word_num arguments are not consistent.'

        # word
        if '' in word_list:  # default '' not in word_list
            raise ValueError('Empty string in word list.')

        self.word_list = [''] + word_list
        self.word_num = len(self.word_list)
        self.word_max_len = max(len(word) for word in self.word_list)

        self.word_raw_count = np.zeros(self.word_num, dtype=np.uint64)
        self.word_raw_count[1:] = word_raw_count
        dtype = get_uint_dtype(self.word_raw_count.max())
        self.word_raw_count = self.word_raw_count.astype(dtype)

        self.word_bool = np.ones(self.word_num, dtype=np.bool_)
        self.word_bool[0] = False

        # category word
        # category_word_hard_bool for generating collocation
        self.category_word_hard_bool = np.zeros((self.category_num, self.word_num), dtype=np.bool_)
        self.category_word_hard_bool[:, 1:] = category_word_hard_bool

        # category_word_soft_bool for setting collocation list in sentence, and EM (with prune)
        self.category_word_soft_bool = np.zeros((self.category_num, self.word_num), dtype=np.bool_)
        self.category_word_soft_bool[:, 1:] = category_word_soft_bool

        self.sparse_posterior = np.zeros((self.category_num, self.word_num), dtype=np.float64)
        self.sparse_posterior[:, 1:] = sparse_posterior

    def get_category_ixs_in_collocation(self) -> Tuple[Tuple[int, ...], ...]:
        return tuple(tuple(tag_ix
                           for tag_ix in collocation
                           if tag_ix < self.category_num)
                     for collocation_ix, collocation in enumerate(self.collocation_list)
                     )

    def initialize_collocation_dictionary(self,
                                          collocation_list: List[Collocation],
                                          collocation_raw_hard_count: List[int]):
        assert len(collocation_list) == len(collocation_raw_hard_count)
        collocation_raw_hard_count = {collocation: count for collocation, count in zip(collocation_list,
                                                                                       collocation_raw_hard_count)}

        for category_ix in range(self.category_num + 1):
            collocation = (category_ix,)
            if collocation in collocation_list:
                collocation_list.remove(collocation)

        collocation = tuple()
        if collocation in collocation_list:
            collocation_list.remove(collocation)

        self.collocation_list = [tuple()] + [(category_ix,)
                                             for category_ix in range(self.category_num)
                                             ] + collocation_list + [(self.category_num,)]
        # 首项的意义：方便在查找的时候，invalid Collocation的index设为0。用在collocation2ix.get(collocation, 0)

        self.collocation_num = len(self.collocation_list)
        self.collocation_max_len = max(len(collocation) for collocation in self.collocation_list)

        self.collocation_raw_hard_count = np.fromiter(
            (collocation_raw_hard_count.get(collocation, 0) for collocation in self.collocation_list),
            dtype=np.uint64, count=self.collocation_num)
        dtype = get_uint_dtype(self.collocation_raw_hard_count.max())
        self.collocation_raw_hard_count = self.collocation_raw_hard_count.astype(dtype)

        self.collocation_bool = np.ones(self.collocation_num, dtype=np.bool_)
        self.collocation_bool[0] = False

        self.category_ixs_in_collocation = self.get_category_ixs_in_collocation()

    def __init__(self):

        # category
        self.category_list = None
        self.category_num = None
        self.category2ix = None

        # word
        self.word_list = None
        self.word_num = None
        self.word_max_len = None
        self.word_raw_count = None
        self.word_seg_count = None
        self.word_bool = None
        self.word_score = None

        self.theta_w = None

        # category word
        self.category_word_hard_bool = None
        self.category_word_soft_bool = None
        self.sparse_posterior = None
        self.category_word_score = None
        self.category_word_approx_score = None

        self.theta_value = None
        self.theta_new = None
        self.theta_c_g_w = None  # c given w

        # collocation
        self.collocation_list = None
        self.collocation_num = None
        self.collocation_max_len = None

        self.collocation_raw_hard_count = None
        self.collocation_raw_soft_count = None
        self.collocation_raw_modified_soft_count = None
        self.collocation_seg_count = None

        self.collocation_bool = None
        self.collocation_score = None
        self.collocation_approx_score = None

        self.category_ixs_in_collocation = None
        self.rho_value = None
        self.rho_new = None

        # log likelihood
        self.log_likelihood = 0.0

    def get_theta_c(self):
        theta_c_count = np.zeros(self.category_num, dtype=np.uint64)
        for category_ixs in self.category_ixs_in_collocation:
            for category_ix in category_ixs:
                theta_c_count[category_ix] += 1
        theta_c = theta_c_count / theta_c_count.sum()
        return theta_c

    def get_theta_value(self):
        theta_value = self.theta_c_g_w * self.theta_w
        for row in theta_value:
            row /= row.sum()
        return theta_value

    def initialize_parameters(self, collocation_raw_soft_count):
        self.collocation_raw_soft_count = collocation_raw_soft_count

        collocation_len2collocation_list = defaultdict(list)
        for collocation in self.collocation_list:
            collocation_len = len(collocation)
            collocation_len2collocation_list[collocation_len].append(collocation)

        collocation2ix = {collocation: ix for ix, collocation in enumerate(self.collocation_list)}

        collo_raw_modi_soft_count = self.collocation_raw_soft_count.copy()
        for collocation_len in range(self.collocation_max_len, 1, -1):
            for collocation in collocation_len2collocation_list[collocation_len]:
                collocation_ix = collocation2ix[collocation]

                collocation_small = collocation[1:]
                if collocation_small in collocation2ix:
                    collocation_small_ix = collocation2ix[collocation_small]
                    if collo_raw_modi_soft_count[collocation_ix] > collo_raw_modi_soft_count[collocation_small_ix]:
                        collo_raw_modi_soft_count[collocation_small_ix] = collo_raw_modi_soft_count[collocation_ix]

                collocation_small = collocation[:-1]
                if collocation_small in collocation2ix:
                    collocation_small_ix = collocation2ix[collocation_small]
                    if collo_raw_modi_soft_count[collocation_ix] > collo_raw_modi_soft_count[collocation_small_ix]:
                        collo_raw_modi_soft_count[collocation_small_ix] = collo_raw_modi_soft_count[collocation_ix]

        if collo_raw_modi_soft_count[1] != collo_raw_modi_soft_count.max():
            collo_raw_modi_soft_count[1] = collo_raw_modi_soft_count[2:].sum()

        self.collocation_raw_modified_soft_count = collo_raw_modi_soft_count

        self.rho_value = self.collocation_raw_modified_soft_count.astype(np.float64)
        self.rho_value /= self.rho_value.sum()

        self.theta_w = self.word_raw_count / self.word_raw_count.sum()
        self.theta_c_g_w = self.sparse_posterior.copy()

        self.theta_value = self.get_theta_value()

    def prune_by_para(self,
                      prune_by_para_threshold_collocation: float = 1e-8,
                      prune_by_para_threshold_word: float = 1e-8):

        active_collocation_num = self.collocation_bool.sum()
        for collocation_ix, collocation in enumerate(self.collocation_list):
            if self.collocation_bool[collocation_ix] and len(collocation) > 1:
                if self.rho_value[collocation_ix] < prune_by_para_threshold_collocation:
                    self.collocation_bool[collocation_ix] = False
        is_prune_collocation = True if self.collocation_bool.sum() < active_collocation_num else False

        active_word_num = self.word_bool.sum()
        for word_ix, word in enumerate(self.word_list):
            if self.word_bool[word_ix] and len(word) > 1:
                if ((self.theta_value[:, word_ix].max() < prune_by_para_threshold_word)
                        or (self.theta_w[word_ix] < prune_by_para_threshold_word)):
                    self.word_bool[word_ix] = False
                    self.category_word_soft_bool[:, word_ix] = False
        is_prune_word = True if self.word_bool.sum() < active_word_num else False

        is_prune_by_para = is_prune_collocation or is_prune_word
        return is_prune_by_para
        # for word_ix, word in enumerate(self.word_list):
        #     if self.word_bool[word_ix]:
        #         for category_ix in range(0 if len(word) > 1 else 1, self.category_num):
        #             if self.category_word_soft_bool[category_ix, word_ix] \
        #                     and self.theta_value[category_ix, word_ix] < prune_by_para_threshold_word:
        #                 self.category_word_soft_bool[category_ix, word_ix] = False
        #         self.word_bool[word_ix] = self.category_word_soft_bool[:, word_ix].any()

    def prune_by_score(self,
                       prune_by_score_threshold_collocation: float = 3.3174483,
                       prune_by_score_threshold_word: float = 3.3174483):
        active_collocation_num = self.collocation_bool.sum()
        for collocation_ix, collocation in enumerate(self.collocation_list):
            if self.collocation_bool[collocation_ix] and len(collocation) > 1:
                if self.collocation_score[collocation_ix] < prune_by_score_threshold_collocation:
                    self.collocation_bool[collocation_ix] = False
        is_prune_collocation = True if self.collocation_bool.sum() < active_collocation_num else False

        active_word_num = self.word_bool.sum()
        for word_ix, word in enumerate(self.word_list):
            if self.word_bool[word_ix] and len(word) > 1:
                if self.word_score[word_ix] < prune_by_score_threshold_word:
                    self.word_bool[word_ix] = False
                    self.category_word_soft_bool[:, word_ix] = False
        is_prune_word = True if self.word_bool.sum() < active_word_num else False

        is_prune_by_score = is_prune_collocation or is_prune_word
        return is_prune_by_score
        # for word_ix, word in enumerate(self.word_list):
        #     if self.word_bool[word_ix]:
        #         for category_ix in range(0 if len(word) > 1 else 1, self.category_num):
        #             if self.category_word_soft_bool[category_ix, word_ix] \
        #                     and self.category_word_score[category_ix, word_ix] < prune_by_score_threshold_word:
        #                 self.category_word_soft_bool[category_ix, word_ix] = False
        #         self.word_bool[word_ix] = self.category_word_soft_bool[:, word_ix].any()
