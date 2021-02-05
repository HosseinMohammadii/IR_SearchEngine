import math
import os
import pickle
import re
import bisect

from base import Frequency
from data import mokassar


class Dictionary:
    main_dict = {}
    """
    {
        token1:{
            doc_1:{
                'frequency': 2,
                'list':[23, 94],
            }
            doc_2:{
                'frequency': 1,
                'list':[35],
            }
        }
    }
    """

    token_doc_frequency_dict = {}

    tf_idf_dict = {}
    """
    {
        doc_1: {
            token1: 0.5,
            token2: 0.1,
            token3: 0.8,
        }
    }
    """

    idf_dict = {}
    """
    {
        doc_1: 0.8,
    }
    """

    champion_dict = {}
    """
    {
        word1: [],
    }
    """
    doc_element_squares_dict = {}
    token_term_frequency_dict = {}
    stop_words = []
    docs_num = 0

    docs_dir = 'sampleDoc/'

    main_all_tokens_num = 0

    def __init__(self, docs_dir: str = None, main_dict_dir: str = None,
                 stop_words_dir: str = None):
        self.docs_dir = docs_dir
        self.main_dict_dir = main_dict_dir
        self.stop_words_dir = stop_words_dir
        self.docs_num = self.docs_num
        self.id2path = {}
        self.path2id = {}
        self.generate_file_paths()

    def generate_file_paths(self):
        doc_id = 1
        for root, d_names, f_names in os.walk(self.docs_dir):
            for f in f_names:
                d = os.path.join(root, f)
                self.id2path[doc_id] = d
                self.path2id[d] = doc_id
                doc_id += 1
        self.docs_num = doc_id


    def get_doc_path_by_id(self, doc_id):
        return self.id2path.get(doc_id, 'Not included')

    def get_doc_id_by_path(self, doc_path):
        return self.path2id.get(doc_path, 'Not included')

    def make_dictionary(self):
        all_tokens_num = 0
        for doc_id in self.id2path.keys():
            with open(self.id2path[doc_id], encoding='utf8') as f:
                line = f.readline()
                cnt = 1
                position = 0
                while line:
                    line = line.strip()
                    normalized_text = self.normalization(line)
                    # print(normalized_text)
                    # tokens = self.tokenization(normalized_text)
                    tokens = self.tokenization(normalized_text)
                    # stemmed = self.stemmer(tokens)
                    for token in tokens:
                        if len(token) < 2:
                            continue
                        # print(token)
                        self.token_term_frequency_dict[token] = self.token_term_frequency_dict.get(token, 0) + 1
                        all_tokens_num = all_tokens_num + 1
                        self.update_dictionary(doc_id, token, position)
                        position += 1

                    line = f.readline()
                    cnt += 1
            self.main_all_tokens_num = all_tokens_num

        # self.remove_stop_words()
        # normalized_text = normalization(text)
        # tokens = tokenization(normalized_text)
        # stemmed = stemmer(tokens)
        # tokens = remove_stop_words(stemmed)
        # for position in range(0, len(tokens)):
        #     word = tokens[position]
        #     update_dictionary(doc_id, word, position)

    def update_dictionary(self, doc_id, word, position):
        if self.main_dict.get(word, None) is None:
            self.main_dict[word] = {}
        if self.main_dict[word].get(doc_id, None) is None:
            self.main_dict[word][doc_id] = {'frequency': 0, 'list': []}

        self.main_dict[word][doc_id]['frequency'] += 1
        self.main_dict[word][doc_id]['list'].append(position)

    def normalization(self, data):
        normal_data = re.sub('\u200c|\u200b|\u200d|\u200e|\u200f|\u202c|\xad|\ufeff|_|\u2067|\u2069|\x7f', ' ', data)
        normal_data = re.sub('[|{}=;&«»%/+*!@#$.؛:",،)(?؟]|-|\d+|[a-zA-Z]', ' ', normal_data)
        normal_data = normal_data.replace("'", " ")
        normal_data = normal_data.replace("ـ", " ")
        normal_data = normal_data.replace("]", " ")
        normal_data = normal_data.replace("[", " ")
        normal_data = normal_data.replace('\n', " ")
        normal_data = normal_data.replace('\r', " ")
        normal_data = re.sub('[ء]', ' ', normal_data)
        normal_data = re.sub('[ؤ]', 'و', normal_data)
        normal_data = re.sub('[ۀ]', 'ه', normal_data)
        normal_data = re.sub('[َ ِ ُ ّ ً ]', ' ', normal_data)
        normal_data = re.sub('[ْْ ]', ' ', normal_data)
        normal_data = re.sub('[ئ]', 'ی', normal_data)
        normal_data = re.sub('[ْي]', 'ی', normal_data)
        normal_data = re.sub('[ك]', 'ک', normal_data)
        normal_data = re.sub('[إاٌآأ]', 'ا', normal_data)
        normal_data = re.sub('[ْ…]', ' ', normal_data)

        return normal_data

    def find_stop_words(self):
        abundance_rate = 1 / 100
        for word, frequency in self.token_term_frequency_dict.items():
            if frequency / self.main_all_tokens_num > abundance_rate and len(
                    self.main_dict[word].keys()) / self.docs_num > 0.6:
                self.stop_words.append(word)

    def remove_stop_words_from_dictionary(self):
        for word in self.stop_words:
            self.main_dict.pop(word)

    def remove_stop_words(self, tokens):
        new_tokens = []
        for token in tokens:
            if token not in self.stop_words:
                new_tokens.append(token)
        return new_tokens

    def stemmer(self, tokens):
        return tokens
        verbAffix = ["*ش", "*نده", "*ا", "*ار", "وا*", "اثر*", "فرو*", "پیش*", "گرو*", "*ه", "*گار", "*ن"]
        ends = ['ات',
                'ان',
                'ترین',
                'تر',
                'م', 'ت', 'ش', 'یی', 'ی', 'ها', 'ٔ', '‌ا', '‌']

        suffix = ["كار", "ناك", "وار", "آسا", "آگین", "بار", "بان", "دان", "زار", "سار", "سان", "لاخ", "مند", "دار",
                  "مرد",
                  "کننده", "گرا", "نما", "متر"]

        prefix = ["بی", "با", "پیش", "غیر", "فرو", "هم", "نا", "یک"]

        def stem(word):
            for end in ends:
                if word.endswith(end):
                    word = word[:-len(end)]

            if word.endswith('ۀ'):
                word = word[:-1] + 'ه'

            return word

        new_tokens = []
        for token in tokens:
            if token in mokassar:
                new_tokens.append(mokassar[token])
            else:
                new_tokens.append(token)
            j = 0
            # for affix in verbAffix:
            #     if (j == 0 and (token[-1] == 'ا' or token[-1] == 'و')):
            #         sTemp = affix.replace("*", token + "ی")
            #     else:
            #         sTemp = affix.replace("*", token)
            #
            #     if normalizeValidation(sTemp, True):
            #         return affix
            #     j = j + 1
            # return ""
        return new_tokens

    def tokenization(self, text):
        return text.split(' ')

    def get_token_docs_ids(self, token):
        token_info = self.main_dict.get(token, None)
        if token_info:
            return token_info.keys()
        return []

    def get_token_champion_docs_ids(self, token, threshold=1):
        return [freq.doc for freq in self.champion_dict.get(token, []) if freq.frequency >= threshold]

    def save_main_dict(self, save_dir, name):
        a_file = open(save_dir + '/' + name + '.pkl', "wb")
        pickle.dump(self.main_dict, a_file)
        a_file.close()

    def load_main_dict(self, save_dir, name):
        a_file = open(save_dir + '/' + name + '.pkl', "rb")
        self.main_dict = pickle.load(a_file)
        a_file.close()

    def save_stop_words(self, save_dir, name):
        a_file = open(save_dir + '/' + name + '.pkl', "wb")
        pickle.dump(self.stop_words, a_file)
        a_file.close()

    def load_stop_words(self, save_dir, name):
        a_file = open(save_dir + '/' + name + '.pkl', "rb")
        self.stop_words = pickle.load(a_file)
        a_file.close()

    def save_frequency_dict(self, save_dir, name):
        a_file = open(save_dir + '/' + name + '.pkl', "wb")
        pickle.dump(self.token_term_frequency_dict, a_file)
        a_file.close()

    def load_frequency_dict(self, save_dir, name):
        a_file = open(save_dir + '/' + name + '.pkl', "rb")
        self.token_term_frequency_dict = pickle.load(a_file)
        a_file.close()

    def fill_tf_idf_empty_dict(self):
        for word in self.main_dict.keys():
            for doc_id in self.main_dict[word]:
                self.tf_idf_dict[doc_id] = {}

    def fill_token_doc_frequency(self):
        for word in self.main_dict.keys():
            self.token_doc_frequency_dict[word] = len(self.main_dict[word].keys())
            self.idf_dict[word] = self.calculate_idf(self.token_doc_frequency_dict[word])

    def fill_tf_idf_dict(self):
        self.fill_token_doc_frequency()
        self.fill_tf_idf_empty_dict()
        for word in self.main_dict.keys():
            for doc_id in self.main_dict[word].keys():
                tfidf = self.calculate_tf(self.main_dict[word][doc_id]['frequency']) \
                        * self.calculate_idf(self.token_doc_frequency_dict[word])

                self.tf_idf_dict[doc_id][word] = tfidf
                # if tfidf < 0:
                #     print('tf_idf is negative')
                self.doc_element_squares_dict[doc_id] = self.doc_element_squares_dict.get(doc_id, 0) + tfidf ** 2

    def normalize_tf_idf(self):
        for doc_id in self.tf_idf_dict.keys():
            doc_vector_size = math.sqrt(self.doc_element_squares_dict[doc_id])
            for word in self.tf_idf_dict[doc_id].keys():
                self.tf_idf_dict[doc_id][word] /= doc_vector_size

    def calculate_tf(self, frequency):
        tf = float(1) + math.log10(frequency)
        if tf < 0:
            print('tf is negative', frequency)
        return tf

    def calculate_idf(self, frequency):
        idf = math.log10(self.docs_num / frequency)
        if idf <= 0:
            print('idf is negative', self.docs_num / frequency)
        return idf

    def fill_champion_dict(self, champion_list_size):
        for token, token_info in self.main_dict.items():
            l = []
            for doc_id, tok_doc_info in token_info.items():
                bisect.insort(l, Frequency(doc_id, tok_doc_info['frequency']))
            l.reverse()
            self.champion_dict[token] = l[:champion_list_size]

    def add_doc(self):
        """to use just and only adding few docs"""
        pass


if __name__ == '__main__':
    d = Dictionary(10, 'sampleDoc/', [1,
                                      2, 3, 4, 5, 6, 7, 8, 9,
                                      ])
    d.make_dictionary()
    # print(d.frequency_dict)
    # print(d.main_dict)
    # print(d.main_dict.get('ریال', ''))
    # print(d.get_token_docs_ids('ریال'))
    # print(sorted(d.frequency_dict.items(), key=lambda x: x[1], reverse=False))
    d.find_stop_words()
    # print(d.stop_words)
    d.remove_stop_words_from_dictionary()
    d.fill_tf_idf_dict()
    d.normalize_tf_idf()
    d.fill_champion_dict(10)
    print(d.token_doc_frequency_dict['جام'])
    print(d.tf_idf_dict[1])
    print(d.champion_dict['ریال'])
    # d.save_main_dict(d.main_dict_dir, 'DICT')
