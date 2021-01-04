from collect import Dictionary

import heapq


from base import Score


class Query:

    def __init__(self, *args, **kwargs):
        pass

    def pre_process_text(self,dictionary,  text):
        normalized_text = dictionary.normalization(text)
        tokens = dictionary.tokenization(normalized_text)
        stemmed_tokens = dictionary.stemmer(tokens)
        return dictionary.remove_stop_words(stemmed_tokens)

    def process_query_text(self, dictionary, text, k):
        tokens = self.pre_process_text(dictionary, text)
        frequency_dict = self.create_frequency_dict(tokens)
        # print('f dict', frequency_dict)
        tf_idf = self.create_tf_idf(dictionary, frequency_dict)
        # print('tf idf', tf_idf)
        normalized_tf_idf = self.normalize_tf_idf(tf_idf)
        # print('norm tf idf', normalized_tf_idf)

        return self.query(dictionary, normalized_tf_idf, k)

    def query(self, dictionary: Dictionary, norm_tf_idf: dict, k):
        ress = {}
        heap_res = []
        heapq.heapify(heap_res)
        champion_docs = self.get_champions(dictionary, norm_tf_idf.keys())
        print(champion_docs)
        for doc in champion_docs:
            score = 0
            for token in norm_tf_idf:
                score += norm_tf_idf[token] * dictionary.tf_idf_dict[doc].get(token, 0)
            ress[doc] = score
            if score > 0:
                heapq.heappush(heap_res, Score(doc, score))
        # print('raw result', ress)
        return self.select_result(heap_res, k)

    def get_champions(self, dictionary, tokens) -> list:
        """
        :returns ordered list of doc id by most including tokens
        """
        doc_results: dict = {}
        for token in tokens:
            doc_ids = dictionary.get_token_champion_docs_ids(token)
            for doc_id in doc_ids:
                doc_results[doc_id] = doc_results.get(doc_id, 0) + 1
        return [t[0] for t in sorted(doc_results.items(), key=lambda x: x[1], reverse=True)]

    def select_result(self, heapified_results: list, k: int) ->  list:
        k = k if k <= len(heapified_results) else len(heapified_results)
        final_result = []
        for i in range(0, k):
            final_result.append(heapq.heappop(heapified_results))
        return final_result

    def create_frequency_dict(self, tokens):
        # {
        #     token1: 1,
        #     token2: 2,
        # }
        ret: dict = {}
        for token in tokens:
            ret[token] = ret.get(token, 0) + 1
        return ret

    def create_tf_idf(self, dictionary: Dictionary, f_dict):
        # {
        #     token1: 0.7,
        #     token2: 0.8,
        # }
        tf_idf = {}
        for token in f_dict.keys():
            tf_idf[token] = dictionary.calculate_tf(f_dict[token]) * dictionary.idf_dict.get(token, 0)
        return tf_idf

    def normalize_tf_idf(self, tf_idf: dict):
        # {
        #     token1: 0.6,
        #     token2: 0.8,
        # }
        norm_tf_idf = {}
        squares_sum = 0.0
        for tf_idf_value in tf_idf.values():
            squares_sum += tf_idf_value ** 2
        for token in tf_idf.keys():
            norm_tf_idf[token] = tf_idf[token] / squares_sum
        return norm_tf_idf


if __name__ == '__main__':
    # d = Dictionary()
    d = Dictionary(10, 'sampleDoc/', [1,
                                      2, 3, 4, 5, 6, 7, 8, 9, 10
                                      ])
    d.make_dictionary()
    # d.load_main_dict()
    # d.load_frequency_dict()
    # d.load_frequency_dict()
    d.find_stop_words()
    d.remove_stop_words_from_dictionary()
    d.fill_tf_idf_dict()
    d.normalize_tf_idf()
    d.fill_champion_dict(10)
    # print(d.tf_idf_dict[4]['ریال'])
    q = Query()
    # print(q.query(['ریال', 'دلار']))
    print(q.process_query_text(d, 'پرسپولیس', 20))
