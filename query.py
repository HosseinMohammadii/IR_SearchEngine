from collect import Dictionary
from cluster import Cluster
import os

import heapq


from base import Score


class Query:

    def __init__(self, *args, **kwargs):
        pass

    def pre_process_text(self, dictionary: Dictionary,  text: str):
        normalized_text = dictionary.normalization(text)
        tokens = dictionary.tokenization(normalized_text)
        stemmed_tokens = dictionary.stemmer(tokens)
        return dictionary.remove_stop_words(stemmed_tokens)

    def process_query_text(self, dictionary: Dictionary, text: str, k: int, clusters: iter = None):
        """
        :param dictionary: gathered information
        :param text: query text
        :param k: determines how many docs is required
        :param clusters: a iterable object that contains Cluster objects
        :return:
        """
        tokens = self.pre_process_text(dictionary, text)
        frequency_dict = self.create_frequency_dict(tokens)
        # print('f dict', frequency_dict)
        tf_idf = self.create_tf_idf(dictionary, frequency_dict)
        # print('tf idf', tf_idf)
        normalized_tf_idf = self.normalize_tf_idf(tf_idf)
        # print('norm tf idf', normalized_tf_idf)

        return self.query(dictionary, normalized_tf_idf, k, clusters)

    def query(self, dictionary: Dictionary, norm_tf_idf: dict, k, clusters: iter = None) -> list:
        """
        :param dictionary: gathered information.
        :param norm_tf_idf: normalized tf idf of query text
        :param k: determines how many docs is required
        :param clusters: a iterable object that contains Cluster objects
        :return:
        """
        # ress = {}

        champion_docs = self.get_champions(dictionary, norm_tf_idf.keys(), threshold=1)

        if clusters is not None and len(clusters) > 0:
            # print('in query clusters is ', clusters)
            cluster_docs = self.get_cluster_docs(dictionary, norm_tf_idf, clusters)
            # print(cluster_docs)
            docs = list(set(cluster_docs) & set(champion_docs))
        else:
            # docs = list(set([]) & set(champion_docs))
            docs = champion_docs

        print('query docs', docs)

        heap_res = self.calculate_scores(dictionary, norm_tf_idf, docs)
        # print('raw result', ress)
        return self.select_results(heap_res, k)

    def calculate_scores(self, dictionary: Dictionary, norm_tf_idf, docs, is_cluster=False, *args, **kwargs):
        heap_res = []
        heapq.heapify(heap_res)
        # print('norm_tf_id in calculate scores', norm_tf_idf)
        # print('docs in calculate scores', docs)
        if is_cluster:
            cluster_index = -1
            for t_cluster in kwargs.pop('clusters'):
                cluster_index += 1
                score = 0
                for token in norm_tf_idf:
                    score += norm_tf_idf[token] * t_cluster.center_tf_idf.get(token, 0)
                    if score > 0:
                        heapq.heappush(heap_res, Score(cluster_index, 'In memory.it is cluster mode', score))
        else:
            for doc in docs:
                score = 0
                for token in norm_tf_idf:
                    score += norm_tf_idf[token] * dictionary.tf_idf_dict[doc].get(token, 0)
                if score > 0:
                    heapq.heappush(heap_res, Score(doc, dictionary.get_doc_path_by_id(doc), score))
        return heap_res

    def get_champions(self, dictionary, tokens, threshold=1) -> list:
        """
        :returns ordered list of doc id by most including tokens
        """
        doc_results: dict = {}
        for token in tokens:
            doc_ids = dictionary.get_token_champion_docs_ids(token, threshold)
            for doc_id in doc_ids:
                doc_results[doc_id] = doc_results.get(doc_id, 0) + 1
        return [t[0] for t in sorted(doc_results.items(), key=lambda x: x[1], reverse=True) if t[1] >= threshold]

    def get_cluster_docs(self, dictionary: Dictionary, norm_tf_idf, clusters: iter) -> list:
        """
        :returns ordered list of doc id by most including tokens
        """
        centers = []
        for cluster in clusters:
            centers.append(cluster.center_doc_id)

        # print('clusters centers', clusters, centers)

        heap_res = self.calculate_scores(dictionary, norm_tf_idf, centers, is_cluster=True, clusters=clusters)
        # print(heap_res)
        first_score: Score = self.select_results(heap_res, 1)[0]
        match_cluster = clusters[first_score.doc]

        return match_cluster.doc_ids

    def select_results(self, heapified_results: list, k: int) -> list:
        k = k if k <= len(heapified_results) else len(heapified_results)
        final_results = []
        for i in range(0, k):
            final_results.append(heapq.heappop(heapified_results))
        return final_results

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
    d = Dictionary('sampleDoc/',)
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

    clusters = []

    for cat in ['بهداشت', 'تاریخ', 'ریاضیات', 'فناوری', 'فیزیک']:
    # for cat in ['بهداشت', ]:
        cluster_ids = []
        cluster_dir = os.path.join("sampleDoc/clustered", cat)
        for dirpath, dirs, files in os.walk(cluster_dir):
            for f in files:
                cluster_ids.append(d.get_doc_id_by_path(os.path.join(cluster_dir, f)))
        cluster = Cluster(d.tf_idf_dict, cluster_ids)
        cluster.find_center()
        clusters.append(cluster)

    print(q.process_query_text(d, 'بیماری', 20, clusters))
