from math import sqrt


class Cluster:

    def __init__(self, all_tf_idf_dict: dict, doc_ids: iter):
        self.center_doc_id = None
        self.center_tf_idf = None
        self.doc_ids = doc_ids
        self.all_tf_idf_dict = all_tf_idf_dict
        self.docs_num = len(doc_ids)

    def find_center(self):
        tf_avg = {}
        # tf_avg = {
        #     'token1': 530.4,
        #     'token2': 446.2,
        # }

        for doc_id in self.doc_ids:
            for token in self.all_tf_idf_dict[doc_id].keys():
                # if self.all_tf_idf_dict[doc_id][token] < 0:
                #     print('tf_idf is negative')
                tf_avg[token] = tf_avg.get(token, 0) + (self.all_tf_idf_dict[doc_id][token])

        center = None
        min_distance = 1000000
        for doc_id in self.doc_ids:
            doc_distance = 0
            for token in self.all_tf_idf_dict[doc_id].keys():
                doc_distance += tf_avg[token] ** 2 + self.all_tf_idf_dict[doc_id][token] ** 2

            doc_distance = sqrt(doc_distance)
            if doc_distance < min_distance:
                center = doc_id
                min_distance = doc_distance

        self.center_tf_idf = tf_avg
        # print(tf_avg)
        assert center is not None
        self.center_doc_id = center
        return center
