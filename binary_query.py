from collect import Dictionary


class Query:

    def __init__(self, dictionary: Dictionary):
        self.dictionary = dictionary

    def process_query_text(self, dictionary, text):
        normalized_text = dictionary.normalization(text)
        tokens = dictionary.tokenization(normalized_text)
        stemmed_tokens = dictionary.stemmer(tokens)
        cleaned_tokens = dictionary.remove_stop_words(stemmed_tokens)
        results = self.query(cleaned_tokens)
        return self.sort_results(results)

    def query(self, tokens: iter):
        doc_results: dict = {}
        for token in tokens:
            doc_ids = self.dictionary.get_token_docs_ids(token)
            for doc_id in doc_ids:
                doc_results[doc_id] = doc_results.get(doc_id, 0) + 1
        return doc_results

    def sort_results(self, results: dict):
        return [t[0] for t in sorted(results.items(), key=lambda x: x[1], reverse=True)]


if __name__ == '__main__':
    # d = Dictionary()
    d = Dictionary(10, 'sampleDoc/', [1,
                                      2, 3, 4, 5, 6, 7, 8, 9,
                                      ])
    d.make_dictionary()
    # d.load_main_dict()
    # d.load_frequency_dict()
    # d.load_frequency_dict()
    d.find_stop_words()
    d.remove_stop_words_from_dictionary()
    d.fill_tf_idf_dict()
    d.normalize_tf_idf()
    q = Query(d)
    # print(q.query(['ریال', 'دلار']))
    print(q.process_query_text('دلار ریال'))
