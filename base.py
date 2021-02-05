class Score:

    def __init__(self, doc, doc_path, score):
        self.doc = doc
        self.score = -1 * score
        self.doc_path = doc_path

    def __lt__(self, other):
        return self.score < other.score

    def __str__(self):
        return 'Score object. doc:{} with score:{}'.format(str(self.doc_path), str(-1 * self.score)[:7])

    def __repr__(self):
        return 'Score object. doc:{} with score:{}'.format(str(self.doc_path), str(-1 * self.score)[:7])


class Frequency:

    def __init__(self, doc, frequency, token=None):
        self.doc = doc
        self.frequency = frequency
        self.token = token

    def __lt__(self, other):
        return self.frequency < other.frequency

    def __str__(self):
        return 'doc:{} with frequency:{}'.format(str(self.doc), str(self.frequency))

    def __repr__(self):
        return 'doc:{} with frequency:{}'.format(str(self.doc), str(self.frequency))
