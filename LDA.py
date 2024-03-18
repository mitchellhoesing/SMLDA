import csv
import pyLDAvis
import pyLDAvis.gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaSeqModel, ldamodel
import numpy as np


class LDA:
    def __init__(self, corpusCSVFile):
        self.tldaModel = None
        self.corpusCSVFile = corpusCSVFile
        self.corpus = []
        self._readCSV(corpusCSVFile)
        self.corpusTexts = self._createResearchPaperTextListFromCorpus()
        self.id2word = corpora.Dictionary(self.corpusTexts)
        self.id2wordCorpus = self._idToWordCorpus(self.corpusTexts, self.id2word)

    def _sortCorpusBy(self, category):
        self.corpus = sorted(self.corpus, key=lambda row: row[category])

    def LDA(self, num_topics=10, random_state=100, update_every=1, chunksize=2000, passes=10, alpha="auto"):
        ldaModel = ldamodel.LdaModel(corpus=self.id2wordCorpus, id2word=self.id2word, num_topics=num_topics,
                                     random_state=random_state, update_every=update_every,
                                     chunksize=chunksize, passes=passes, alpha=alpha)

        print("Running LDA...")
        vis = pyLDAvis.gensim.prepare(ldaModel, self.id2wordCorpus, self.id2word, mds="mmds", R=50)
        pyLDAvis.save_html(vis, 'LDAVisualization.html')
        print("Finished... Results visualization saved to current directory as \"LDAVisualization.html\"")

    def TLDA(self, num_topics=10, random_state=100, chunksize=2000, passes=10, alphas=0.01, time_slice=None):
        self._sortCorpusBy("year")
        print("Running tLDA...")
        self.tldaModel = LdaSeqModel(corpus=self.id2wordCorpus, id2word=self.id2word, num_topics=num_topics,
                                     random_state=random_state, chunksize=chunksize, passes=passes, alphas=alphas,
                                     time_slice=time_slice)
        print("Finished tLDA...")

    def printTLDATopics(self, timeSlice=0):
        self._checkModelIsNone()
        print(self.tldaModel.print_topics(time=timeSlice))

    def printTLDATopicEvolution(self, topic=0):
        self._checkModelIsNone()
        print(self.tldaModel.print_topic_times(topic=topic))

    def visualizeTLDAResults(self, time=0):
        self._checkModelIsNone()
        # doc_topic, topic_term, doc_lengths, term_frequency, vocab = self.tldaModel.dtm_vis(time=time, corpus=self.corpus)
        #
        # vis_wrapper = pyLDAvis.prepare(topic_term_dists=topic_term, doc_topic_dists=doc_topic, doc_lengths=doc_lengths,
        #                                vocab=vocab, term_frequency=term_frequency)

        # pyLDAvis.display(vis_wrapper)
        vis = pyLDAvis.gensim.prepare(self.tldaModel, self.corpus, dictionary=self.id2word)
        pyLDAvis.save_html(vis, 'tLDAVisualization.html')

    def _checkModelIsNone(self):
        if self.tldaModel is None:
            print("EXITING: Run corpus.TLDA() to generate results before printing results")
            exit()

    @staticmethod
    def _idToWordCorpus(corpusTexts, id2word):
        id2wordCorpus = []
        for researchPaperText in corpusTexts:
            new = id2word.doc2bow(researchPaperText)
            id2wordCorpus.append(new)
        return id2wordCorpus

    def _createResearchPaperTextListFromCorpus(self):
        # Make a list of each research paper text, each of which are a list of words.
        researchPaperTexts = []
        for researchPaper in self.corpus:
            researchPaperWords = []
            for word in researchPaper['text'].split():
                researchPaperWords.append(word)
            researchPaperTexts.append(researchPaperWords)
        return researchPaperTexts

    def _readCSV(self, corpusCSVFile):
        with open(corpusCSVFile, newline='') as fd:
            corpusReader = csv.DictReader(fd)
            for row in corpusReader:
                self.corpus.append(row)


