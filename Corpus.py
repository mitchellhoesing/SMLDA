import csv
import pyLDAvis
import pyLDAvis.gensim
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LdaSeqModel


class Corpus:
    def __init__(self, corpusCSVFile):
        self.corpusCSVFile = corpusCSVFile
        self.corpus = []
        self._readCSV(corpusCSVFile)

    def _sortCorpusBy(self, category):
        self.corpus = sorted(self.corpus, key=lambda row: row[category])

    def runLDA(self, num_topics=10, random_state=100, update_every=1, chunksize=2000, passes=10, alpha="auto"):
        corpusTexts = self._createResearchPaperTextListFromCorpus()
        id2word = corpora.Dictionary(corpusTexts)
        id2wordCorpus = self._idToWordCorpus(corpusTexts, id2word)
        ldaModel = gensim.models.ldamodel.LdaModel(corpus=id2wordCorpus,
                                                   id2word=id2word,
                                                   num_topics=num_topics,
                                                   random_state=random_state,
                                                   update_every=update_every,
                                                   chunksize=chunksize,
                                                   passes=passes,
                                                   alpha=alpha)

        print("Running LDA...")
        vis = pyLDAvis.gensim.prepare(ldaModel, id2wordCorpus, id2word, mds="mmds", R=50)
        pyLDAvis.save_html(vis, 'LDAVisualization.html')
        print("Finished... Results visualization saved to current directory as \"LDAVisualization.html\"")

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

    def runTLDA(self, num_topics=10, random_state=100, chunksize=2000, passes=10, alphas=0.01, time_slice=None):
        self._sortCorpusBy("year")
        corpusTexts = self._createResearchPaperTextListFromCorpus()
        id2word = corpora.Dictionary(corpusTexts)
        id2wordCorpus = self._idToWordCorpus(corpusTexts, id2word)
        print("Running tLDA...")
        tldaModel = LdaSeqModel(corpus=id2wordCorpus, id2word=id2word, num_topics=num_topics, random_state=random_state,
                                chunksize=chunksize, passes=passes, alphas=alphas, time_slice=time_slice)
        tldaModel.print_topics(time=0)
