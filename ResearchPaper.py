import spacy
import re


class ResearchPaper:
    def __init__(self, uid, year, text):
        self.nlp = spacy.load('en_core_web_sm')
        self._uid = uid
        self._year = year
        self._text = text

    def removeStopWords(self):
        document = self.nlp(self._text)
        filteredWords = [token.text for token in document if not token.is_stop]
        self._text = ' '.join(filteredWords)
        del document

    def lemmatize(self):
        document = self.nlp(self._text)
        self._text = " ".join([token.lemma_ for token in document])
        del document

    def removeNonAlphaNumerics(self):
        # For some reason the text has some newlines as literally the characters '\n'.
        # _removeNewLineLiterals() needs to happen before the below regex is run, so I am enforcing the dependency here.
        self._removeNewLineLiterals()
        # Remove non-alphanumeric characters except spaces and dashes.
        self._text = re.sub(r'[^A-Za-z0-9\s-]+', "", self._text)
        # We do not want dashes and many spaces removed, just substituted with a single space.
        self._subDashesAndManySpacesWithSpace()

    def _subDashesAndManySpacesWithSpace(self):
        # Substitute a space where dashes and sequential spaces, greater than or equal to two, exist.
        self._text = re.sub(r'(-|\s{2,})', " ", self._text)

    def _removeNewLineLiterals(self):
        # Remove all literal '\n'
        self._text = re.sub(r"(\\n)+", "", self._text)

    def appendToCSV(self, preprocessedCSVPath):
        with open(preprocessedCSVPath, 'a', encoding='utf-8') as fd:
            output = self._uid + "," + self._year + "," + self._text + "\n"
            fd.write(output)


