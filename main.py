import os
import csv
import nltk
from Corpus import Corpus
from ResearchPaper import ResearchPaper


def main():
    nltk.download('stopwords')
    inputCSVFile = r"inputCSV/SMdata.csv"
    preprocessedCSVFile = r"preprocessedCSV/corpusCSV.csv"

    # Delete below
    rowCount = 0

    preprocess = True
    if preprocess:
        os.remove(preprocessedCSVFile)
        print('Preprocessing data...')
        with open(inputCSVFile, newline='', encoding="utf-8") as csvfile:
            paperReader = csv.reader(csvfile, delimiter=',')
            for row in paperReader:
                researchPaper = ResearchPaper(uid=row[0], year=row[1], text=row[3])
                researchPaper.removeNonAlphaNumerics()
                researchPaper.removeStopWords()
                researchPaper.lemmatize()
                researchPaper.appendToCSV(preprocessedCSVFile)

                # Delete below
                rowCount += 1
                if rowCount == 50:
                    break
    else:
        print("Skipping data preprocessing...")

    corpus = Corpus(preprocessedCSVFile)
    # corpus.runLDA(num_topics=15, random_state=100, update_every=1, chunksize=2000, passes=10, alpha="auto")
    corpus.runTLDA(num_topics=10, random_state=100, chunksize=9, passes=5, alphas=0.01, time_slice=[5, 4])


if __name__ == '__main__':
    main()

