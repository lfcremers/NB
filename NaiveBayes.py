import sys
import getopt
import os
import math
import operator

class NaiveBayes:
    class TrainSplit:
        """Represents a set of training/testing data. self.train is a list of Examples, as is self.test.
        """
        def __init__(self):
          self.train = []
          self.test = []

    class Example:
        """Represents a document with a label. klass is 'pos' or 'neg' by convention.
           words is a list of strings.
        """
        def __init__(self):
            self.klass = ''
            self.words = []


    def __init__(self):
        """NaiveBayes initialization"""
        self.FILTER_STOP_WORDS = False
        self.BOOLEAN_NB = False
        self.BEST_MODEL = False
        self.stopList = set(self.readFile('data/english.stop'))
        self.numFolds = 10

    #############################################################################
    # TODO TODO TODO TODO TODO
    # Implement the Multinomial Naive Bayes classifier and the Naive Bayes Classifier with
    # Boolean (Binarized) features.
    # If the BOOLEAN_NB flag is true, your methods must implement Boolean (Binarized)
    # Naive Bayes (that relies on feature presence/absence) instead of the usual algorithm
    # that relies on feature counts.
    #
    # If the BEST_MODEL flag is true, include your new features and/or heuristics that
    # you believe would be best performing on train and test sets.
    #
    # If any one of the FILTER_STOP_WORDS, BOOLEAN_NB and BEST_MODEL flags is on, the
    # other two are meant to be off. That said, if you want to include stopword removal
    # or binarization in your best model, write the code accordingl

    def classify(self, words):
        """ TODO
            'words' is a list of words to classify. Return 'pos' or 'neg' classification.

        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)

        for word in words:
            print(word)

        
        return 'pos'




    def addExample(self, klass, words):
        """
        compute the score, and then select the threshold score based on the training set; find a threshold that gives you the highest accuracy
        need to compute a score for each training example; the most natural way is to hold examples out of the training sample, choose threshold

        you don't need threshold; you get a uniqe probabilyt for negative and positive, and you pick the one that's greater
        you have to implement laplace smoothing in case its not seen; in the directions it tells you every freq you need to calculate; 
        if you caltulate each one then you get frequency of each word with each class in the positive and negative documents

        accuracy threshold for the binary Naive Bayes
        implemented in the same way as pseudocode from book with add 1 smoothing; it made 80.8 accuracy and 80.6 with -f flags
        other TAs thought it was good but binary bayes accuracy was lower
        does that mean implementation was wrong; why would it be?
        binary classifuing whether it was there outperformed the other one
        when she did binary she got binary of .74; 
        thumbs up sentiment classification using ML techniques; look at this document for ideas for improvement; the last part

         * TODO
         * Train your model on an example document with label klass ('pos' or 'neg') and
         * words, a list of strings.
         * You should store whatever data structures you use for your classifier
         * in the NaiveBayes class.
         * Returns nothing
        """
        if self.FILTER_STOP_WORDS:
            words = self.filterStopWords(words)

        
        if 'counts' not in self.__dict__:
            self.counts={}
        if 'V' not in self.__dict__:
            self.V=0

        for word in words:
            if word not in self.counts:
                self.counts[word]={'pos':0,'neg':0}
                self.V+=1

            self.counts[word][klass]+=1

        if 'docnum' not in self.__dict__:
            self.docnum=0
        self.docnum+=1
        if 'posnum' not in self.__dict__:
            self.posnum=0
            self.negnum=0
            
        if klass=="neg":
            self.negnum+=1
        else:
            self.posnum+=1
        
        

      

    # END TODO (Modify code beyond here with caution)
    #############################################################################


    def readFile(self, fileName):
        """
         * Code for reading a file.  you probably don't want to modify anything here,
         * unless you don't like the way we segment files.
        """
        contents = []
        f = open(fileName)
        for line in f:
            contents.append(line)
        f.close()
        result = self.segmentWords('\n'.join(contents))
        return result

  
    def segmentWords(self, s):
        """
         * Splits lines on whitespace for file reading
        """
        return s.split()

  
    def trainSplit(self, trainDir):
        """Takes in a trainDir, returns one TrainSplit with train set."""
        split = self.TrainSplit()
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        for fileName in posTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
            example.klass = 'pos'
            split.train.append(example)
        for fileName in negTrainFileNames:
            example = self.Example()
            example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
            example.klass = 'neg'
            split.train.append(example)
        return split

    def train(self, split):
        for example in split.train:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            self.addExample(example.klass, words)


    def crossValidationSplits(self, trainDir):
        """Returns a lsit of TrainSplits corresponding to the cross validation splits."""
        splits = []
        posTrainFileNames = os.listdir('%s/pos/' % trainDir)
        negTrainFileNames = os.listdir('%s/neg/' % trainDir)
        #for fileName in trainFileNames:
        for fold in range(0, self.numFolds):
            split = self.TrainSplit()
            for fileName in posTrainFileNames:
              example = self.Example()
              example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
              example.klass = 'pos'
              if fileName[2] == str(fold):
                  split.test.append(example)
              else:
                  split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                if fileName[2] == str(fold):
                    split.test.append(example)
                else:
                    split.train.append(example)
            yield split


    def test(self, split):
        """Returns a list of labels for split.test."""
        labels = []
        for example in split.test:
            words = example.words
            if self.FILTER_STOP_WORDS:
                words =  self.filterStopWords(words)
            guess = self.classify(words)
            labels.append(guess)
        return labels
  
    def buildSplits(self, args):
        """Builds the splits for training/testing"""
        trainData = []
        testData = []
        splits = []
        trainDir = args[0]
        if len(args) == 1:
            print('[INFO]\tPerforming %d-fold cross-validation on data set:\t%s' % (self.numFolds, trainDir))

            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fold in range(0, self.numFolds):
                split = self.TrainSplit()
                for fileName in posTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                    example.klass = 'pos'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                for fileName in negTrainFileNames:
                    example = self.Example()
                    example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                    example.klass = 'neg'
                    if fileName[2] == str(fold):
                        split.test.append(example)
                    else:
                        split.train.append(example)
                splits.append(split)
        elif len(args) == 2:
            split = self.TrainSplit()
            testDir = args[1]
            print('[INFO]\tTraining on data set:\t%s testing on data set:\t%s' % (trainDir, testDir))
            posTrainFileNames = os.listdir('%s/pos/' % trainDir)
            negTrainFileNames = os.listdir('%s/neg/' % trainDir)
            for fileName in posTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (trainDir, fileName))
                example.klass = 'pos'
                split.train.append(example)
            for fileName in negTrainFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (trainDir, fileName))
                example.klass = 'neg'
                split.train.append(example)

            posTestFileNames = os.listdir('%s/pos/' % testDir)
            negTestFileNames = os.listdir('%s/neg/' % testDir)
            for fileName in posTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/pos/%s' % (testDir, fileName))
                example.klass = 'pos'
                split.test.append(example)
            for fileName in negTestFileNames:
                example = self.Example()
                example.words = self.readFile('%s/neg/%s' % (testDir, fileName))
                example.klass = 'neg'
                split.test.append(example)
            splits.append(split)
        return splits
  
    def filterStopWords(self, words):
        """Filters stop words."""
        filtered = []
        for word in words:
            if not word in self.stopList and word.strip() != '':
                filtered.append(word)
        return filtered

def test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL):
    nb = NaiveBayes()
    splits = nb.buildSplits(args)
    avgAccuracy = 0.0
    fold = 0
    classifier = None
    for split in splits:
        classifier = NaiveBayes()
        classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
        classifier.BOOLEAN_NB = BOOLEAN_NB
        classifier.BEST_MODEL = BEST_MODEL
        accuracy = 0.0
        for example in split.train:
            words = example.words
            classifier.addExample(example.klass, words)

        for example in split.test:
            words = example.words
            guess = classifier.classify(words)
            if example.klass == guess:
                accuracy += 1.0
        accuracy = accuracy / len(split.test)
        avgAccuracy += accuracy
        print('[INFO]\tFold %d Accuracy: %f' % (fold, accuracy))
        fold += 1
    avgAccuracy = avgAccuracy / fold
    print('[INFO]\tAccuracy: %f' % avgAccuracy)

    # interpret the decision rule of the model of the last fold
    pos_signal_words, neg_signal_words = analyze_model(classifier)
    print('[INFO]\tWords for pos class: %s' % ','.join(pos_signal_words))
    print('[INFO]\tWords for neg class: %s' % ','.join(neg_signal_words))

    
def classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, trainDir, testFilePath):
    classifier = NaiveBayes()
    classifier.FILTER_STOP_WORDS = FILTER_STOP_WORDS
    classifier.BOOLEAN_NB = BOOLEAN_NB
    classifier.BEST_MODEL = BEST_MODEL
    trainSplit = classifier.trainSplit(trainDir)
    classifier.train(trainSplit)
    testFile = classifier.readFile(testFilePath)
    print(classifier.classify(testFile))
    
def main():
    FILTER_STOP_WORDS = False
    BOOLEAN_NB = False
    BEST_MODEL = False
    (options, args) = getopt.getopt(sys.argv[1:], 'fbm')
    if ('-f','') in options:
        FILTER_STOP_WORDS = True
    elif ('-b','') in options:
        BOOLEAN_NB = True
    elif ('-m','') in options:
        BEST_MODEL = True

    if len(args) == 2 and os.path.isfile(args[1]):
        classifyFile(FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL, args[0], args[1])
    else:
        test10Fold(args, FILTER_STOP_WORDS, BOOLEAN_NB, BEST_MODEL)


def analyze_model(nb_classifier):
    # TODO: This function takes a <nb_classifier> as input, and outputs two word list <pos_signal_words> and
    #  <neg_signal_words>. <pos_signal_words> is a list of 10 words signaling the positive klass, and <neg_signal_words>
    #  is a list of 10 words signaling the negative klass.
    pass


if __name__ == "__main__":
    main()
