import math


class NaiveBayes:

    def __init__(self, train_lines, train_labels, test_lines, test_labels):
        self.train_lines = train_lines
        self.train_labels = train_labels
        self.test_lines = test_lines
        self.test_labels = test_labels
        self.alpha = 1
        self.pro_ham = 0
        self.pro_spam = 0
        self.ham_pro_table = {}
        self.spam_pro_table = {}
        self.total_ham_words = 0
        self.total_spam_words = 0
        self.V = 0

    
    #train Member function trains the data, building the word probability tables for both ham adn spam

    def train(self):
        ham_word_counts = {}
        spam_word_counts = {}
        spam_lines = 0
        ham_lines = 0


        # Counts the total number of words in both ham and spam, the number of lines that are ham and spam, as well as the word counts for each individual word in spam or ham

        for line, label in zip(self.train_lines, self.train_labels):
            if label == 0:
                ham_lines += 1
                for word in line:
                    self.total_ham_words += 1
                    ham_word_counts[word] = ham_word_counts.get(word, 0) + 1
            else:
                spam_lines += 1
                for word in line:
                    self.total_spam_words += 1
                    spam_word_counts[word] = spam_word_counts.get(word, 0) + 1          


        #Builds the probability that any given line is ham or a line is spam
        self.pro_ham = ham_lines / len(self.train_lines)
        self.pro_spam = spam_lines / len(self.train_lines)

        #Creates a variable that stores all unique words between ham and spam
        unique_words = ham_word_counts.keys() | spam_word_counts.keys()
        self.V = len(unique_words)

        #Builds probability chart, applying laplace smoothing for unique words
        for word in unique_words:
            self.ham_pro_table[word] = (ham_word_counts.get(word, 0) + self.alpha) / (self.total_ham_words + self.alpha * self.V)
            self.spam_pro_table[word] = (spam_word_counts.get(word, 0) + self.alpha) / (self.total_spam_words + self.alpha * self.V)



    #prediction Member function iterates through the lines to test, and predicts whether the line is ham or spam
    #The function returns a new array containing the prediction for each tested line

    def prediction(self):
        predictions = []

        for line in self.test_lines:
            ham_prediction = math.log(self.pro_ham)
            spam_prediction = math.log(self.pro_spam)
            for word in line:
                    
                    #Note: Laplace smoothing is applied again for words that have not yet been seen by the model
                    ham_prediction += math.log(self.ham_pro_table.get(word, (self.alpha) / (self.total_ham_words + self.alpha * self.V)))
                    spam_prediction += math.log(self.spam_pro_table.get(word, (self.alpha) / (self.total_spam_words + self.alpha * self.V)))
                    
            if ham_prediction >= spam_prediction:
                predictions.append(0)
            else:
                predictions.append(1)
        
        return predictions

