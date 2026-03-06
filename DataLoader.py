from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import re
import random
import nltk


#DataLoader Class is used to handle data preprocessing, loading, and splitting
class DataLoader:

    def __init__(self, file_path, seed = 10, test_ratio = 0.2):
        self.file_path = file_path
        self.seed = seed
        self.test_ratio = test_ratio
        self.lines = []
        self.labels = []

    #preprocess Member function converts text to lowercase, removes special characters, tokenizes, and applies stemming

    def preprocess(self, text):

        #Convert to lowercase, remove special characters, and tokenize

        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        words = word_tokenize(text)

        #Initializes the stemmer

        stemmer = PorterStemmer()

        #Creating new list for the processed words to be stored into

        new_words = []

        #Goes through each word in the list of tokenized words and applies the stemmer to them, before adding them to the new_words array

        for word in words:
            stemmed = stemmer.stem(word)
            new_words.append(stemmed)

        return new_words
    
    #load_data Member function will load and preprocess data from a file and return text and labels in two separate arrays

    def load_data(self):

        lines, labels = [], []

        #Opening and reading the file

        with open(self.file_path, 'r', encoding='utf-8')as file:
            for line in file:
                
                #Splits the line into label and text

                label, text = line.strip().split(maxsplit = 1)

                #Appends the preprocessed text into the 'lines' array

                lines.append(self.preprocess(text))

                #Appends 1 to 'labels' if the label = 'spam', 0 if it does not (ham) 

                labels.append(1 if label.lower()=='spam' else 0)

        self.lines = lines
        self.labels = labels


    
    #split_data Member function splits the data into training and testing sets manually

    def split_data(self):
        #Zips the lines and labels so that the length of the lists can be made into a variable
        data = list(zip(self.lines, self.labels))
        size = len(data)

        random.seed(self.seed)
        random.shuffle(data)
        
        #Unzips the lines and labels, and turns them back into lists
        lines, labels = zip(*data)
        lines = list(lines)
        labels = list(labels)
        
        #Instantiates the data splitter 
        data_splitter = int(((1-self.test_ratio) * size))

        #Uses the data splitter to set the lines and labels into training and testing arrays
        line_training = lines[:data_splitter]
        label_training = labels[:data_splitter]
        line_testing = lines[data_splitter:]
        label_testing = labels[data_splitter:]

        return line_training, label_training, line_testing, label_testing






        




        




