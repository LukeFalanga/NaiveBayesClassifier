from DataLoader import DataLoader
from NaiveBayes import NaiveBayes
from EvaluationMetrics import EvaluationMetrics



#For the project purposes, the following will be 'SMSSpamCollection.txt'
fileToRead = input(f"Please Enter The File to Train and Test: ")

#Data loading
loader = DataLoader(fileToRead)
loader.load_data()
lines_to_train, labels_to_train, lines_to_test, labels_to_test = loader.split_data()

num_test_samples = len(lines_to_test)
num_train_samples = len(lines_to_train)

#Make Naive-Bayes Predictions
predictor = NaiveBayes(lines_to_train, labels_to_train, lines_to_test, labels_to_test)
predictor.train()
predictions = predictor.prediction()

#Compute the Evaluation Metrics
evaluator = EvaluationMetrics(predictions, labels_to_test)
evaluator.compute_metrics()

#Prints the computations to the "result.log" file

with open('result.log', 'w') as file:
    file.write(f"Number of training Samples: {num_train_samples}\n")
    file.write(f"Number of testing Samples: {num_test_samples}\n")
    file.write(f"TP: {evaluator.TP}\n")
    file.write(f"FN: {evaluator.FN}\n")
    file.write(f"TN: {evaluator.TN}\n")
    file.write(f"FP: {evaluator.FP}\n")
    file.write(f"Accuracy: {evaluator.accuracy * 100:.2f}%\n")
    file.write(f"Precision: {evaluator.precision * 100:.2f}%\n")
    file.write(f"Recall: {evaluator.recall * 100:.2f}%\n")
    file.write(f"F1 Score: {evaluator.F1 * 100:.2f}%\n")
