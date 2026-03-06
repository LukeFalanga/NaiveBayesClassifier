

class EvaluationMetrics:

    def __init__(self, predictions, test_labels):
        self.predictions = predictions
        self.test_labels = test_labels
        self.TP = 0
        self.FN = 0
        self.TN = 0
        self.FP = 0
        self.accuracy = 0
        self.precision = 0
        self.recall = 0
        self.F1 = 0

    #compute_metrics member function compares the array of predictions with the array of actual test labels, and classifies each case before evaluating metrics

    def compute_metrics(self):
        
        for i in range(len(self.predictions)):
            if self.predictions[i] == 1 and self.test_labels[i] == 1:
                self.TP += 1
            elif self.predictions[i] == 1 and self.test_labels[i] == 0:
                self.FP += 1
            elif self.predictions[i] == 0 and  self.test_labels[i] == 1:
                self.FN += 1
            elif self.predictions[i] == 0 and self.test_labels[i] == 0:
                self.TN += 1
        
        self.accuracy = ((self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN))  if (self.TP + self.TN + self.FP + self.FN) > 0 else 0
        self.precision = (self.TP / (self.TP + self.FP))  if (self.TP + self.FP) else 0
        self.recall = (self.TP / (self.TP + self.FN))  if (self.TP + self.FN) else 0
        self.F1 = 2 * ((self.precision * self.recall) / (self.precision + self.recall)) if (self.precision + self.recall) > 0 else 0

       
