'use strict';

var NaiveBayes = require('ml-naive-bayes');
var nb = new NaiveBayes();

// training data
var cases = [[6,148,72,35,0,33.6,0.627,5],
             [1.50,85,66.5,29,0,26.6,0.351,31],
             [8,183,64,0,0,23.3,0.672,32],
             [0.5,89,65.5,23,94,28.1,0.167,21],
             [0,137,40,35,168,43.1,2.288,33]];
var predictions = [1, 0, 1, 0, 1];
nb.train(trainingSet, predictions);

// predicting case
var dataset = [[6,148,72,35,0,33.6,0.627,5],
               [1.50,85,66.5,29,0,26.6,0.351,31]];
var ans = nb.predict(dataset);
