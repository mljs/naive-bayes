'use strict';

var Matrix = require('ml-matrix');
var Stat = require('ml-stat');

module.exports.NaiveBayes = NaiveBayes;
module.exports.separateClasses = separateClasses;

/**
 * Constructor for the Naive Bayes classifier, the parameters here is just for loading purposes.
 *
 * @param reload
 * @param model
 * @constructor
 */
function NaiveBayes(reload, model) {
    if(reload) {
        this.means = model.means;
        this.calculateProbabilities = model.calculateProbabilities;
    }
}

/**
 * Function that trains the classifier with a matrix that represents the training set and an array that
 * represents the label of each row in the training set. the labels must be numbers between 0 to n-1 where
 * n represents the number of classes.
 *
 * WARNING: in the case that one class, all the cases in one or more features have the same value, the
 * Naive Bayes classifier will not work well.
 * @param trainingSet
 * @param trainingLabels
 */
NaiveBayes.prototype.train = function (trainingSet, trainingLabels) {
    var C1 = Math.sqrt(2*Math.PI); // constant to precalculate the squared root
    if(!Matrix.isMatrix(trainingSet)) trainingSet = new Matrix(trainingSet);
    else trainingSet = trainingSet.clone();

    if(trainingSet.rows !== trainingLabels.length)
        throw new RangeError("the size of the training set and the training labels must be the same.");

    var separatedClasses = separateClasses(trainingSet, trainingLabels);
    var calculateProbabilities = new Array(separatedClasses.length);
    this.means = new Array(separatedClasses.length);
    for(var i = 0; i < separatedClasses.length; ++i) {
        var means = Stat.matrix.mean(separatedClasses[i]);
        var std = Stat.matrix.standardDeviation(separatedClasses[i], means);

        var logPriorProbability = Math.log(separatedClasses[i].rows / trainingSet.rows);
        calculateProbabilities[i] = new Array(means.length + 1);

        calculateProbabilities[i][0] = logPriorProbability;
        for(var j = 1; j < means.length + 1; ++j) {
            var currentStd = std[j - 1];
            calculateProbabilities[i][j] = [(1 / (C1 * currentStd)), -2*currentStd*currentStd];
        }

        this.means[i] = means;
    }

    this.calculateProbabilities = calculateProbabilities;
};

/**
 * function that predicts each row of the dataset (must be a matrix).
 *
 * @param dataset
 * @returns {Array}
 */
NaiveBayes.prototype.predict = function (dataset) {
    if(dataset[0].length === this.calculateProbabilities[0].length)
        throw new RangeError('the dataset must have the same features as the training set');

    var predictions = new Array(dataset.length);

    for(var i = 0; i < predictions.length; ++i) {
        predictions[i] = getCurrentClass(dataset[i], this.means, this.calculateProbabilities);
    }

    return predictions;
};

/**
 * Function the retrieves a prediction with one case.
 *
 * @param currentCase
 * @param mean - Precalculated means of each class trained
 * @param classes - Precalculated value of each class (Prior probability and probability function of each feature)
 * @returns {number}
 */
function getCurrentClass(currentCase, mean, classes) {
    var maxProbability = 0;
    var predictedClass = -1;

    // going through all precalculated values for the classes
    for(var i = 0; i < classes.length; ++i) {
        var currentProbability = classes[i][0]; // initialize with the prior probability
        for(var j = 1; j < mean[0].length + 1; ++j) {
            currentProbability += calculateLogProbability(currentCase[j - 1], mean[i][j - 1], classes[i][j][0], classes[i][j][1]);
        }

        currentProbability = Math.exp(currentProbability);
        if(currentProbability > maxProbability) {
            maxProbability = currentProbability;
            predictedClass = i;
        }
    }

    return predictedClass;
}

/**
 * Function that export the NaiveBayes model.
 * @returns {{modelName: string, means: *, calculateProbabilities: *}}
 */
NaiveBayes.prototype.export = function () {
    return {
        modelName: "NaiveBayes",
        means: this.means,
        calculateProbabilities: this.calculateProbabilities
    };
};

/**
 * Function that create a Naive Bayes classifier with the given model.
 * @param model
 * @returns {NaiveBayes}
 */
NaiveBayes.load = function (model) {
    if(model.modelName !== 'NaiveBayes')
        throw new RangeError("The given model is invalid!");

    return new NaiveBayes(true, model);
};

/**
 * function that retrieves the probability of the feature given the class.
 * @param value - value of the feature.
 * @param mean - mean of the feature for the given class.
 * @param C1 - precalculated value of (1 / (sqrt(2*pi) * std)).
 * @param C2 - precalculated value of (2 * std^2) for the denominator of the exponential.
 * @returns {number}
 */
function calculateLogProbability(value, mean, C1, C2) {
    var value = value - mean;
    return Math.log(C1 * Math.exp((value * value) / C2))
}

/**
 * Function that retuns an array of matrices of the cases that belong to each class.
 * @param X - dataset
 * @param y - predictions
 * @returns {Array}
 */
function separateClasses(X, y) {
    var features = X.columns;

    var classes = 0;
    var totalPerClasses = new Array(100); // max upperbound of classes
    for (var i = 0; i < y.length; i++) {
        if(totalPerClasses[y[i]] === undefined) {
            totalPerClasses[y[i]] = 0;
            classes++;
        }
        totalPerClasses[y[i]]++;
    }
    var separatedClasses = new Array(classes);
    var currentIndex = new Array(classes);
    for(i = 0; i < classes; ++i) {
        separatedClasses[i] = new Matrix(totalPerClasses[i], features);
        currentIndex[i] = 0;
    }
    for(i = 0; i < X.rows; ++i) {
        separatedClasses[y[i]].setRow(currentIndex[y[i]], X.getRow(i));
        currentIndex[y[i]]++;
    }
    return separatedClasses;
}
