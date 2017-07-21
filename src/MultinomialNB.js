"use strict";

var Matrix = require('ml-matrix').Matrix;
var Utils = require('./utils');

class MultinomialNB {
    constructor(model) {
    }

    train(trainingSet, trainingLabels) {
        trainingSet = Matrix.checkMatrix(trainingSet);
        var separateClasses = Utils.separateClasses(trainingSet, trainingLabels);
        this.priorProbability = new Matrix(separateClasses.length, 1);

        for(var i = 0; i < separateClasses.length; ++i) {
            this.priorProbability[i][0] = Math.log(separateClasses[i].length / trainingSet.rows);
        }

        var features = trainingSet.columns;
        this.conditionalProbability = new Matrix(separateClasses.length, features);
        for(i = 0; i < separateClasses.length; ++i) {
            var classValues = Matrix.checkMatrix(separateClasses[i]);
            var total = classValues.sum();
            var divisor = total + features;
            this.conditionalProbability.setRow(i, classValues.sum('column').add(1).div(divisor).apply(matrixLog));
        }
    }

    predict(dataset) {
        dataset = Matrix.checkMatrix(dataset);
        var predictions = new Array(dataset.rows);
        for(var i = 0; i < dataset.rows; ++i) {
            var currentElement = dataset.getRowVector(i);
            predictions[i] = this.conditionalProbability.clone().mulRowVector(currentElement).sum('row')
                             .add(this.priorProbability).maxIndex()[0];
        }
    }

    toJSON() {

    }

    load() {

    }
}

function matrixLog(i, j) {
    this[i][j] = Math.log(this[i][j]);
}

module.exports = MultinomialNB;
