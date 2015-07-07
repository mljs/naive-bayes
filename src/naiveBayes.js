'use strict';

var Matrix = require('ml-matrix');

module.exports = NaiveBayes;

function NaiveBayes(reload, model) {
    if(reload) {
        this.trueProbability = model.trueProbability;
        this.falseProbability = model.falseProbability;
        this.p1v = model.p1v;
        this.p1d = model.p1d;
        this.p0v = model.p0v;
        this.p0d = model.p0d;
    }
}

NaiveBayes.prototype.train = function (trainingSet, trainingLabels) {
    if(Matrix.isMatrix(trainingSet)) {
        trainingSet = Matrix(trainingSet);
    }
    if(Matrix.isMatrix(trainingLabels)) {
        trainingLabels = Matrix.rowVector(trainingLabels);
    }

    var cases = trainingSet.rows;
    var features = trainingSet.columns;

    this.trueProbability = trainingLabels.sum() / cases;
    this.falseProbability = 1 - this.trueProbability;

    this.p1v = Matrix.zeros(1, features);
    this.p0v = Matrix.zeros(1, features);

    this.p1d = 2;
    this.p0d = 2;

    for (var i = 0; i < cases; i++) {
        if(y[i] === 1) {
            this.p1v.add(trainingSet.getRowVector(i));
            this.p1d++;
        } else {
            this.p0v.add(trainingSet.getRowVector(i));
            this.p0d++;
        }
    }

    this.p1v = this.p1v.divS(this.p1d).apply(applyLog);
    this.p0v = this.p0v.divS(this.p0d).apply(applyLog);
};

NaiveBayes.prototype.predictCase = function (currentCase) {
    var p1 = currentCase.clone().mul(this.p1v).sum() + Math.log(this.trueProbability);
    var p0 = currentCase.clone().mul(this.p0v).sum() + Math.log(this.falseProbability);

    return (p1 > p0) ? 1 : 0;
};

NaiveBayes.prototype.predict = function (dataset) {
    if(Matrix.isMatrix(dataset)) {
        dataset = Matrix(dataset);
    }

    var cases = dataset.rows;
    var predictions = new Array(cases);

    for(var i = 0 ; i < cases; i++) {
        predictions[i] = this.predictCase(dataset.getRowVector(i));
    }

    return predictions;
};

function applyLog(i, j) {
    this[i][j] = Math.log(this[i][j]);
    return this;
}