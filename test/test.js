'use strict';

var NaiveBayes = require('..');
var separateClasses = require('..').separateClasses;
var Matrix = require('ml-matrix');

describe('Naive bayes', function () {

    var cases = [[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]];
    var predictions = [0, 0, 0, 1, 1];

    it('Basic test', function () {
        var nb = new NaiveBayes();
        nb.train(cases, predictions);
        var results = nb.predict(cases);

        results[0].should.be.equal(0);
        results[1].should.be.equal(0);
        results[2].should.be.equal(0);
        results[3].should.be.equal(1);
        results[4].should.be.equal(1);
    });

    it('separate classes', function () {
        var matrixCases = new Matrix(cases);
        var separatedResult = separateClasses(matrixCases, predictions);
        separatedResult.length.should.be.equal(2);
        separatedResult[0].rows.should.be.equal(3);
        separatedResult[1].rows.should.be.equal(2);
    });

    it('Small test', function () {
        var cases = [[6,148,72,35,0,33.6,0.627,5],
                     [1.50,85,66.5,29,0,26.6,0.351,31],
                     [8,183,64,0,0,23.3,0.672,32],
                     [0.5,89,65.5,23,94,28.1,0.167,21],
                     [0,137,40,35,168,43.1,2.288,33]];
        var predictions = [1, 0, 1, 0, 1];
        var nb = new NaiveBayes();
        nb.train(cases, predictions);
        var result = nb.predict(cases);

        result[0].should.be.equal(1);
        result[1].should.be.equal(0);
        result[2].should.be.equal(1);
        result[3].should.be.equal(0);
        result[4].should.be.equal(1);
    });

    it('two feature test', function () {
        var cases = [
                      [0, 0],
                      [0.5, 0.5],
                      [1, 0],
                      [0, 1],
                      [1, 1]
                    ];
        var predictions = [0, 0, 1, 1, 1];
        var nb = new NaiveBayes();
        nb.train(cases, predictions);
        var result = nb.predict(cases);

        result[0].should.be.equal(0);
        result[1].should.be.equal(0);
        result[2].should.be.equal(1);
        result[3].should.be.equal(1);
        result[4].should.be.equal(1);
    });

    it('Third feature test', function () {
        var cases = [
                      [0, 0, 0],
                      [0, 0, 1],
                      [0.5, 0.5, 0],
                      [0.5, 0.5, 0.5],
                      [1, 0, 0],
                      [0, 1, 1],
                      [1, 1, 0.5]
                    ];
        var predictions = [0, 1, 0, 0, 1, 1, 1];
        var nb = new NaiveBayes();
        nb.train(cases, predictions);
        var result = nb.predict(cases);

        result[0].should.be.equal(0);
        result[1].should.be.equal(1);
        result[2].should.be.equal(0);
        result[3].should.be.equal(0);
        result[4].should.be.equal(1);
        result[5].should.be.equal(1);
        result[6].should.be.equal(1);
    });

    it('Export and import', function () {
        var cases = [[6,148,72,35,0,33.6,0.627,5],
            [1.50,85,66.5,29,0,26.6,0.351,31],
            [8,183,64,0,0,23.3,0.672,32],
            [0.5,89,65.5,23,94,28.1,0.167,21],
            [0,137,40,35,168,43.1,2.288,33]];
        var predictions = [1, 0, 1, 0, 1];
        var nb = new NaiveBayes();
        nb.train(cases, predictions);

        var model = nb.export();
        nb = NaiveBayes.load(model);

        var result = nb.predict(cases);

        result[0].should.be.equal(1);
        result[1].should.be.equal(0);
        result[2].should.be.equal(1);
        result[3].should.be.equal(0);
        result[4].should.be.equal(1);
    });
});
