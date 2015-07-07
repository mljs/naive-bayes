'use strict';

var NaiveBayes = require('../src/naiveBayes');

describe('Naive bayes', function () {

    var cases = [[0, 0], [0, 1], [1, 0], [1, 1]];
    var predictions = [0, 0, 0, 1];

    it('Main test', function () {
        var nb = NaiveBayes(false);
        console.log(nb);
        nb.train(cases, predictions);
        console.log(nb.predict(cases));
    });
});
