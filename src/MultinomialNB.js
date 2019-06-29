import { Matrix } from 'ml-matrix';

import { separateClasses } from './utils';

export class MultinomialNB {
  /**
   * Constructor for Multinomial Naive Bayes, the model parameter is for load purposes.
   * @constructor
   * @param {object} model - for load purposes.
   */
  constructor(model) {
    if (model) {
      this.conditionalProbability = Matrix.checkMatrix(
        model.conditionalProbability
      );
      this.priorProbability = Matrix.checkMatrix(model.priorProbability);
    }
  }

  /**
   * Train the classifier with the current training set and labels, the labels must be numbers between 0 and n.
   * @param {Matrix|Array} trainingSet
   * @param {Array} trainingLabels
   */
  train(trainingSet, trainingLabels) {
    trainingSet = Matrix.checkMatrix(trainingSet);

    if (trainingSet.rows !== trainingLabels.length) {
      throw new RangeError(
        'the size of the training set and the training labels must be the same.'
      );
    }

    var separateClass = separateClasses(trainingSet, trainingLabels);

    this.priorProbability = new Matrix(separateClass.length, 1);

    for (var i = 0; i < separateClass.length; ++i) {
      this.priorProbability.set(i, 0, Math.log(
        separateClass[i].rows / trainingSet.rows
      ));
    }

    var features = trainingSet.columns;
    this.conditionalProbability = new Matrix(separateClass.length, features);
    for (i = 0; i < separateClass.length; ++i) {
      var classValues = Matrix.checkMatrix(separateClass[i]);
      var total = classValues.sum();
      var divisor = total + features;
      this.conditionalProbability.setRow(
        i,
        Matrix.rowVector(classValues
          .sum('column'))
          .add(1)
          .div(divisor)
          .apply(matrixLog)
      );
    }
  }

  /**
   * Retrieves the predictions for the dataset with the current model.
   * @param {Matrix|Array} dataset
   * @return {Array} - predictions from the dataset.
   */
  predict(dataset) {
    dataset = Matrix.checkMatrix(dataset);
    var predictions = new Array(dataset.rows);
    for (var i = 0; i < dataset.rows; ++i) {
      var currentElement = dataset.getRowVector(i);
      const v = Matrix.columnVector(this.conditionalProbability
        .clone()
        .mulRowVector(currentElement)
        .sum('row'));
      predictions[i] = v
        .add(this.priorProbability)
        .maxIndex()[0];
    }

    return predictions;
  }

  /**
   * Function that saves the current model.
   * @return {object} - model in JSON format.
   */
  toJSON() {
    return {
      name: 'MultinomialNB',
      priorProbability: this.priorProbability,
      conditionalProbability: this.conditionalProbability
    };
  }

  /**
   * Creates a new MultinomialNB from the given model
   * @param {object} model
   * @return {MultinomialNB}
   */
  static load(model) {
    if (model.name !== 'MultinomialNB') {
      throw new RangeError(`${model.name} is not a Multinomial Naive Bayes`);
    }

    return new MultinomialNB(model);
  }
}

function matrixLog(i, j) {
  this.set(i, j, Math.log(this.get(i, j)));
}
