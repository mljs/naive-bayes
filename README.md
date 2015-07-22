# Naive Bayes
Naive bayes classifier.

## Methods

### new NaiveBayes()

Constructor that takes no arguments.

__Example__

```js
var nb = new NaiveBayes();
```

### train(trainingSet, predictions)

Train the Naive Bayes model to the given training set and predictions

__Arguments__

* `trainingSet` - A matrix of the training set.
* `trainingLabels` - An array of value for each case in the training set.

__Example__

```js
var cases = [[6,148,72,35,0,33.6,0.627,5],
             [1.50,85,66.5,29,0,26.6,0.351,31],
             [8,183,64,0,0,23.3,0.672,32],
             [0.5,89,65.5,23,94,28.1,0.167,21],
             [0,137,40,35,168,43.1,2.288,33]];
var predictions = [1, 0, 1, 0, 1];

nb.train(trainingSet, predictions);
```

### predict(dataset)

Predict the values of the dataset.

__Arguments__

* `dataset` - A matrix that contains the dataset.

__Example__

```js
var dataset = [[6,148,72,35,0,33.6,0.627,5],
               [1.50,85,66.5,29,0,26.6,0.351,31]];

var ans = nb.predict(dataset);
```

### export()

Exports the actual Naive Bayes model to an Javascript Object.

### load(model)

Returns a new Naive Bayes Classifier with the given model.

__Arguments__

* `model` - Javascript Object generated from export() function.

## Authors

- [Jefferson Hernandez](https://github.com/JeffersonH44)

## Licence

[MIT](./LICENSE)