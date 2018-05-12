# ml-naivebayes

[![NPM version][npm-image]][npm-url]
[![build status][travis-image]][travis-url]
[![Test coverage][codecov-image]][codecov-url]
[![npm download][download-image]][download-url]

Naive bayes classifiers.

## Installation

`npm install ml-naivebayes`

## [API Documentation](https://mljs.github.io/naive-bayes/)

## Usage

### [GaussianNB](./src/GaussianNB.js)

```js
// assuming that you created Xtrain, Xtest, Ytrain, Ytest
import { GaussianNB } from 'ml-naivebayes';

var model = new GaussianNB();
model.train(Xtrain, Ytrain);

var predictions = model.predict(Xtest);
```

### [MultinomialNB](./src/MultinomialNB.js)

```js
// assuming that you created Xtrain, Xtest, Ytrain, Ytest
import { MultinomialNB } from 'ml-naivebayes';

var model = new MultinomialNB();
model.train(Xtrain, Ytrain);

var predictions = model.predict(Xtest);
```

## Authors

* [Jefferson Hernandez](https://github.com/JeffersonH44)

## License

[MIT](./LICENSE)

[npm-image]: https://img.shields.io/npm/v/ml-naivebayes.svg?style=flat-square
[npm-url]: https://npmjs.org/package/ml-naivebayes
[travis-image]: https://img.shields.io/travis/mljs/naive-bayes/master.svg?style=flat-square
[travis-url]: https://travis-ci.org/mljs/naive-bayes
[codecov-image]: https://img.shields.io/codecov/c/github/mljs/naive-bayes.svg?style=flat-square
[codecov-url]: https://codecov.io/github/mljs/naive-bayes
[download-image]: https://img.shields.io/npm/dm/ml-naivebayes.svg?style=flat-square
[download-url]: https://npmjs.org/package/ml-naivebayes
