Implementing the Naive Bayes Algorithm
======================================
Naive Bayes is a family of statistical classifiers based on the `Bayes' Theorem`_. These classifiers share a common principle: each feature is strongly independent of others, regardless of any possible correlations between them.


Classifier Building in Scikit-learn
-----------------------------------

Naive Bayes learners and classifiers can be extremely fast compared to more sophisticated methods. The decoupling of the class conditional feature distributions means that each distribution can be independently estimated as a one dimensional distribution. This in turn helps to alleviate problems stemming from the curse of dimensionality. On the flip side, although naive Bayes is known as a decent classifier, it is known to be a bad estimator, so the probability outputs from predict_proba are not to be taken too seriously.

I will be using GaussianNB, which implements the Gaussian Naive Bayes algorithm for classification.

.. _Bayes' Theorem: https://en.wikipedia.org/wiki/Bayes%27_theorem
