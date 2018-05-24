import itertools

import numpy as np


class AnnotationModelBase():

    validLabels = []
    """Subclasses compute the probability of classifiers assigning a particular label
    given a particular true label.
    """

    def __init__(self, **args):
        pass


class AnnotationModelBinary(AnnotationModelBase):
    """Implements a Bernoulli model of a classifier assigning
    annotationLabel when trueLabel is true.
    """

    validLabels = [True, False]

    def __init__(self, **args):
        super().__init__(**args)

    def __call__(self, trueLabel, annotation):
        """Computes the probability of a classifier assigning annotationLabel when
        trueLabel is true.

        Depends upon the precomputation of the classifier's skill. Recall that
        the skill is defined as the probability of assigning any valid label given
        a specific true label.
        """
        print(annotation.classifier.skills)
        return annotation.classifier.skills[trueLabel] if (
            trueLabel == annotation.label
        ) else 1.0 - annotation.classifier.skills[trueLabel]


class AnnotationPriorBase():

    validLabels = AnnotationModelBase.validLabels
    """Subclasses compute the prior probability of a subject having a particular
    true label.
    """

    def __init__(self, **args):
        self._priorsForLabels = None


class AnnotationPriorBinary(AnnotationPriorBase):
    """Implements a Beta distribution prior for the annotation provided
    by a single classifier.
    """

    validLabels = AnnotationModelBinary.validLabels

    def __init__(self, **args):
        super().__init__(**args)

        self._successProb = args.get('successProb', 0.5)
        self._failureProb = 1.0 - self.successProb

    @property
    def successProb(self):
        return self._successProb

    @successProb.setter
    def successProb(self, successProb):
        """Setter maintains unit sum of outcome probabilities.
        """
        self.successProb = successProb
        self.failureProb = 1.0 - self.successProb

    @property
    def failureProb(self):
        return self._failureProb

    @successProb.setter
    def successProb(self, failureProb):
        """Setter maintains unit sum of outcome probabilities.
        """
        self.failureProb = failureProb
        self.successProb = 1.0 - self.failureProb

    def __call__(self, trueLabel):
        """Return probability obtaining trueLabel.
        """
        if trueLabel in AnnotationPriorBinary.validLabels:
            return self.successProb if trueLabel else self.failureProb
        else:
            raise ValueError('The trueLabel argument must be in of {}.'.format(
                AnnotationPriorBinary.validLabels))
