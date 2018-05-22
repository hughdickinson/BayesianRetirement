import itertools

import numpy as np

from Classifiers import Classifier


class AnnotationBase():
    pass


class AnnotationBinary(AnnotationBase):
    def __init__(self,
                 classifier=None,
                 zooniverseAnnotations=None,
                 taskName=None,
                 trueValue=None,
                 falseValue=None):
        if not isinstance(classifier, Classifier):
            raise TypeError(
                'The classifier argument must be of type {}. Type {} passed.'.
                format(type(Classifier), type(classifier)))
        self._classifier = classifier
        self._zooniverseAnnotations = zooniverseAnnotations
        self._label = self.extractLabel(self.zooniverseAnnotations)
        self._taskName = taskName
        self._trueValue = trueValue
        self._falseValue = falseValue

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def trueValue(self):
        return self._trueValue

    @trueValue.setter
    def label(self, trueValue):
        self._trueValue = trueValue

    @property
    def falseValue(self):
        return self._falseValue

    @falseValue.setter
    def label(self, falseValue):
        self._falseValue = falseValue

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        if not isinstance(classifier, Classifier):
            raise TypeError(
                'The classifier argument must be of type {}. Type {} passed.'.
                format(type(Classifier), type(classifier)))
        self._classifier = classifier

    @property
    def taskName(self):
        return self._taskName

    @taskName.setter
    def taskName(self, taskName):
        self._taskName = taskName

    @property
    def zooniverseAnnotations(self):
        return self._zooniverseAnnotations

    @zooniverseAnnotations.setter
    def zooniverseAnnotations(self, zooniverseAnnotations):
        self._zooniverseAnnotations = zooniverseAnnotations

    def extractLabel(self):
        if self.taskName in self.zooniverseAnnotations:
            annotationValue = self.zooniverseAnnotations[self.taskName][
                'value']
            if self.trueValue is not None and annotationValue == self.trueValue:
                return True
            if self.trueValue is not None and annotationValue == self.falseValue:
                return False
        return None


class AnnotationKeyPoint(AnnotationBase):
    def __init__(self, x=None, y=None, label=None, classifier=None):
        self._x = x
        self._y = y
        self._label = label
        self._classifier = classifier

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, x):
        self._x = x

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, y):
        self._y = y

    @property
    def label(self):
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def classifier(self):
        return self._classifier

    @classifier.setter
    def classifier(self, classifier):
        self._classifier = classifier


class Annotations():
    def __init__(self, annotations):
        self._annotations = [
            annotation for annotation in annotations
            if issubclass(annotation, AnnotationBase)
        ]

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        self._annotations = [
            annotation for annotation in annotations
            if issubclass(annotation, AnnotationBase)
        ]

    def items(self):
        for annotation in self._annotations:
            yield annotation

    def getUniqueLabels(self):
        return np.unique(
            [annotation.label for annotation in self._annotations])


class AnnotationPriorBase():
    """Base class for prior distributions over possible annotation labels.
    """
    pass


class AnnotationModelBase():
    """Base class for annotation label probability models e.g. Bernoulli model for
    binary choice tasks.
    """
    pass
