import numpy as np


class AnnotationBase():
    pass


class AnnotationBinary(AnnotationBase):
    def __init__(self, label=None, classifier=None):
        self._label = label
        self._classifier = classifier

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
        self._annotations = [annotation for annotation in annotations if issubclass(annotation, AnnotationBase)]

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        self._annotations = [annotation for annotation in annotations if issubclass(annotation, AnnotationBase)]

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
