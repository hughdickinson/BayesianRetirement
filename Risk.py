# Subjects(Difficulty) -> Annotations -> Classifiers(Skill)
import scipy.stats as scistats
import numpy as np

from Annotations import Annotations
from AnnotationModels import AnnotationModelBase, AnnotationPriorBase
from Subjects import Subject, Subjects


class LossModelBase():
    """Base class. Subclasses evaluate the loss associated with predicting a label
    given a nominal true label. Note that the loss model should include the case when
    the predicted and true labels are equal.
    """

    def __init__(self):
        pass

    def __call__(self, trueLabel, predictedLabel):
        pass


class LossModelBinary(LossModelBase):
    def __init__(self, falsePosLoss=1, falseNegLoss=1):
        self._falsePosLoss = falsePosLoss
        self._falseNegLoss = falseNegLoss

    def __call__(self, trueLabel, predictedLabel):
        if trueLabel and not predictedLabel:
            # False Negative
            return self._falseNegLoss
        elif predictedLabel and not trueLabel:
            #False positive
            return self._falsePosLoss
        else:
            return 0

    @property
    def falsePosLoss(self):
        return self._falsePosLoss

    @falsePosLoss.setter
    def falsePosLoss(self, falsePosLoss):
        self._falsePosLoss = falsePosLoss

    @property
    def falseNegLoss(self):
        return self._falseNegLoss

    @falseNegLoss.setter
    def falseNegLoss(self, falseNegLoss):
        self._falseNegLoss = falseNegLoss


class Risk():
    """Computes a metric that can be used to define decision thresholds for each
    subject based on its classification history.
    """

    def __call__(self, annotations, subject, lossModel, annotationModel,
                 annotationPriorModel):
        """Evaluate the risk.
        Arguments:
        -- annotations - Annotations class encapsulating all annotations pertaining to this
        subject.
        -- subject - Subject class encapsulating the information about this subject.
        -- lossModel - Subclass of LossModelBase that quantifies the loss of predicting
        a label given a nominal true label.
        -- annotationModel - Subclass of AnnotationModelBase that computes the posterior
        probability of a particuler classifier assigning a particular label given a
        specific true label, given the annotation history of all classifiers.
        -- annotationModel - Subclass of AnnotationPriorBase that computes the prior
        probability of any classifier assigning a particular label given a specific true
        label.
        """
        if not isinstance(annotations, Annotations):
            raise TypeError(
                'The annotations argument must be of type {}. Type {} passed.'.
                format(type(Annotations).__name__, type(annotations)))
        if not isinstance(subject, Subject):
            raise TypeError(
                'The subject argument must be of type {}. Type {} passed.'.
                format(type(Subject).__name__, type(subject)))
        if not issubclass(type(lossModel), LossModelBase):
            raise TypeError(
                'The lossModel argument must inherit from {}. Type {} passed.'.
                format(type(LossModelBase).__name__, type(lossModel)))
        if not issubclass(type(annotationModel), AnnotationModelBase):
            raise TypeError(
                'The annotationModel argument must inherit from {}. Type {} passed.'.
                format(type(AnnotationModelBase).__name__, type(annotationModel)))
        if not issubclass(type(annotationPriorModel), AnnotationPriorBase):
            raise TypeError(
                'The annotationPriorModel argument must inherit from {}. Type {} passed.'.
                format(type(AnnotationPriorBase).__name__, type(annotationPriorModel)))

        trueLabelRisks = []
        posteriorProbSum = 0
        for trueLabel in annotations.getUniqueLabels():
            posteriorProb = annotationPriorModel(trueLabel) * np.prod([
                annotationModel(trueLabel, annotation)
                for annotation in annotations.items()
            ])
            # print('posteriorProb =>', posteriorProb, 'lossModel(trueLabel, subject.trueLabel) =>', lossModel(trueLabel, subject.trueLabel))
            # print(trueLabel, subject.trueLabel)
            trueLabelRisks.append(
                (lossModel(trueLabel, subject.trueLabel) * posteriorProb))
            posteriorProbSum += posteriorProb
        risk = np.sum(trueLabelRisks)/posteriorProbSum

        return risk
