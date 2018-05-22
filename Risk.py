# Subjects(Difficulty) -> Annotations -> Classifiers(Skill)
import scipy.stats as scistats

from Annotations import Annotation, Annotations
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


class AnnotationPriorBase():

    validLabels = AnnotationModelBase.validLabels
    """Subclasses compute the prior probability of a subject having a particular
    true label.
    """

    def __init__(self):
        self._priorsForLabels = None


class AnnotationPriorBinary():
    """Implements a Beta distribution prior for the annotation provided
    by a single classifier.
    """

    validLabels = AnnotationModelBinary.validLabels

    def __init__(self, **args):
        super.__init()

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


class AnnotationModelBase():

    validLabels = []
    """Subclasses compute the probability of classifiers assigning a particular label
    given a particular true label.
    """
    pass


class AnnotationModelBinary(AnnotationModelBase):
    """Implements a Bernoulli model of a classifier assigning
    annotationLabel when trueLabel is true.
    """

    validLabels = [True, False]

    def __init__(self, **args):
        super.__init(**args)

    def __call__(self, trueLabel, annotation):
        """Computes the probability of a classifier assigning annotationLabel when
        trueLabel is true.

        Depends upon the precomputation of the classifier's skill. Recall that
        the skill is defined as the probability of assigning any valid label given
        a specific true label.
        """
        return annotation.classifier.skills[trueLabel] if (
            trueLabel == annotation.label
        ) else 1.0 - annotation.classifier.skills[trueLabel]


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
                format(type(Annotations), type(annotations)))
        if not isinstance(subject, Subject):
            raise TypeError(
                'The subject argument must be of type {}. Type {} passed.'.
                format(type(Subject), type(subject)))
        if not issubclass(lossModel, LossModelBase):
            raise TypeError(
                'The lossModel argument must inherit from {}. Type {} passed.'.
                format(type(LossModelBase), type(lossModel)))
        if not issubclass(annotationModel, AnnotationModelBase):
            raise TypeError(
                'The annotationModel argument must inherit from {}. Type {} passed.'.
                format(type(AnnotationModelBase), type(annotationModel)))
        if not issubclass(annotationPriorModel, AnnotationPriorBase):
            raise TypeError(
                'The annotationPriorModel argument must inherit from {}. Type {} passed.'.
                format(type(AnnotationPriorBase), type(annotationPriorModel)))

        # TODO: This should be called before computing classifier skills upon which it
        # depends! In other words needs to be moved to a main driver routine somewhere.
        subject.computeTrueLabel()

        trueLabelRisks = []
        for trueLabel in annotations.getUniqueLabels():
            posteriorProb = annotationPriorModel(trueLabel) * np.prod([
                annotationModel(trueLabel, annotation)
                for annotation in annotations
            ])
            trueLabelRisks.append(
                (lossModel(trueLabel, subject.trueLabel()) * posteriorProb) / posteriorProb)
        risk = np.sum(trueLabelRisks)
