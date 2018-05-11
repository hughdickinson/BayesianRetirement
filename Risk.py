# Subjects -> Annotations -> Classifiers

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


class Risk():
    """Computes a metric that can be used to define decision thresholds for each
    subject based on its classification history.
    """

    def __call__(self, annotations, subject, lossModel):
        """Evaluate the risk.
        Arguments:
        -- annotations - Annotations class encapsulating all annotations pertaining to this
        subject.
        -- subject - Subject class encapsulating the information about this subject.
        -- lossModel - Subclass of LossModelBase that quantifies the loss of predicting
        a label given a nominal true label.
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
                'The subject argument must inherit from {}. Type {} passed.'.
                format(type(LossModelBase), type(lossModel)))
        pass
