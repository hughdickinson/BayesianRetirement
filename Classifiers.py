import numpy as np
import scipy.stats as scistats


class Classifier():
    def __init__(self, skillModel=None, skillPrior=None):
        self._skillModel = skillModel
        self._skillPrior = skillPrior

    @property
    def skillModel(self):
        return self._skillModel

    @skillModel.setter
    def skillModel(self, skillModel):
        self._skillModel = skillModel

    @property
    def skillModel(self):
        return self._skillModel

    @skillModel.setter
    def skillModel(self, skillModel):
        self._skillModel = skillModel


class Classifiers():
    def __init__(self, classifiers):
        self._classifiers = [
            classifier for classifier in classifiers
            if isinstance(classifier, Classifier)
        ]

    @property
    def classifiers(self):
        return self._classifiers

    @classifiers.setter
    def classifiers(self, classifiers):
        self._classifiers = [
            classifier for classifier in classifiers
            if isinstance(classifier, Classifier)
        ]

    def items(self):
        for classifier in self._classifiers:
            yield classifier


class ClassifierSkillPriorBase():
    def __call__(self, annotation, classifier, **args):
        pass


class ClassifierSkillPriorBinary(ClassifierSkillPriorBase):
    def __call__(self, subjects, **args):
        """Evaluate a beta PDF prior for all (should be 2!) labels.
        The prior is evaluated by summing over all classifiers and subjects.

        Keyword arguments:
        -- nBeta - Real-valued coefficient controlling the strength of the prior.
        Roughly, the prior dominates the worker skill until nBeta classifications
        have been performed. Thereafter the annotation history dominates. Default is:
        5.0.
        -- lowCountProb - Real-valued argument that overrides the Bernoulli probability
        model for both labels until sufficient annotations have been performed. Default
        is: 0.8.
        -- lowCountThreshold - The number of annotations required before the Bernoulli
        probability model is used in place of the override value. Default is: 2.

        Returns: Dictionary with labels as keys and computed skill priors as values.
        """
        if not isinstance(subjects, Subjects):
            raise TypeError(
                'The subjects argument must be of type {}. Type {} passed.'.
                format(type(Subjects), type(subjects)))

        nBeta = args.get('nBeta', 5.0)
        lowCountProb = args.get('lowCountProb', 0.8)
        lowCountThreshold = args.get('lowCountThreshold', 2)

        uniqueLabels = np.unique([
            annotation.label for subject in subjects
            for annotation in subject.annotations
        ])
        skillPriors = {uniqueLabel: None for uniqueLabel in uniqueLabels}
        #TODO: Does the this formulation assume a value of lowCountThreshold?
        for trueLabel in uniqueLabels:
            # Obtain a list of subjects with true (or consensus) label matching trueLabel.
            subjectsMatchingTrueLabel = subjects.subset(label=trueLabel)
            # Obtain the list of all (matching and non-matching) labels for subjects
            # with true (or consensus) label matching trueLabel.
            labelsForSubjectsMatchingTrueLabel = np.asarray([
                annotation.label for subject in subjectsMatchingTrueLabel
                for annotation in subject.annotations
            ])
            # Count the total number of (matching and non-matching) predictions.
            nLabelsForSubjectsMatchingTrueLabel = labelsForSubjectsMatchingTrueLabel.size
            # Count the total number of matching predictionsself.
            nCorrectLabelsForSubjectsMatchingTrueLabel = np.sum(
                labelsForSubjectsMatchingTrueLabel == trueLabel)
            # Compute the value of the prior model.
            skillPriors.update({
                trueLabel: (nBeta * lowCountProb +
                            nCorrectLabelsForSubjectsMatchingTrueLabel) /
                (nBeta + nLabelsForSubjectsMatchingTrueLabel)
            })
        return skillPriors


class ClassifierSkillModelBase():
    def __call__(self, annotation, classifier, **args):
        pass


class ClassifierSkillModelBinary(ClassifierSkillPriorBase):
    """For a binary classification task, the probability model is Bernoulli.
    """

    def __call__(self, classifier, subjects, probModel, **args):
        """Evaluate a Beta PDF skill model for all (should be 2!) labels
        for a single classfier.
        The model is evaluated by summing over all subjects for a single
        classifier.

        Keyword arguments:
        -- nBeta - Real-valued coefficient controlling the strength of the prior.
        Roughly, the prior dominates the worker skill until nBeta classifications
        have been performed. Thereafter the annotation history dominates. Default is:
        5.0.
        -- lowCountProb - Real-valued argument that overrides the Bernoulli probability
        model for both labels until sufficient annotations have been performed. Default
        is: 0.8.
        -- lowCountThreshold - The number of annotations required before the Bernoulli
        probability model is used in place of the override value. Default is: 2.

        Returns: Dictionary with labels as keys and computed skills as values.
        """
        if not isinstance(classifier, Classifier):
            raise TypeError(
                'The classifier argument must be of type {}. Type {} passed.'.
                format(type(Classifier), type(classifier)))
        if not isinstance(subjects, Subjects):
            raise TypeError(
                'The subjects argument must be of type {}. Type {} passed.'.
                format(type(Subjects), type(subjects)))

        nBeta = args.get('nBeta', 5.0)
        lowCountProb = args.get('lowCountProb', 0.8)
        lowCountThreshold = args.get('lowCountThreshold', 2)

        # Get annotations for this classifier
        classifierAnnotations = np.asarray([
            annotation for subject in subjects
            for annotation in subject.annotations
            if id(annotation.classifier) == id(self)
        ])

        # Get unique labels for this classifier
        uniqueLabels = np.unique([
            annotation.label for subject in subjects
            for annotation in classifierAnnotations
        ])

        # Get subjects with annotations provided by this classifier
        classifierSubjects = Subjects([
            subject for subject in subjects if id(classifier) in
            [id(annotation.classifier) for annotation in subject.annotations]
        ])

        skills = {uniqueLabel: None for uniqueLabel in uniqueLabels}

        #TODO: Does the this formulation assume a value of lowCountThreshold?
        # Subsequent line differs from the prior model since only a single
        # classifier's annotations are considered.
        for trueLabel in uniqueLabels:
            # Obtain a list of subjects with true (or consensus) label matching trueLabel.
            subjectsMatchingTrueLabel = classifierSubjects.subset(label=trueLabel)
            # Obtain the list of all (matching and non-matching) labels for subjects
            # with true (or consensus) label matching trueLabel.
            labelsForSubjectsMatchingTrueLabel = np.asarray([
                annotation.label for subject in subjectsMatchingTrueLabel
                for annotation in subject.annotations if id(annotation.classfier) == classifier
            ])
            # Count the total number of (matching and non-matching) predictions.
            nLabelsForSubjectsMatchingTrueLabel = labelsForSubjectsMatchingTrueLabel.size
            # Count the total number of matching predictionsself.
            nCorrectLabelsForSubjectsMatchingTrueLabel = np.sum(
                labelsForSubjectsMatchingTrueLabel == trueLabel)
            # Compute the value of the prior model.
            skills.update({
                trueLabel: (nBeta * probModel(trueLabel) +
                            nCorrectLabelsForSubjectsMatchingTrueLabel) /
                (nBeta + nLabelsForSubjectsMatchingTrueLabel)
            })
        return skills