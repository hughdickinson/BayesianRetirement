import numpy as np
import scipy.stats as scistats

from Subjects import Subjects


class ClassifierSkillPriorBase():
    pass


class ClassifierSkillPriorBinary(ClassifierSkillPriorBase):
    def __call__(self, subjects, initMode, **args):
        """Evaluate a beta PDF prior for all (should be 2!) labels.
        The prior is evaluated by summing over all classifiers and subjects.

        Intuitively, the method computes the probability of any classifier assigning
        a label given that a specific label is true.

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
            annotation.label for subject in subjects.items()
            for annotation in subject.annotations.items()
        ])
        # if initializing assume a random classifier model
        if initMode:
            return {
                uniqueLabel: 1.0 / len(uniqueLabels)
                for uniqueLabel in uniqueLabels
            }

        skillPriors = {uniqueLabel: None for uniqueLabel in uniqueLabels}
        #TODO: Does the this formulation assume a value of lowCountThreshold? Seems to be 2.
        for trueLabel in uniqueLabels:
            # Obtain a list of subjects with true (or consensus) label matching trueLabel.
            subjectsMatchingTrueLabel = subjects.subset(trueLabel=trueLabel)
            # Obtain the list of all (matching and non-matching) labels for subjects
            # with true (or consensus) label matching trueLabel.
            labelsForSubjectsMatchingTrueLabel = np.asarray([
                annotation.label
                for subject in subjectsMatchingTrueLabel.items()
                for annotation in subject.annotations.items()
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
    pass


class ClassifierSkillModelBinary(ClassifierSkillPriorBase):
    """For a binary classification task, the probability model is Bernoulli.
    """

    def __call__(self, classifier, subjects, priors, initMode, **args):
        """Evaluate a Beta PDF skill model for all (should be 2!) labels
        for a single classfier.
        The model is evaluated by summing over all subjects for a single
        classifier.

        Intuitively, the method computes the probability of this classifier assigning
        a label given that a specific label is true.

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
        # if not isinstance(classifier, Classifier):
        #     raise TypeError(
        #         'The classifier argument must be of type {}. Type {} passed.'.
        #         format(type(Classifier), type(classifier)))
        if not isinstance(subjects, Subjects):
            raise TypeError(
                'The subjects argument must be of type {}. Type {} passed.'.
                format(type(Subjects), type(subjects)))
        # if not issubclass(type(priorModel), ClassifierSkillPriorBase):
        #     raise TypeError(
        #         'The priorModel argument must a subclass of type {}. Type {} passed.'.
        #         format(type(ClassifierSkillPriorBase), type(priorModel)))

        if initMode:
            return priors

        nBeta = args.get('nBeta', 5.0)
        lowCountProb = args.get('lowCountProb', 0.8)
        lowCountThreshold = args.get('lowCountThreshold', 2)

        # Get annotations for this classifier
        classifierAnnotations = np.asarray([
            annotation for subject in subjects.items()
            for annotation in subject.annotations.items()
            if annotation.classifier.id == classifier.id
        ])

        print('classifierAnnotations => {}'.format(classifierAnnotations))

        # Get unique labels for this classifier
        uniqueLabels = np.unique([
            annotation.label for subject in subjects.items()
            for annotation in classifierAnnotations
        ])

        print('uniqueLabels => {}'.format(uniqueLabels))

        # Get subjects with annotations provided by this classifier
        classifierSubjects = Subjects([
            subject for subject in subjects.items() if id(classifier) in [
                id(annotation.classifier)
                for annotation in subject.annotations.items()
            ]
        ])

        skills = {uniqueLabel: None for uniqueLabel in uniqueLabels}

        print('skills => '.format(skills))

        #TODO: Does the this formulation assume a value of lowCountThreshold?
        # Subsequent line differs from the prior model since only a single
        # classifier's annotations are considered.
        for trueLabel in uniqueLabels:
            # Obtain a list of subjects with true (or consensus) label matching trueLabel.
            subjectsMatchingTrueLabel = classifierSubjects.subset(
                trueLabel=trueLabel)
            # Obtain the list of all (matching and non-matching) labels for subjects
            # with true (or consensus) label matching trueLabel.
            labelsForSubjectsMatchingTrueLabel = np.asarray([
                annotation.label
                for subject in subjectsMatchingTrueLabel.items()
                for annotation in subject.annotations.items()
                if annotation.classifier.id == classifier.id
            ])
            # Count the total number of (matching and non-matching) predictions.
            nLabelsForSubjectsMatchingTrueLabel = labelsForSubjectsMatchingTrueLabel.size
            # Count the total number of matching predictionsself.
            nCorrectLabelsForSubjectsMatchingTrueLabel = np.sum(
                labelsForSubjectsMatchingTrueLabel == trueLabel)
            # Compute the value of the model.
            print('priors => {}'.format(priors))
            skills.update({
                trueLabel: (nBeta * priors[trueLabel] +
                            nCorrectLabelsForSubjectsMatchingTrueLabel) /
                (nBeta + nLabelsForSubjectsMatchingTrueLabel)
            })
        return skills
