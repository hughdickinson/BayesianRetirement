import numpy as np

from Annotations import AnnotationBase, Annotations


class Subject():
    def __init__(self,
                 id=None,
                 annotations=None,
                 trueLabel=None,
                 difficulty=None):
        self._id = id
        self._annotations = annotations if annotations is not None else Annotations(
            [])
        self._difficulty = difficulty
        self._trueLabel = trueLabel

    def __eq__(self, other):
        return self.id == other.id

    def __hash__(self):
        return hash(self.id)

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        self._annotations = Annotations([
            annotation for annotation in annotations
            if issubclass(annotation, AnnotationBase)
        ])

    @property
    def difficulty(self):
        return self._difficulty

    @difficulty.setter
    def difficulty(self, difficulty):
        self._difficulty = difficulty

    @property
    def trueLabel(self):
        return self._trueLabel

    @trueLabel.setter
    def trueLabel(self, trueLabel):
        self._trueLabel = trueLabel

    def computeTrueLabel(self, annotationPriorModel):
        validLabels = self.annotations.getUniqueLabels()
        # Predict subject label
        labelMlEstimates = []
        for trueLabel in validLabels:
            if len(self.annotations.annotations) > 0:
                dataProb = np.product([
                    labelModel(trueLabel, annotation)
                    for annotation in self.annotations.items()
                ])
            else:
                dataProb = 1.0
            labelMlEstimates.append(annotationPriorModel(trueLabel) * dataProb)

        self._trueLabel = validLabels[np.argmax(labelMlEstimates)]

    def __str__(self):
        return '\n'.join(['-==Subject==-'] + [
            '{} => {}'.format(name[1:], value)
            for name, value in vars(self).items()
        ] + ['-==Subject==-'])


class Subjects():
    def __init__(self, subjects=[]):
        self._subjects = [
            subject for subject in subjects if isinstance(subject, Subject)
        ]

    @property
    def subjects(self):
        return self._subjects

    @subjects.setter
    def subjects(self, classifiers):
        self._subjects = [
            subject for subject in subjects if isinstance(subject, Subject)
        ]

    def items(self):
        for subject in self.subjects:
            yield subject

    def append(self, subject):
        if isinstance(subject, Subject):
            self.subjects.append(subject)
        else:
            raise TypeError(
                'The subject argument must an instance of type {}. Type {} passed.'.
                format(Subject, type(subject)))

    def merge(self, subjects):
        for subject in subjects.items():
            for knownSubject in self.items():
                if knownSubject.id == subject.id:
                    print(True)
                    knownSubject.annotations.append(subject.annotations)
                    break
            else:
                print(False)
                self.append(subject)

    def subsetCriterion(self, subject, id, trueLabel):
        # Returns True by default if id and trueLabel are None
        if id is not None and subject.id != id or trueLabel is not None and trueLabel != trueLabel:
            return False
        return True

    def subset(self, id=None, trueLabel=None):
        return Subjects([
            subject for subject in self.subjects
            if isinstance(subject, Subject) and subsetCriterion(subject, id, trueLabel)
        ])

    def __str__(self):
        return '\n'.join(str(subject) for subject in self.subjects)


class SubjectDifficultyPriorBase():
    pass


class SubjectDifficultyModelBase():
    pass
