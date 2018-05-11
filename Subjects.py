class Subject():
    def __init__(self, annotations=None, trueLabel=None, difficulty=None):
        self._annotations = annotations
        self._difficulty = difficulty

    @property
    def annotations(self):
        return self._annotations

    @annotations.setter
    def annotations(self, annotations):
        self._annotations = [annotation for annotation in annotations if issubclass(annotation, AnnotationBase)

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


class Subjects():
    def __init__(self, subjects):
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
        for subject in self._subjects:
            yield subject

    def subset(self, trueLabel):
        return Subjects([
            subject for subject in self._subjects
            if isinstance(subject, Subject) and subject.trueLabel == trueLabel
        ])


class SubjectDifficultyPriorBase():
    pass


class SubjectDifficultyModelBase():
    pass
