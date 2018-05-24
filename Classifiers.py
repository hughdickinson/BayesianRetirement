import numpy as np


class Classifier():
    def __init__(self,
                 id=None,
                 skillModel=None,
                 skillPriorModel=None,
                 **kwargs):
        self._id = id
        self._skillModel = skillModel
        self._skillPriorModel = skillPriorModel
        self._skillPriors = None
        self._skills = None
        # TODO: Should classifier maintain a list of their annotations for computational efficiency?

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
    def skills(self):
        if self._skills is None:
            raise RuntimeError(
                'Classifier skills have not been computed. Call computeSkills() first.'
            )
        return self._skills

    @property
    def skillPriors(self):
        if self._skillPriors is None:
            raise RuntimeError(
                'Classifier skill priors have not been computed. Call computeSkillPriors() first.'
            )
        return self._skillPriors

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

    def computeSkillPriors(self, subjects, initMode, **args):
        self._skillPriors = self._skillPriorModel(
            subjects, initMode=initMode, **args)

    def computeSkills(self, subjects, initMode=False, **args):
        if self._skillPriors is None:
            self.computeSkillPriors(subjects, initMode=initMode, **args)
            print('self.skillPriors => {}'.format(self.skillPriors))
        self._skills = self.skillModel(self, subjects, self.skillPriors,
                                       initMode, **args)

    def __str__(self):
        return '\n'.join(['-**Classifier**-'] + [
            '{} => {}'.format(name[1:], value)
            for name, value in vars(self).items()
        ] + ['-**Classifier**-'])


class Classifiers():
    def __init__(self, classifiers=[]):
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
        for classifier in self.classifiers:
            yield classifier

    def append(self, classifier):
        if isinstance(type(classifier), Classifier):
            self.classifiers.append(classifier)
        else:
            raise TypeError(
                'The classfier argument must an instance of type {}. Type {} passed.'.
                format(type(Classifier), type(classifier)))

    def __str__(self):
        return '\n'.join(str(classifier) for classifier in self.classifiers)
