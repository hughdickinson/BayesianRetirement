import matplotlib.pyplot as mplplot
import numpy as np

# from Annotations import Annotations


class LabelType():
    def __init__(self, labelTypeName='Undefined'):
        self._labelTypeName = labelTypeName

    @property
    def labelTypeName(self):
        return self._labelTypeName

    def __str__(self):
        return self.labelTypeName


class RealValuedLabelType(LabelType):
    def __init__(self):
        super().__init__('Real')

    @staticmethod
    def getNumLabels(self, **args):
        return np.inf


class IntValuedLabelType(LabelType):
    def __init__(self):
        super().__init__('Int')

    @staticmethod
    def getNumLabels(self, **args):
        # NOTE: Introspection "hack" to avoid having to include Annotations
        if 'annotations' not in args or not (
                'getUniqueLabels' in dir(args['annotations'])
                and callable(args['annotations'].getUniqueLabels)):
            return 0

        return args['annotations'].getUniqueLabels()


class BoolValuedLabelType(LabelType):
    def __init__(self):
        super().__init__('Bool')

    @staticmethod
    def getNumLabels(self, **args):
        return 2


#NOTE: Cannot be mapped to IntValuedLabelType since plotting behaviour is different.
class CategoricalLabelType(LabelType):
    def __init__(self):
        super().__init__('Categorical')

    @staticmethod
    def getNumLabels(self, **args):
        if 'annotations' not in args or not (
                'getUniqueLabels' in dir(args['annotations'])
                and callable(args['annotations'].getUniqueLabels)):
            return 0

        return args['annotations'].getUniqueLabels()
