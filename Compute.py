# Execute Algorithm 1 from "Lean Crowdsourcing: Combining Humans and Machines in an Online System"
# (http://openaccess.thecvf.com/content_cvpr_2017/papers/Branson_Lean_Crowdsourcing_Combining_CVPR_2017_paper.pdf)

from AnnotationModels import AnnotationModelBinary, AnnotationPriorBinary
from Annotations import AnnotationBinary
from Classifiers import Classifiers
from ClassifierSkillModels import (ClassifierSkillModelBinary,
                                   ClassifierSkillPriorBinary)
from IO import CaesarSQSReceiver
from Risk import LossModelBinary, Risk
from Subjects import Subjects

# Model and Prior Model instances
annotationModel = AnnotationModelBinary()
annotationPriorModel = AnnotationPriorBinary()
classifierModel = ClassifierSkillModelBinary()
classifierPriorModel = ClassifierSkillPriorBinary()
lossModel = LossModelBinary(
    falsePosLoss=1, falseNegLoss=1)  # FP and FN just as bad as each other!

# LOOP OVER:
knownSubjects = Subjects([])
for loop in range(5):
    # 1. Obtain annotations
    receiver = CaesarSQSReceiver(
        "https://sqs.us-east-1.amazonaws.com/927935712646/CaesarSpaceWarpsStaging",
        annotationType=AnnotationBinary)
    subjects = receiver.extracts(
        taskName='T0',
        trueValue=1,
        falseValue=0,
        skillModel=classifierModel,
        skillPriorModel=classifierPriorModel)
    knownSubjects.merge(subjects)

# 3. Compute classifier skills (based on previously annotated subjects)
knownClassifiers = Classifiers([
    annotation.classifier for annotation in knownSubjects.annotations
])

for classifier in knownClassifiers.items():
    classifier.computeSkills(knownSubjects)

# Compute best estimate of true labels
for subject in knownSubjects.items():
    subject.computeTrueLabel(
        annotationModel=annotationModel,
        annotationPriorModel=annotationPriorModel)

# 2. Compute subject difficulties.
# TODO: Not currently implemented

# 4. Compute subject risks
riskEvaluator = Risk()
for subject in knownSubjects.items():
    risk = riskEvaluator(
        annotations=subject.annotations,
        subject=subject,
        lossModel=lossModel,
        annotationModel=annotationModel,
        annotationPriorModel=annotationPriorModel)
    print('Subject {}: Risk {}'.format(subject.id, risk))

# 5. Identify subjects for retirement/redployment etc.

# 6. Save results if required.

# 7. Transmit results if required.
