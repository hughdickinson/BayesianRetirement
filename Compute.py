# Execute Algorithm 1 from "Lean Crowdsourcing: Combining Humans and Machines in an Online System"
# (http://openaccess.thecvf.com/content_cvpr_2017/papers/Branson_Lean_Crowdsourcing_Combining_CVPR_2017_paper.pdf)

from Annotations import AnnotationBinary
from IO import CaesarSQSReceiver
from Subjects import Subjects

# LOOP OVER:
knownSubjects = Subjects([])
for loop in range(5):
    # 1. Obtain annotations
    receiver = CaesarSQSReceiver(
        "https://sqs.us-east-1.amazonaws.com/927935712646/CaesarSpaceWarpsStaging",
        annotationType=AnnotationBinary)
    subjects = receiver.extracts(
        taskName='T0', trueValue=1, falseValue=0)
    knownSubjects.merge(subjects)

# 2. Compute subject difficulties.
# TODO: Not currently implemented

# 3. Compute classifier skills (based on previously annotated subjects)

# 4. Compute subject risks

# 5. Identify subjects for retirement/redployment etc.

# 6. Save results if required.

# 7. Transmit results if required.
