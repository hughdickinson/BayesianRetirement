# Execute Algorithm 1 from "Lean Crowdsourcing: Combining Humans and Machines in an Online System"
# (http://openaccess.thecvf.com/content_cvpr_2017/papers/Branson_Lean_Crowdsourcing_Combining_CVPR_2017_paper.pdf)

from IO import CaesarSQSReceiver

# LOOP OVER:
    # 1. Obtain annotations
receiver = CaesarSQSReceiver("https://sqs.us-east-1.amazonaws.com/927935712646/CaesarSpaceWarpsStaging")
print([list(extract.keys()) for extract in receiver.sqsReceive()[0]])
    # TODO: Update list of known subjects?

    # 2. Compute subject difficulties.
    # TODO: Not currently implemented

    # 3. Compute classifier skills.

    # 4. Compute subject risks

    # 5. Identify subjects for retirement/redployment etc.

    # 6. Save results if required.

    # 7. Transmit results if required.
