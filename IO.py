# I/O functionality for Bayesian Retrement code

import hashlib
import json
import collections

import numpy as np
import scipy.stats as scistats

import boto3
from Annotations import AnnotationBinary, Annotations
from Classifiers import Classifier, Classifiers
from ClassifierSkillModels import (ClassifierSkillModelBinary,
                                   ClassifierSkillPriorBinary)
from Subjects import Subject, Subjects


class Storage():
    def __init__(self):
        pass


class Receiver():
    def __init__(self):
        pass

    def annotations(self):
        # Receive new annotations and return a new Annotations list
        # NOTE: Annotations implicity encapsulate classifiers.
        pass


class Transmitter():
    def __init__(self):
        pass

    def reductions(self):
        # Transmit appropriately formatted reduction data.
        # NOTE: Expectation is that subclasses will override.
        raise NotImplementedError(
            'This base class does not currently implement this method.')


class UniqueSQSMessage(object):
    def __init__(self, message):
        self.classification_id = int(message['classification_id'])
        self.message = message

    def __eq__(self, other):
        return self.classification_id == other.classification_id

    def __hash__(self):
        return hash(self.classification_id)


class CaesarSQSReceiver(Receiver):
    def __init__(self, queueUrl, annotationType=None):
        # Create immutable SQS client
        self._sqs = boto3.client('sqs')
        self._queueUrl = queueUrl
        self._annotationType = annotationType

    @property
    def sqs(self):
        return self._sqs

    @property
    def queueUrl(self):
        return self._queueUrl

    @queueUrl.setter
    def queueUrl(self, queueUrl):
        self._queueUrl = queueUrl

    @property
    def annotationType(self):
        return self._annotationType

    @annotationType.setter
    def annotationType(self, taskNames):
        self._annotationType = annotationType

    def extracts(self, **extraArgs):
        """ Receive new annotations and return a new Subjects list
        Subjects implicitly encapsulate a list of annotations and annotations
        implicity encapsulate classifiers. Processing of raw extracted
        annotations is delegated to instances of the concrete AnnotationBase
        subclass.

        The intention is that the list of subjects is used to update
        annotations for known subjects or add new subjects with a single
        annotation to the set of known subjects.

        Arguments:
        -- extraArgs - Arguments forwarded to the conrete AnnotationBase
        subclass's constructor and the Classifier constructor.
        """
        uniqueMessages = self.sqsReceive()[0]
        extractSummaries = [
            self.parseExtractSummary(uniqueMessage)
            for uniqueMessage in uniqueMessages
        ]
        # NOTE: Current design passes extracted annotations data to the
        # AnnotationBase subclass's constructor for processing.
        subjects = Subjects([
            Subject(
                id=subjectId,
                annotations=Annotations([
                    self.annotationType(
                        id=classificationId,
                        classifier=Classifier(id=classifierId, **extraArgs),
                        zooniverseAnnotations=zooniverseAnnotations,
                        **extraArgs)
                ])) for classificationId, subjectId, classifierId,
            zooniverseAnnotations in extractSummaries
        ])

        return subjects

    def sqsReceive(self):
        response = self.sqs.receive_message(
            QueueUrl=self.queueUrl,
            AttributeNames=['SentTimestamp', 'MessageDeduplicationId'],
            MaxNumberOfMessages=10,  # Allow up to 10 messages to be received
            MessageAttributeNames=['All'],
            VisibilityTimeout=
            40,  # Allows the message to be retrieved again after 40s
            WaitTimeSeconds=
            20  # Wait at most 20 seconds for an extract enables long polling
        )

        receivedMessageIds = []
        receivedMessages = []
        uniqueMessages = set()

        # Loop over messages
        if 'Messages' in response:
            for message in response['Messages']:
                # extract message body expect a JSON formatted string
                # any information required to deduplicate the message should be
                # present in the message body
                messageBody = message['Body']
                # verify message body integrity
                messageBodyMd5 = hashlib.md5(messageBody.encode()).hexdigest()

                if messageBodyMd5 == message['MD5OfBody']:
                    receivedMessages.append(json.loads(messageBody))
                    receivedMessageIds.append(
                        receivedMessages[-1]['classification_id'])
                    uniqueMessages.add(UniqueSQSMessage(receivedMessages[-1]))
                    # the message has been retrieved successfully - delete it.
                    # self.sqsDelete(message['ReceiptHandle'])

        return [uniqueMessage.message for uniqueMessage in uniqueMessages
                ], receivedMessages, receivedMessageIds

    def sqsDelete(self, receiptHandle):
        self.sqs.delete_message(
            QueueUrl=self.queueUrl, ReceiptHandle=receiptHandle)

    def parseExtractSummary(self, fullExtract):
        # Parse an extract in JSON format and instantiate a new Annotation.
        classificationId = fullExtract['classification_id']
        classifierId = fullExtract['user_id']
        subjectId = fullExtract['subject_id']
        annotations = fullExtract['data']['classification']['annotations']

        return classificationId, subjectId, classifierId, annotations


class CaesarTransmitter(Transmitter):
    def __init__(self):
        pass

    def reductions(self):
        # Transmit appropriately formatted reduction data to Caesar.
        pass


class BinarySimulationReceiver(Receiver):
    def __init__(self, numClassifiers, numSubjects, numAnnotationsPerSubject,
                 trueProb, successProb):
        """Class to simulate reception of binary classifications.

        Parameters
        ----------
        numClassifiers : int
            Number of independent classifiers to simulate.
        numSubjects : int
            Number of distinct subjects to simulate.
        numAnnotationsPerSubject : int
            Number of annotations to simulate for each subject.
            Must be <= numClassifiers
        trueProb : float or array-like with size numClassifiers
            Probability that a subject's correct label is True.
        successProb : float or array-like with size numClassifiers
            Probability that a classifier will correctly classify a
            subject.

        Returns
        -------
        BinarySimulationReciever
            Initialized instance.

        """
        self._numClassifiers = numClassifiers
        self._numSubjects = numSubjects
        self._numAnnotationsPerSubject = min(numAnnotationsPerSubject,
                                             self.numClassifiers)
        self._trueProb = trueProb
        self._successProb = successProb
        self._classifiers = None
        self._annotationIds = [0]

    @property
    def numClassifiers(self):
        return self._numClassifiers

    @numClassifiers.setter
    def numClassfiers(self, numClassifiers):
        if self.numAnnotationsPerSubject > numClassfiers:
            self.numAnnotationsPerSubject = numClassfiers
        self._numClassifiers = numClassifiers

    @property
    def numSubjects(self):
        return self._numSubjects

    @numSubjects.setter
    def numSubjects(self, numSubjects):
        self._numSubjects = numSubjects

    @property
    def numAnnotationsPerSubject(self):
        return self._numAnnotationsPerSubject

    @numAnnotationsPerSubject.setter
    def numAnnotationsPerSubject(self, numAnnotationsPerSubject):
        self._numAnnotationsPerSubject = min(numAnnotationsPerSubject,
                                             self.numClassifiers)

    @property
    def trueProb(self):
        return self._trueProb

    @trueProb.setter
    def trueProb(self, trueProb):
        self._trueProb = trueProb

    @property
    def successProb(self):
        return self._successProb

    @successProb.setter
    def successProb(self, successProb):
        self._successProb = successProb

    @property
    def classifiers(self):
        return self._classifiers

    @classifiers.setter
    def classifiers(self, classifiers):
        self._classifiers = classifiers

    @property
    def annotationIds(self):
        return self._annotationIds

    def genClassifiers(self):
        """Generate a set of simulated classifiers with
        appropriate skill settings.

        Returns
        -------
        None
            Sets the `classifiers` instance attribute directly.

        """
        # Models are simple placeholders
        skillModel = ClassifierSkillModelBinary()
        skillPriorModel = ClassifierSkillPriorBinary()
        self.classifiers = Classifiers([
            Classifier(
                id=classifierId,
                skillModel=skillModel,
                skillPriorModel=skillPriorModel)
            for classifierId in range(self.numClassfiers)
        ])

    def genAnnotationId(self):
        id  = self.annotationIds[-1]
        self.annotationIds.append(id + 1)
        return id

    def genAnnotation(self, classifier, trueLabel, isCorrect):
        """Generate a simulated annotation.

        Returns
        -------
        AnnotationBinary
            A simulated annotation.

        """
        annotation = AnnotationBinary(
            id=self.genAnnotationId(),
            classifier=classifier,
            zooniverseAnnotations={
                'T0': [{
                    'value':
                    bool(trueLabel) if bool(isCorrect) else not bool(trueLabel)
                }]
            },
            taskName='T0',
            trueValue=1,
            falseValue=0)
        return annotation

    def getProbs(self, prob, requiredCount):
        independentProbs = isinstance(prob, collections.abc.Collection) and len(prob) == requiredCount and np.all(np.isreal(prob))

        if independentProbs :
            # independent success probability for each classifier
            indicators = scistats.bernoulli(prob).rvs()
        elif np.isreal(prob):
            # single shared success probability for all classifiers
            indicators = scistats.bernoulli(prob).rvs(
                requiredCount)
        else:
            raise TypeError('BinarySimulationReceiver.getProbs: "prob" must be a real-valued numeric type or an array-like thereof.')

        return indicators

    def genSubjects(self):
        """Generate an ensemble of simulated subjects.

        The true label for each subject is a random variate drawn from a
        Bernoulli distribution with success probability self.trueProb.

        To determine whether a classifier corretly labels the subject
        a random variate is drawn from a Bernoulli distribution with success
        probability self.successProb.

        Returns
        -------
        Subjects
            Collection of simulated subjects.

        """

        if self.classifiers is None:
            self.genClassifiers()

        successIndicators = self.getProbs(self.successProb, self.numClassfiers)

        truthIndicators = self.getProbs(self.trueProb, self.numSubjects)

        # TODO: Simulation should include difficulty once implemented
        # NOTE: At most one annotation per classifier is generated for each subject.
        subjects = Subjects([
            Subject(
                id=subjectId,
                annotations=Annotations([
                    self.genAnnotation(classifier, trueLabel, isCorrect)
                    for isCorrect, classifier in zip(
                        successIndicators,
                        np.random.choice(
                            self.classifiers.classifiers,
                            replace=False,
                            size=self.numAnnotationsPerSubject))
                ]),
                trueLabel=trueLabel) for trueLabel, subjectId in zip(
                    truthIndicators,
                    range(self.numSubjects))
        ])
        return subjects


class SQLiteStorage(Storage):
    pass


class FileStorage(Storage):
    pass


# testSim = BinarySimulationReciever(
#     numClassifiers=200,
#     numSubjects=200,
#     numAnnotationsPerSubject=40,
#     trueProb=0.5,
#     successProb=0.1)
#
# %matplotlib inline
# testSubjects = testSim.genSubjects()
# testSubjects.plotAnnotations()
