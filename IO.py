# I/O functionality for Bayesian Retrement code

import hashlib
import json

import boto3

from Annotations import Annotations
from Classifiers import Classifier
from Subjects import Subject, Subjects


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

    def extracts(self, **annotationArgs):
        """ Receive new annotations and return a new Subjects list
        Subjects implicitly encapsulate a list of annotations and annotations
        implicity encapsulate classifiers. Processing of raw extracted
        annotations is delegated to instances of the concrete AnnotationBase
        subclass.

        The intention is that the list of subjects is used to update
        annotations for known subjects or add new subjects with a single
        annotation to the set of known subjects.

        Arguments:
        -- annotationArgs - Arguments forwarded to the conrete AnnotationBase
        subclass's constructor.
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
                        classifier=classifier,
                        zooniverseAnnotations=zooniverseAnnotations,
                        **annotationArgs)
                ])) for classificationId, subjectId, classifier,
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
        classifier = Classifier(id=fullExtract['user_id'])
        subjectId = fullExtract['subject_id']
        annotations = fullExtract['data']['classification']['annotations']

        return classificationId, subjectId, classifier, annotations


class CaesarTransmitter(Transmitter):
    def __init__(self):
        pass

    def reductions(self):
        # Transmit appropriately formatted reduction data to Caesar.
        pass


class Storage():
    pass


class SQLiteStorage(Storage):
    pass


class FileStorage(Storage):
    pass
