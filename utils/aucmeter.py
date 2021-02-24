"""
https://github.com/pytorch/tnt/blob/master/LICENSE
The original file is https://github.com/pytorch/tnt/blob/master/torchnet/meter/aucmeter.py
"""

import numbers
import sklearn
import numpy as np
import torch


class AUCMeter(object):
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.
    The AUCMeter is designed to operate on one-dimensional `Tensor`s `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a sigmoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """
    def __init__(self):
        super(AUCMeter, self).__init__()
        self.reset()

    def reset(self):
        """Resets the scores and targets buffers."""
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()
        self.cached = None

    def add(self, output, target):
        """Add scores and target to the buffer.
        :param output: prediction values, can be a torch Tensor or a numpy array
        :param target: target values, can be a torch Tensor, a numpy array or an integer
        """
        if torch.is_tensor(output):
            output = output.cpu().squeeze().numpy()
        elif isinstance(output, list):
            output = np.array(output)

        if torch.is_tensor(target):
            target = target.cpu().squeeze().numpy()
        elif isinstance(target, list):
            target = np.array(target)
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])

        if np.ndim(output) != 1:
            raise ValueError('wrong output size (1D expected)')
        if np.ndim(target) != 1:
            raise ValueError('wrong target size (1D expected)')
        if output.shape[0] != target.shape[0]:
            raise ValueError('number of outputs and targets does not match')
        if not np.all(np.add(np.equal(target, 1), np.equal(target, 0))):
            raise ValueError('targets should be binary (0, 1)')

        self.scores = np.append(self.scores, output)
        self.targets = np.append(self.targets, target)
        self.cached = None

    def value(self, max_fpr=1.0):
        """Computes the auc.
        :param max_fpr: maximum false positive rate to consider, with 0 <= max_fpr <= 1
        :returns: auc score, given the max_fpr rate
        """
        if max_fpr <= 0:
            raise ValueError('max_fpr must be positive')

        # case when number of elements added are 0
        if not self.scores.size or not self.targets.size:
            return 0.5  # means totally random

        if self.cached is not None:
            fpr, tpr = self.cached
        else:
            fpr, tpr, _ = sklearn.metrics.roc_curve(self.targets, self.scores, sample_weight=None)
            self.cached = (fpr, tpr)

        # calculating area under curve using trapezoidal rule
        max_index = np.searchsorted(fpr, [max_fpr], side='right').item()
        area = np.trapz(tpr[:max_index], fpr[:max_index]) / (max_fpr)

        return area