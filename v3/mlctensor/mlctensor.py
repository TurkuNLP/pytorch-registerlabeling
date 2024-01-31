#
# Multi-Label Confusion Tensor
# as proposed by Krstinic et al.
#
#
import numpy as np


# Set use_unknown to False to compute confusion tensor without the additional class "unknown"
class mlcTensor:
    def __init__(self, gt=None, pred=None, use_unknown=True):
        self.use_unknown = use_unknown
        self.confusion_tensor = None

        if gt is not None:
            self.setTrueLabels(gt)
        if pred is not None:
            self.setPredictedLabels(pred)

    def setTrueLabels(self, gt):
        self.q = gt.shape[1]
        if self.use_unknown:
            self.gt = self.addUnknown(gt.T)
            self.q += 1
        else:
            self.gt = gt.T

    def setPredictedLabels(self, pred):
        if self.gt is None:
            raise Exception("GT labels not povided")

        if self.use_unknown:
            self.pred = self.addUnknown(pred.T)
        else:
            self.pred = pred.T

        if self.gt.shape != self.pred.shape:
            self.pred = None
            raise Exception("GT labels and predictions do not match")

    def addUnknown(self, m):
        h, w = m.shape
        mu = np.zeros((h + 1, w))
        mu[:h, :] = m
        mu[h, :] = (np.sum(mu, axis=0) == 0).astype(float)
        return mu

    def getContribution2(self, t, p):
        tc = np.sum(t)
        pc = np.sum(p)
        t1 = np.logical_and(t, p).astype(t.dtype)  # p1=t1

        t2 = t - t1
        p2 = p - t1

        CR = np.diag(t1) + np.outer(t2, p) / pc
        CP = np.diag(t1) + np.outer(t, p2) / tc
        return np.stack((CR, CP))

    def getContribution1(self, t, p):
        C = np.zeros((2, self.q, self.q))

        tc = np.sum(t)
        pc = np.sum(p)

        t1 = np.logical_and(t, p).astype(t.dtype)  # p1=t1

        t2 = t - t1
        p2 = p - t1

        t2c = np.sum(t2)
        p2c = np.sum(p2)

        CR = C[0, :, :]
        CP = C[1, :, :]

        CR += np.diag(t1)
        CP += np.diag(t1)
        if np.array_equal(t, p):
            return C

        if t2c == 0 and p2c > 0:
            CP += np.outer(t, p2) / tc
        elif t2c > 0 and p2c == 0:
            CR += np.outer(t2, p) / pc
        else:
            CR += np.outer(t2, p2) / p2c
            CP += np.outer(t2, p2) / t2c

        return C

    # Set unique to False to compute contribution to Confusion Tensor differently
    # for different patterns of True and Predicted labels
    def computeConfusionTensor(self, gt=None, pred=None, unique=True):
        if gt is not None:
            self.setTrueLabels(gt)
        if pred is not None:
            self.setPredictedLabels(pred)

        if unique:
            contribution_func = self.getContribution2
        else:
            contribution_func = self.getContribution1

        self.confusion_tensor = np.zeros((2, self.q, self.q))
        nsamples = self.gt.shape[1]
        for k in range(nsamples):
            g = self.gt[:, k]
            p = self.pred[:, k]
            self.confusion_tensor += contribution_func(g, p)

        return self.confusion_tensor

    def getConfusionMatrix(self):
        return self.confusion_matrix

    def getRecall(self):
        R = self.confusion_tensor[0, :, :].copy()
        s = R.sum(axis=1)
        for k in range(R.shape[0]):
            if s[k] > 0:
                R[k, :] = R[k, :] / s[k]
        return R

    def getPrecision(self):
        P = self.confusion_tensor[1, :, :].copy()
        s = P.sum(axis=0)
        for k in range(P.shape[1]):
            if s[k] > 0:
                P[:, k] = P[:, k] / s[k]
        return P

    def getPrecisionByClass(self):
        if self.confusion_matrix is None:
            return None
        p = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=0)
        return p

    def getRecallByClass(self):
        if self.confusion_matrix is None:
            return None
        r = self.confusion_matrix.diagonal() / self.confusion_matrix.sum(axis=1)
        return r

    def getInstanceContribution(self, i):
        g = self.gt[:, i]
        p = self.pred[:, i]
        return self.getContribution(g, p)

    def getOneVsRest(self):
        vvr = np.zeros((self.q, 2, 2))
        vvr[:, 1, 1] = self.confusion_matrix.diagonal()
        vvr[:, 0, 0] = vvr[:, 1, 1].sum() - vvr[:, 1, 1]
        vvr[:, 0, 1] = self.confusion_matrix.sum(axis=0) - vvr[:, 1, 1]
        vvr[:, 1, 0] = self.confusion_matrix.sum(axis=1) - vvr[:, 1, 1]
        return vvr
