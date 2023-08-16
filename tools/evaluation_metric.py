import numpy as np
from scipy import stats
from collections import Counter
import sklearn
from sklearn import metrics

class Purity:
    def __init__(self, preds, labels):
        contingency_matrix = metrics.cluster.contingency_matrix(labels, preds)
        self.purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)

    def __call__(self):
        return self.purity

class ClassificationAcc:
    def __init__(self, preds, labels, num_classes):
        super().__init__()
        self.preds = np.array(preds)
        self.labels = np.array(labels)
        self.num_classes = num_classes
        self.count_matrix = np.zeros((self.num_classes,self.num_classes), dtype=np.int32)
        self.TP_class = np.zeros((self.num_classes), dtype=np.int32)
        self.T_class = np.zeros((self.num_classes), dtype=np.int32)
        self.P_class = np.zeros((self.num_classes), dtype=np.int32)

        for i in range(len(self.preds)):
            self.count_matrix[self.labels[i]][self.preds[i]]+=1
        self.recall_matrix = self.count_matrix / np.sum(self.count_matrix, axis=-1)[...,None]
        self.precision_matrix = self.count_matrix / np.sum(self.count_matrix, axis=0)[None,...]

        for c in range(0, num_classes):
            self.TP_class[c] += np.sum((self.preds == c) & (self.labels == c))
            self.P_class[c] += np.sum(self.preds == c)
            self.T_class[c] += np.sum(self.labels == c)
        self.precision_class = self.TP_class / (self.P_class + 0.0001)
        self.recall_class = self.TP_class / self.T_class
        self.precision = np.sum(self.TP_class) / np.sum(self.P_class)
        self.recall = np.sum(self.TP_class) / np.sum(self.T_class)
        self.avg_precision = np.mean(self.precision_class)
        self.avg_recall = np.mean(self.recall_class)
        self.accuracy = np.sum(self.count_matrix * np.identity(num_classes)) / len(self.preds)

        self.results = {'precision': self.avg_precision, 'recall': self.avg_recall, 'count_matrix': self.count_matrix, 'accuracy': self.accuracy}

    def __call__(self):
        return self.results

    @classmethod
    def aggregate(cls, results_list):
        f_results = {key:0 for key in results_list[0].keys()}
        for results in results_list:
            for key, value in results.items():
                f_results[key] += value
        f_results = {key:value/len(results_list) for key,value in f_results.items()}
        return f_results


class ClusterEvalIoU:
    def __init__(self, preds, labels, IoU_thres=0.5):
        super().__init__()
        self.preds = preds
        self.labels = labels
        self.IoU_thres = IoU_thres

        unique_labels = Counter(list(self.labels))
        unique_preds = Counter(list(self.preds))

        self.TP = 0
        for cluster_id in unique_labels.keys():
            if cluster_id == -1 : continue
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            mode, count = stats.mode(self.preds[point_ids], axis=None, keepdims=False)
            IoU = count / (unique_labels[cluster_id]+unique_preds[mode]-count)
            if mode>-1 and IoU >= IoU_thres:
                self.TP+=1

        self.P = len(unique_preds) - (1 if unique_preds[-1]!=0 else 0)
        self.T = len(unique_labels) - (1 if unique_labels[-1]!=0 else 0)
        self.precision = 0 if self.P==0 else self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = 0 if self.T==0 else self.TP / self.T  # what percent of actual clusters are identified
        self.F1 = 0 if (self.precision==0 or self.recall==0) else 2 * self.precision * self.recall / (self.precision + self.recall)

    def __call__(self):
        return self.precision, self.recall

class ClusterEvalMode:
    def __init__(self, preds, labels):
        super().__init__()
        self.preds = preds
        self.labels = labels

        unique_labels = Counter(list(self.labels))
        unique_preds = Counter(list(self.preds))

        self.TP = 0
        for cluster_id in unique_labels.keys():
            if cluster_id == -1 : continue
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            mode_pred, count_pred = stats.mode(self.preds[point_ids], axis=None, keepdims=False)
            point_ids_preds = np.argwhere(self.preds == mode_pred)[:,0]
            mode_label, count_label = stats.mode(self.labels[point_ids_preds], axis=None, keepdims=False)
            if mode_pred>-1 and mode_label>-1 and mode_label == cluster_id:
                self.TP += 1

        self.P = len(unique_preds) - (1 if unique_preds[-1]!=0 else 0)
        self.T = len(unique_labels) - (1 if unique_labels[-1]!=0 else 0)
        self.precision = 0 if self.P==0 else self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = 0 if self.T==0 else self.TP / self.T  # what percent of actual clusters are identified
        self.F1 = 0 if (self.precision==0 or self.recall==0) else 2 * self.precision * self.recall / (self.precision + self.recall)

    def __call__(self):
        return self.precision, self.recall


class ClusterEvalModeSoft:
    def __init__(self, preds, labels):
        super().__init__()
        self.preds = preds
        self.labels = labels

        unique_labels = Counter(list(self.labels))
        unique_preds_c = self.preds.shape[-1]

        self.TP = 0
        for cluster_id in unique_labels.keys():
            if cluster_id == -1 : continue
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            ecounts = np.sum(self.preds[point_ids], axis=0)
            mode_pred = np.argmax(ecounts)
            ecounts = np.bincount(self.labels, self.preds[:,mode_pred])
            mode_label = np.argmax(ecounts)
            if mode_pred>-1 and mode_label>-1 and mode_label == cluster_id:
                self.TP += 1

        self.P = unique_preds_c
        self.T = len(unique_labels) - (1 if unique_labels[-1]!=0 else 0)
        self.precision = 0 if self.P==0 else self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = 0 if self.T==0 else self.TP / self.T  # what percent of actual clusters are identified
        self.F1 = 0 if (self.precision==0 or self.recall==0) else 2 * self.precision * self.recall / (self.precision + self.recall)

    def __call__(self):
        return self.precision, self.recall

class ClusterEvalModeC:
    def __init__(self, preds, labels):
        super().__init__()
        self.preds = preds
        self.labels = labels

        unique_labels = set(list(self.labels))

        self.TP_C = 0
        for cluster_id in unique_labels:
            if cluster_id == -1 : continue
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            mode_pred, count_pred = stats.mode(self.preds[point_ids], axis=None, keepdims=False)
            point_ids_preds = np.argwhere(self.preds == mode_pred)[:,0]
            mode_label, count_label = stats.mode(self.labels[point_ids_preds], axis=None, keepdims=False)
            if mode_pred>-1 and mode_label>-1 and mode_label == cluster_id:
                self.TP_C += count_label

        self.recall_C = self.TP_C / len(self.labels)

    def __call__(self):
        return self.recall_C

class ClusterEvalModeCSoft:
    def __init__(self, preds, labels):
        super().__init__()
        self.preds = preds
        self.labels = labels

        unique_labels = set(list(self.labels))

        self.TP_C = 0
        for cluster_id in unique_labels:
            if cluster_id == -1 : continue
            point_ids = np.argwhere(self.labels == cluster_id)[:,0]
            ecounts = np.sum(self.preds[point_ids], axis=0)
            mode_pred = np.argmax(ecounts)
            count_pred = ecounts[mode_pred]
            ecounts = np.bincount(self.labels, self.preds[:,mode_pred])
            mode_label = np.argmax(ecounts)
            count_label = ecounts[mode_label]
            if mode_pred>-1 and mode_label>-1 and mode_label == cluster_id:
                assert abs(count_label-count_pred)<0.1
                self.TP_C += count_label

        self.recall_C = self.TP_C / len(self.labels)

    def __call__(self):
        return self.recall_C

class ClusterEvalSoft:
    def __init__(self, preds, labels):
        super().__init__()
        self.preds = preds
        self.labels = labels
        self.noise_mask = labels == -1
        
        unique_labels = np.unique(self.labels)
        unique_labels = unique_labels[unique_labels != -1]
        reverse_map = {label:i for i,label in enumerate(unique_labels)}
        reverse_map[-1] = 0 # it doesn't matter, we will overwrite it
        self.labels = np.array([reverse_map[label] for label in self.labels])
        c_count = np.max(self.labels)+1

        C = np.zeros((len(self.labels), c_count))
        ids = np.arange(len(self.labels))
        C[ids, self.labels[ids]] = 1
        C[self.noise_mask] = 0

        SP = np.log(self.preds + 0.0001)
        SN = np.log(1.0001 - self.preds)

        SS = np.matmul(np.transpose(C), SP) + np.matmul(np.transpose(1-C), SN)
        column_modes = np.argmax(SS, axis=-1)
        row_modes = np.argmax(SS, axis=0)
        rows = np.arange(SS.shape[0])
        chosen_rows = rows[row_modes[column_modes[rows]] == rows]
#        print(chosen_rows)
#        print(column_modes[chosen_rows])

        self.TP = len(chosen_rows)
        self.P = SS.shape[-1]
        self.T = SS.shape[0]
        self.precision = 0 if self.P==0 else self.TP / self.P  # what percent of clusters are actual clusters
        self.recall = 0 if self.T==0 else self.TP / self.T  # what percent of actual clusters are identified
        self.F1 = 0 if (self.precision==0 or self.recall==0) else 2 * self.precision * self.recall / (self.precision + self.recall)

    def __call__(self):
        return self.precision, self.recall

class ClusterEvalAll:
    def __init__(self, preds, labels):
        super().__init__()
        self.results = {}
        preds = preds - np.min(preds) # start at 0, there cannot be -1
        if len(preds.shape)>1 :
            preds_ = preds
            preds = np.argmax(preds, axis=-1)
        else:
            K = np.max(preds)+1
            preds_ = np.ones((len(preds), K)) * 0.01/(K-1)
            preds_[np.arange(len(preds)), preds] = 0.99
        IoU = ClusterEvalIoU(preds, labels, IoU_thres=0.5)
        self.results['IoU_TP'] = IoU.TP
        self.results['IoU_T'] = IoU.T
        self.results['IoU_P'] = IoU.P
        self.results['IoU_precision'] = IoU.precision
        self.results['IoU_recall'] = IoU.recall
        self.results['IoU_F1'] = IoU.F1

        Mode = ClusterEvalMode(preds, labels)
        self.results['Mode_TP'] = Mode.TP
        self.results['Mode_T'] = Mode.T
        self.results['Mode_P'] = Mode.P
        self.results['Mode_precision'] = Mode.precision
        self.results['Mode_recall'] = Mode.recall
        self.results['Mode_F1'] = Mode.F1

        ModeC = ClusterEvalModeC(preds, labels)
        self.results['Mode_TP_C'] = ModeC.TP_C
        self.results['Mode_recall_C'] = ModeC.recall_C

        purity = Purity(preds, labels)
        self.results['Purity'] = purity()

        AMI = sklearn.metrics.adjusted_mutual_info_score(labels, preds)
        self.results['AMI'] = AMI

        ARand = sklearn.metrics.adjusted_rand_score(labels, preds)
        self.results['ARand'] = ARand

        if preds_ is not None:
            ModeProb = ClusterEvalSoft(preds_, labels)
            self.results['ModeProb_TP'] = ModeProb.TP
            self.results['ModeProb_T'] = ModeProb.T
            self.results['ModeProb_P'] = ModeProb.P
            self.results['ModeProb_precision'] = ModeProb.precision
            self.results['ModeProb_recall'] = ModeProb.recall
            self.results['ModeProb_F1'] = ModeProb.F1

    def __call__(self):
        return self.results

    @classmethod
    def aggregate(cls, results_list):
        f_results = {key:0 for key in results_list[0].keys()}
        for results in results_list:
            for key, value in results.items():
                f_results[key] += value
        f_results = {key:value/len(results_list) for key,value in f_results.items()}
        return f_results


