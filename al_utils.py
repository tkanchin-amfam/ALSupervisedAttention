import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
import math

class RALSASampling:
	def __init__(self, dataset, validation, model, query):
		self.dataset = dataset
		self.validation = validation
		self.model = model
		self.query = query
		self.train()

	def train(self):
		X_tr, R_tr, Y_tr = self.dataset.format_sklearn()
		X_val, R_val, Y_val = self.validation.format_sklearn()
		self.model.train((X_tr, Y_tr, R_tr), (X_val, Y_val, R_val))

	def make_query(self, n=1):
		self.train()
		unlabeled_entry_ids, X_pool = zip(*self.dataset.get_unlabeled_entries())
		if self.query == 'uncertain':
			pred = self.model.predict(np.array(X_pool))[0]
			pred = np.sum(-pred * np.log(pred), axis=1) ## entropy
			if n == 1:
				ask_id = np.argmax(pred)
				return unlabeled_entry_ids[ask_id]
			else:
				## top-n elements in decreasing order
				ask_ids = pred.argsort()[-n:]
				return [unlabeled_entry_ids[x] for x in ask_ids]
		elif self.query == 'attention_uncertain':
			pred = self.model.predict(np.array(X_pool))
			task_entropy = np.sum(-pred[0]*np.log(pred[0]), axis=1) ## entropy task
			att_entropy = np.sum(-(pred[1]*np.log(pred[1]) + (1-pred[1])*np.log(1-pred[1])), axis=1) ## token attention task
			len_unlabeled = self.dataset.len_unlabeled()
			len_labeled = self.dataset.len_labeled()
			lambda_p = float(len_unlabeled)/len_labeled
			lambda_p = math.floor(((1+np.tanh(lambda_p))/2) * 100)/100
			pred = lambda_p*(task_entropy) + (1-lambda_p)*(att_entropy)
			ask_ids = pred.argsort()[-n:]
			return [unlabeled_entry_ids[x] for x in ask_ids]
		elif self.query == 'random':
			try:
				return np.random.choice(unlabeled_entry_ids, n, replace=False).tolist()
			except:
				return list(set(np.random.choice(unlabeled_entry_ids, n, replace=True).tolist()))

	def get_auc(self, tst_ds):
		X, R, Y = tst_ds.format_sklearn()
		pred = self.model.predict(X)[0]
		return roc_auc_score(np.argmax(Y, axis=1), pred[:, 1])

class RALMSampling:
	def __init__(self, dataset, model, qc='uncertain'):
		self.dataset = dataset
		self.model = model
		self.qc = qc
		self.train()

	def train(self):
		X, _, Y= self.dataset.format_sklearn(one_hot=False)
		self.model.fit(X, Y)

	def make_query(self):
		self.train()
		unlabeled_entry_ids, X_pool = zip(*self.dataset.get_unlabeled_entries())
		if self.qc == 'random':
			return np.random.choice(unlabeled_entry_ids, 1, replace=False)[0]
		elif self.qc == 'uncertain':
			ask_id = np.argmax(-np.max(self.model.predict_proba(X_pool), axis=1))
			return unlabeled_entry_ids[ask_id]

	def get_auc(self, tst_ds):
		X, _, Y = tst_ds.format_sklearn(one_hot=False)
		pred = self.model.decision_function(X)
		return roc_auc_score(Y, pred)

