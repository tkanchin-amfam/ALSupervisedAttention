from sklearn.model_selection import train_test_split
import numpy as np, pandas as pd
from nltk import tokenize
import ast
from keras.utils import to_categorical

class Dataset(object):

	def __init__(self, X, R, Y):
		self.data = list(zip(X, R, Y))

	def __len__(self):
		return len(self.data)

	def len_labeled(self):
		return len(self.get_labeled_entries())

	def len_unlabeled(self):
		return len(list(filter(lambda entry: entry[2] is None, self.data)))

	def get_num_of_labels(self):
		return len({entry[2] for entry in self.data if entry[2] is not None})

	def update(self, entry_id, new_label):
		self.last_entry_id = entry_id
		self.data[entry_id] = (self.data[entry_id][0], self.data[entry_id][1], new_label)

	def get_last_entry(self):
		return self.data[self.last_entry_id]

	def format_sklearn(self, train=False, one_hot=True):
		X, R, Y = zip(*self.get_labeled_entries())
		if one_hot:
			Y = to_categorical(Y, num_classes=2)
		return np.array(X), np.array(R), np.array(Y) 

	def get_entries(self):
		return self.data

	def get_labeled_entries(self):
		return list(filter(lambda entry: entry[2] is not None, self.data))

	def get_unlabeled_entries(self):
		return [(idx, entry[0]) for idx, entry in enumerate(self.data) if entry[2] is None]

class TFHubExtract(object):

	def __init__(self, path="http://tfhub.dev/google/universal-sentence-encoder/4"):

		path = '/Users/tsk014/Desktop/Rocket2/backend_fico/artifacts/tf_hub_model'

		import tensorflow as tf
		import tensorflow_hub as hub

		# Create graph and finalize (finalizing optional but recommended).
		g = tf.Graph()
		with g.as_default():
			# We will be feeding 1D tensors of text into the graph.
			self.text_input = tf.placeholder(dtype=tf.string, shape=[None])
			embed = hub.Module(path)
			self.embedded_text = embed(self.text_input)
			init_op = tf.group([tf.global_variables_initializer(), tf.tables_initializer()])
		g.finalize()
		self.session = tf.Session(graph=g)	
		self.session.run(init_op)

	def predict(self, x):
		return self.session.run(self.embedded_text, feed_dict={self.text_input: x})

def pad_truncate(X, R, MAX_TOKENS, SENT_DIM=512):
	
	## truncate
	X = [x[:MAX_TOKENS] if len(x) > MAX_TOKENS else x for x in X]
	R = [x[:MAX_TOKENS] if len(x) > MAX_TOKENS else x for x in R]

	## pad
	X = np.array([x + [[0]*SENT_DIM for _ in range(MAX_TOKENS-len(x))] for x in X])
	R = np.array([x + [0 for _ in range(MAX_TOKENS-len(x))] for x in R])

	return X, R

def load_data_nn(FILE, MAX_TOKENS, TOKEN_TYPE, TEST_SIZE, VAL_SIZE, INIT_BUDGET):
	
	'''
	@ FILE: csv file with column names: documents, rationales, label
	@ MAX_TOKENS: maximum tokens per document
	@ TOKEN_TYPE: word or sentence token
	'''

	## load dataframe as csv file
	df = pd.read_csv(FILE)
	df['rationales'] = df['rationales'].apply(lambda x: ast.literal_eval(x))

	if TOKEN_TYPE == 'word':
		df['document_tokens'] = df['documents'].apply(lambda x: tokenize.word_tokenize(x))
	elif TOKEN_TYPE == 'sentence':
		df['document_tokens'] = df['documents'].apply(lambda x: tokenize.sent_tokenize(x))



	tfHub = TFHubExtract()
	df['features'] = df['document_tokens'].apply(lambda x: tfHub.predict(x).tolist())
	
	print("Extracted Features")

	## calculate R
	def get_overlap(sentence, rationales):
		for rationale in rationales:
			if rationale in sentence:
				return True
		return False
	
	rationales = df['rationales'].tolist()
	document_tokens = df['document_tokens'].tolist()
	df['overlap'] = [[1 if get_overlap(y, rationales[x]) else 0 for y in document_tokens[x]] for x in range(len(document_tokens))]	

	## data prep
	X = np.array(df['features'].tolist())	
	Y = np.array(df['labels'].tolist())
	Y[Y == -1] = 0
	R = np.array(df['overlap'].tolist())

	# pad and truncate
	X, R = pad_truncate(X, R, MAX_TOKENS)

	X_tr, X_te, Y_tr, Y_te, R_tr, R_te = train_test_split(X, Y, R, test_size=TEST_SIZE, stratify=Y, random_state=64)
	X_tr, X_val, Y_tr, Y_val, R_tr, R_val = train_test_split(X_tr, Y_tr, R_tr, test_size=VAL_SIZE, stratify=Y_tr, random_state=64)
	
	n_labeled = int(X_tr.shape[0] * INIT_BUDGET)
	X_ul, X_l, Y_ul, Y_l, R_ul, R_l = train_test_split(X_tr, Y_tr, R_tr, test_size=n_labeled, stratify=Y_tr, random_state=64)

	trn_ds = Dataset(np.concatenate((X_ul, X_l), axis=0),
		np.concatenate((R_ul, R_l), axis=0),
		np.concatenate([[None] * (len(Y_tr) - n_labeled), Y_l]))	
	tst_ds = Dataset(X_te, R_te, Y_te)
	val_ds = Dataset(X_val, R_val, Y_val)	

	return trn_ds, tst_ds, val_ds, Y_ul


def load_data(FILE, C, TEST_SIZE, INIT_BUDGET):
	'''
	@ FILE: csv file with column names: documents, rationales, label
	@ C: value to weight document and rationale feaures
	'''

	## load dataframe as csv file
	df = pd.read_csv(FILE)
	df['rationales'] = df['rationales'].apply(lambda x: ast.literal_eval(x))
	df['rationales'] = df['rationales'].apply(lambda x: ' '.join(x))

	tfHub = TFHubExtract()
	document_features = tfHub.predict(df['documents'].tolist()).tolist()
	rationales_features = tfHub.predict(df['rationales'].tolist()).tolist()

	print("Extracted Features")

	X = C * np.array(document_features) + (1-C)*np.array(rationales_features)
	Y = np.array(df['labels'].tolist())
	Y[Y == -1] = 0

	R = np.zeros(len(Y))

	X_tr, X_te, Y_tr, Y_te, R_tr, R_te = train_test_split(X, Y, R, test_size=TEST_SIZE, stratify=Y, random_state=64)
	
	X_ul, X_l, Y_ul, Y_l, R_ul, R_l = train_test_split(X_tr, Y_tr, R_tr, test_size=INIT_BUDGET, stratify=Y_tr, random_state=64)

	trn_ds = Dataset(np.concatenate((X_ul, X_l), axis=0),
		np.concatenate((R_ul, R_l), axis=0),
		np.concatenate([[None] * (len(Y_tr) - INIT_BUDGET), Y_l]))	
	tst_ds = Dataset(X_te, R_te, Y_te)

	return trn_ds, tst_ds, Y_ul





