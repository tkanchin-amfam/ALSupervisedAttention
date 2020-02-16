from json_tricks import dumps, load
import numpy as np
from keras import backend as K
from ralsa import *
from al_utils import RALSASampling
from data_utils import load_data_nn

FILE = 'data/icml_imdb_small.csv'
## model type
MODEL_TYPE = 'AN' #or 'AN'
SOFTMAX_ATTENTION = True #or 'True' 
if MODEL_TYPE == 'AN':
	assert SOFTMAX_ATTENTION == True, "Please turn on the parameter 'SOFTMAX_ATTENTION'"
## neural network
TOKEN_DIM = 512
ATT_DIM = 128
EPOCHS = 70
BATCH_SIZE = 64
TASK_LOSS_WEIGHT = 1.0
ATT_LOSS_WEIGHT = 1.0
SOFTMAX_ATTENTION = False
CHECKPOINT_DIR = 'checkpoints'
MAX_TOKENS = 150
TOKEN_TYPE = 'sentence' #or word

## active learning
TEST_SIZE = 0.33
VAL_SIZE = 0.33
INIT_BUDGET = 0.05
BUDGET = 0.05
AL_STOP = 0.75 # active learning stops at 25% of training labels
N_ROUNDS = 2
SAMPLING = 'uncertain' #or random or attention_uncertain
RESULTS_FILE = '{}_{}_results.json'.format(SAMPLING, MODEL_TYPE)

def get_model_params():
	return {'MAX_TOKENS': MAX_TOKENS,
	'TOKEN_DIM': TOKEN_DIM, 
	'ATT_DIM': ATT_DIM, 
	'EPOCHS': EPOCHS, 
	'BATCH_SIZE':BATCH_SIZE,
	'TASK_LOSS_WEIGHT': TASK_LOSS_WEIGHT,
	'ATT_LOSS_WEIGHT': ATT_LOSS_WEIGHT,
	'CHECKPOINT_DIR': CHECKPOINT_DIR,
	'SOFTMAX_ATTENTION': SOFTMAX_ATTENTION}

if __name__ == '__main__':
	auc_tracker = []
	for _ in range(N_ROUNDS):
		K.clear_session()
		## load data
		## pre-compute the features for efficiency
		## this code is for demonstration and very general
		trn_ds, tst_ds, val_ds, labeler = load_data_nn(FILE, 
			MAX_TOKENS, TOKEN_TYPE, TEST_SIZE, VAL_SIZE, INIT_BUDGET)
		print("Active Learning starts")
		## define classifier
		if MODEL_TYPE == 'RALSA':
			classifier = RALSA(**get_model_params())
		elif MODEL_TYPE == 'AN':
			model_params = get_model_params()
			model_params['SOFTMAX_ATTENTION'] = False
			classifier = AN(**model_params)
		qs = RALSASampling(trn_ds, val_ds, classifier, SAMPLING)
		## active learning
		tracker = []
		tracker.append(qs.get_auc(tst_ds))
		while(trn_ds.len_unlabeled() > int(len(trn_ds) * AL_STOP)):
			ask_id = qs.make_query(n=int(len(trn_ds) * BUDGET))
			if type(ask_id) == list:
				for aid in ask_id:
					trn_ds.update(aid, labeler[aid])
			else:
				trn_ds.update(ask_id, labeler[ask_id])
			auc = qs.get_auc(tst_ds)
			print(auc)
			tracker.append(auc)
			K.clear_session()
		auc_tracker.append(tracker)
	if N_ROUNDS > 1:
		auc_tracker = np.mean(auc_tracker, axis=0).tolist()
	else:
		auc_tracker = auc_tracker[0]

	with open(RESULTS_FILE, 'w') as f:
		f.write(dumps(auc_tracker))

