from json_tricks import dumps, load
import numpy as np
from al_utils import RALMSampling
from data_utils import load_data
from sklearn.linear_model import LogisticRegression

FILE = 'data/icml_imdb_small.csv'

## model type
MODEL_TYPE = 'RALM' #or LM
C = 0.7 # for LM, C = 1.0

if MODEL_TYPE == 'LM':
	assert C == 1.0, "Please fix the C Value"

## active learning
TEST_SIZE = 0.33
INIT_BUDGET = 2
AL_STOP = 0.8 # active learning stops at 20% of training labels
N_ROUNDS = 2
SAMPLING = 'uncertain' #or random or attention_uncertain
RESULTS_FILE = '{}_{}_results.json'.format(SAMPLING, MODEL_TYPE)

if __name__ == '__main__':

	auc_tracker = []

	for _ in range(N_ROUNDS):

		## load data
		## pre-compute the features for efficiency
		## this code is for demonstration and very general
		trn_ds, tst_ds, labeler = load_data(FILE, C, TEST_SIZE, INIT_BUDGET)

		print("Active Learning starts")

		## define classifier
		model = LogisticRegression(solver='lbfgs', multi_class='auto')

		qs = RALMSampling(trn_ds, model, SAMPLING)

		## active learning
		tracker = []
		tracker.append(qs.get_auc(tst_ds))

		while(trn_ds.len_unlabeled() > int(len(trn_ds) * AL_STOP)):
		
			ask_id = qs.make_query()
			trn_ds.update(ask_id, labeler[ask_id])
			
			auc = qs.get_auc(tst_ds)

			tracker.append(auc)

		auc_tracker.append(tracker)

	if N_ROUNDS > 1:
		auc_tracker = np.mean(auc_tracker, axis=0).tolist()
	else:
		auc_tracker = auc_tracker[0]

	with open(RESULTS_FILE, 'w') as f:
		f.write(dumps(auc_tracker))

