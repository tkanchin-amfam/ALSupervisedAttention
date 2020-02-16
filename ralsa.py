import tensorflow as tf
from keras.layers import Embedding, Dense, Input, GRU, Bidirectional, TimeDistributed, Masking, Dropout
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import ModelCheckpoint
from keras import initializers
from keras.models import load_model
from keras.losses import binary_crossentropy
import os, shutil

'''
## ref: https://stackoverflow.com/questions/43547402/how-to-calculate-f1-macro-in-keras
'''
def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision
	
	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2*((precision*recall)/(precision+recall+K.epsilon()))

class AttLayer(Layer):
	def __init__(self, attention_dim, softmax_attention, **kwargs):
		self.init = initializers.get('normal')
		self.supports_masking = True
		self.attention_dim = attention_dim
		self.softmax_attention = softmax_attention
		super(AttLayer, self).__init__(**kwargs)

	def build(self, input_shape):
		assert len(input_shape) == 3
		self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
		self.b = K.variable(self.init((self.attention_dim, )))
		self.u = K.variable(self.init((self.attention_dim, 1)))
		self.trainable_weights = [self.W, self.b, self.u]
		super(AttLayer, self).build(input_shape)

	def compute_mask(self, inputs, mask=None):
		return None

	def call(self, x, mask=None):
		# size of x :[batch_size, sel_len, attention_dim]
		# size of u :[batch_size, attention_dim]
		# uit = tanh(xW+b)
		uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
		ait = K.squeeze(K.dot(uit, self.u), -1)
		if self.softmax_attention:
			ait = K.exp(ait)
			if mask is not None:
				# Cast the mask to floatX to avoid float64 upcasting in theano
				ait *= K.cast(mask, K.floatx())		
			ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		else:
			ait = K.sigmoid(ait)
			if mask is not None:
				# Cast the mask to floatX to avoid float64 upcasting in theano
				ait *= K.cast(mask, K.floatx())	
		weighted_input = x * K.expand_dims(ait)
		output = K.sum(weighted_input, axis=1)
		return [output, ait]

	def compute_output_shape(self, input_shape):
		return [(input_shape[0], input_shape[-1]), (input_shape[0], input_shape[1])]

	def get_config(self):
		config = {'attention_dim': self.attention_dim, 'softmax_attention': self.softmax_attention}
		base_config = super(AttLayer, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

class RALSA(object):

	def __init__(self, **kwargs):
		for k in kwargs.keys():
			self.__setattr__(k, kwargs[k])

	def get_model(self):
		sent_input = Input(shape=(self.MAX_TOKENS, self.TOKEN_DIM), dtype='float32')
		s_att = AttLayer(self.ATT_DIM, self.SOFTMAX_ATTENTION, name='att')(sent_input)
		preds = Dense(2, activation='softmax', name='dense_1')(s_att[0])
		model = Model(sent_input, [preds, s_att[1]])
		model.compile(loss={'dense_1': 'binary_crossentropy', 'att':'binary_crossentropy'}, 
			loss_weights={'dense_1':self.TASK_LOSS_WEIGHT, 'att':self.ATT_LOSS_WEIGHT}, optimizer='adam', metrics=[f1])
		return model

	def train(self, data_tr, data_val):

		## make checkpoints directory
		if not os.path.exists(self.CHECKPOINT_DIR):
			os.makedirs(self.CHECKPOINT_DIR)
		else:
			shutil.rmtree(self.CHECKPOINT_DIR)
			os.makedirs(self.CHECKPOINT_DIR)
		file_path = '{}/model.h5'.format(self.CHECKPOINT_DIR)
		self.checkpoint = ModelCheckpoint(file_path, monitor='val_dense_1_f1', save_best_only=True, mode='max')
		self.model = self.get_model()
		self.fit(data_tr, data_val)

	def fit(self, data_tr, data_val):
		self.model.fit(data_tr[0], [data_tr[1], data_tr[2]], 
			validation_data=(data_val[0], [data_val[1], data_val[2]]), 
			epochs=self.EPOCHS, 
			batch_size=self.BATCH_SIZE, 
			verbose=0, 
			callbacks=[self.checkpoint])

	def predict(self, data_te):
		model = load_model('{}/model.h5'.format(self.CHECKPOINT_DIR), 
			custom_objects={'AttLayer':AttLayer, 'f1': f1})
		return model.predict(data_te)

class AN(object):
	def __init__(self, **kwargs):
		for k in kwargs.keys():
			self.__setattr__(k, kwargs[k])

	def get_model(self):
		sent_input = Input(shape=(self.MAX_TOKENS, self.TOKEN_DIM), dtype='float32')		
		s_att = AttLayer(self.ATT_DIM, self.SOFTMAX_ATTENTION, name='att')(sent_input)
		preds = Dense(2, activation='softmax', name='dense_1')(s_att[0])
		model = Model(sent_input, preds)
		model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[f1])
		return model

	def train(self, data_tr, data_val):

		## make checkpoints directory
		if not os.path.exists(self.CHECKPOINT_DIR):
			os.makedirs(self.CHECKPOINT_DIR)
		else:
			shutil.rmtree(self.CHECKPOINT_DIR)
			os.makedirs(self.CHECKPOINT_DIR)
		file_path = '{}/model.h5'.format(self.CHECKPOINT_DIR)
		self.checkpoint = ModelCheckpoint(file_path, monitor='val_f1', save_best_only=True, mode='max')
		self.model = self.get_model()
		self.fit(data_tr, data_val)

	def fit(self, data_tr, data_val):
		self.model.fit(data_tr[0], data_tr[1], 
			validation_data=(data_val[0], data_val[1]), 
			epochs=self.EPOCHS, 
			batch_size=self.BATCH_SIZE, 
			verbose=0, 
			callbacks=[self.checkpoint])

	def predict(self, data_te):
		model = load_model('{}/model.h5'.format(self.CHECKPOINT_DIR), 
			custom_objects={'AttLayer':AttLayer, 'f1': f1})
		pred =  model.predict(data_te)
		return pred, pred

class Predictor(object):

	def __init__(self, checkpoint, softmax_attention):
		self.checkpoint = checkpoint
		self.softmax_attention = softmax_attention
		self.get_model()
		self.set_att_params()
		print(self.model.summary())

	def get_model(self):
		self.model = load_model(self.checkpoint, custom_objects={'AttLayer':AttLayer, 'f1': f1})
    
	def set_att_params(self):
		self.att_weights = self.model.get_layer("att").W
		self.att_bias = self.model.get_layer("att").b
		self.att_context = self.model.get_layer("att").u

	def get_att_weights(self, data_te):
		uit = K.tanh(K.bias_add(K.dot(K.constant(data_te[0]), self.att_weights), self.att_bias))
		zit = K.dot(uit, self.att_context)
		if self.softmax_attention:
			ait = K.squeeze(ait, -1)
			ait = K.exp(ait)
			ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
		else:
			ait = K.sigmoid(K.squeeze(zit, -1))
		return K.eval(ait)

	def predict_proba(self, data_te):
		pred = self.model.predict(data_te)
		if type(pred) == tuple:
			return pred
		else:
			return pred, pred
