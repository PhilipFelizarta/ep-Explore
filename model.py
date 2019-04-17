import keras
from keras.models import Model, model_from_json
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, MaxPooling2D, AveragePooling2D, Concatenate, Dropout, AlphaDropout
from keras.layers import GlobalAveragePooling2D, Multiply, Permute, Reshape
from keras.initializers import glorot_uniform
from keras.callbacks import TensorBoard, EarlyStopping
from keras import regularizers
from keras import backend as K

import helper
import chess
import chess.pgn

import numpy as np


class Agent(object): #Will contain all the functions needed to train our resnet
	def __init__(self, res_blocks, filters, lambd, name, lr=0.02):
		self.res_blocks = res_blocks #Resblock number
		self.filters = filters #number of 3x3 filters per resblock
		self.lambd = lambd #lambda for l2 reg
		self.name = name #For file naming
		self.lr = lr #Learning rate
		self.model = self.create_model() #Create model for state-space exploration
		self.running_model = self.create_model() #Create a running-average mod
		self.train_count = 0 #Track the number of times the agent has been trained
		self.elo = 0 #Begin with 0 elo!
		#self.FIM = 0 fischer information matrix for Elastic Weight consolidation

	def res_block(self, X, block, size=3, train=True): #Standard resblock with skip connection after two convolutions
		bn_name = 'bn_block' + block

		res = Conv2D(filters=self.filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', name='res_block1_' + block, kernel_initializer=glorot_uniform(), 
			trainable=train, kernel_regularizer=regularizers.l2(self.lambd))(X)
		res = BatchNormalization(axis=-1, name=bn_name + '_1', epsilon=1e-5)(res)
		res = Activation('relu')(res)
		res = Conv2D(filters=self.filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', name='res_block2_' + block, kernel_initializer=glorot_uniform(), 
			trainable=train, kernel_regularizer=regularizers.l2(self.lambd))(res)
		res = self.se_block(res, train=train)
		res = BatchNormalization(axis=-1, name=bn_name + '_2', epsilon=1e-5)(res)
		X = Add()([X, res])
		X = Activation('relu')(X)

		return X

	def se_block(self, X, ratio=4, train=True): #Segment Excitation Block
		se = GlobalAveragePooling2D(data_format='channels_last')(X)
		se = Dense(self.filters // ratio, activation='relu', trainable=train, kernel_regularizer=regularizers.l2(self.lambd))(se)
		se = Dense(self.filters, activation='sigmoid', trainable=train, kernel_regularizer=regularizers.l2(self.lambd))(se)
		se = Reshape((1, 1, self.filters))(se)
		se = Multiply()([X, se])
		return se

	def stem(self, X, filters, stage='stem', size=3, train=True): #Stem block for inputs and heads
		stem = Conv2D(filters=filters, kernel_size=(size,size), strides=(1,1), padding='same', data_format='channels_last', name='Conv_' + stage, kernel_initializer=glorot_uniform(), 
			trainable=train, kernel_regularizer=regularizers.l2(self.lambd))(X)
		stem = BatchNormalization(axis=-1, name="bn_block" + stage + '_1', epsilon=1e-5)(stem)
		stem = Activation('relu')(stem)
		return stem

	def create_model(self, train=True, dropout=0.5): #Create a resnet
		inputs = Input(shape=(8,8,12)) #Single input

		#Stem Layer
		X = self.stem(inputs, self.filters, train=train)

		#resnet
		for i in range(self.res_blocks):
			X = self.res_block(X, str(i + 1), train=train)

		#policy head output
		policy = self.stem(X, 32, stage='policy', size=1, train=train)
		policy = Flatten()(policy)
		policy = Dropout(dropout)(policy)
		policy = Dense(1858, activation='softmax', name='fcPolicy', kernel_initializer=glorot_uniform(), 
			kernel_regularizer =regularizers.l2(self.lambd), trainable=train)(policy)

		#value head output
		value = self.stem(X, 32, stage='value', size=1, train=train)
		value = Flatten()(value)
		value = Dropout(dropout)(value)
		value = Dense(256, activation='relu', name='denseValue', kernel_initializer=glorot_uniform(), 
			kernel_regularizer=regularizers.l2(self.lambd), trainable=train)(value)
		value = Dropout(dropout)(value)
		value = Dense(1, activation='tanh', name='fcValue', kernel_initializer=glorot_uniform(),
			kernel_regularizer=regularizers.l2(self.lambd), trainable=train)(value)

		#Create model
		model = Model(inputs= inputs, outputs= [policy,value] , name='ep-Explore')

		optimizer = keras.optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=True)

		model.compile(optimizer=optimizer, loss={'fcPolicy': "categorical_crossentropy", 'fcValue': "mean_squared_error"}, metrics={'fcPolicy':'accuracy'})

		return model

	def train(self, inputs, pLabel, vLabel, lr, input_val=None, output_val=None): #Train with SWA (const learning rate)
		early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

		#Set new learning rate
		self.lr = lr
		
		#Create an sgd.. train it
		model_sgd = self.create_model()
		model_sgd.set_weights(self.model.get_weights())
		if input_val is not None:
			model_sgd.fit(inputs, [pLabel, vLabel], epochs=30, validation_data=(input_val, output_val), batch_size=1024, callbacks=[early])
		else:
			model_sgd.fit(inputs, [pLabel, vLabel], epochs=30, validation_split=0.02, batch_size=1024, callbacks=[early])


		#Create an swa model... train it!
		model_swa = self.create_model()
		model_swa.set_weights(model_sgd.get_weights())


		for i in range(25):
			print((i + 1))
			if input_val is not None:
				model_sgd.fit(inputs, [pLabel, vLabel], epochs=1, validation_data=(input_val, output_val), batch_size=1024)
			else:
				model_sgd.fit(inputs, [pLabel, vLabel], epochs=1, validation_split=0.02, batch_size=1024)
			swa = np.array(model_swa.get_weights()) * (i + 1)
			sgd = np.array(model_sgd.get_weights())
			swa = (swa + sgd)/(i + 2)

			new_weights = list()
			for weight in swa: #Append new weights layer by layer
				new_weights.append(weight)

			#Create new model with the weights
			model_swa.set_weights(new_weights)

		#Fix batch normalization on swa model
		self.fix_bn(model_swa, inputs, pLabel, vLabel, input_val=input_val, output_val=output_val)

		#update our running average model over all the times we have trained
		if self.train_count == 0:
			self.running_model.set_weights(model_swa.get_weights())
		else:
			run = np.array(self.running_model.get_weights()) * self.train_count
			new = np.array(model_swa.get_weights())

			update = (run + new)/(self.train_count + 1)

			new_weights = list()
			for weight in update:
				new_weights.append(weight)

			self.running_model.set_weights(new_weights)
			print("Running Average")
			self.fix_bn(self.running_model, inputs, pLabel, vLabel, input_val=input_val, output_val=output_val)

		self.train_count += 1
		
		#simulate a set between the previous model and the new one
		elo_gain = self.simulate_set(self.model, model_swa, setNum=100)
		self.elo += elo_gain

		#Calculate/Update the fischer information matrix? (not implemented)
		#FIM_new = self.calc_FIM(model_swa, inputs, pLabel, vLabel)
		#self.FIM += FIM_new 

		#update model
		self.model = model_swa

	def fix_bn(self, model, inputs, pLabel, vLabel, input_val=None, output_val=None):
		#Create a new model with frozen weights (batch norm not frozen)
		new_model = self.create_model(train=False)
		new_model.set_weights(model.get_weights())

		#We will train the frozen model's batch normalization layers
		optimizer = keras.optimizers.SGD(lr=self.lr, momentum=0.9, decay=0.0, nesterov=True)
		new_model.compile(optimizer=optimizer, loss={'fcPolicy': "categorical_crossentropy", 'fcValue': "mean_squared_error"}, metrics={'fcPolicy':'accuracy'})

		if input_val is not None:
			new_model.fit(inputs, [pLabel, vLabel], validation_data=(input_val, output_val), epochs=1, batch_size=1024)
		else:
			new_model.fit(inputs, [pLabel, vLabel], validation_split=0.02, epochs=1, batch_size=1024)
		#Now place the new weights back
		model.set_weights(new_model.get_weights())

	def ep_move(self, board): #Given a python-chess board, move with epsilon-greedy exploration
		turn = board.turn
		k_castle = board.has_kingside_castling_rights(turn)
		q_castle = board.has_queenside_castling_rights(turn)
		kb_castle = board.has_kingside_castling_rights(not turn)
		qb_castle = board.has_queenside_castling_rights(not turn)

		check = board.is_check()

		position = helper.conversion(str(board), turn, k_castle, q_castle, check, kb_castle, qb_castle)

		#We will be exploring the statespace using only the policy-head
		policy, _ = self.model.predict(position)

		policy = policy.reshape(1858, 1)
		move = helper.policy_to_move(policy, board, dirichlet=True)

		#Push move now
		try:
			board.push_uci(move)
		except:
			print("Error with move")
			for legal_move in board.legal_moves:
				board.push(legal_move)
				break

	def move(self, model, board, temp=1.0, dirichlet=True): #Given a python-chess board, give a normal move
		turn = board.turn
		k_castle = board.has_kingside_castling_rights(turn)
		q_castle = board.has_queenside_castling_rights(turn)
		kb_castle = board.has_kingside_castling_rights(not turn)
		qb_castle = board.has_queenside_castling_rights(not turn)

		check = board.is_check()

		position = helper.conversion(str(board), turn, k_castle, q_castle, check, kb_castle, qb_castle)

		#We will be exploring the statespace using only the policy-head
		policy, _ = model.predict(position)

		policy = policy.reshape(1858, 1)
		move = helper.policy_to_move(policy, board, dirichlet=dirichlet, temp=temp)

		#Push move now
		try:
			board.push_uci(move)
		except:
			print("Error with move")
			for legal_move in board.legal_moves:
				board.push(legal_move)
				break

	def simulate_set(self, modelOne, modelTwo, setNum=50):
		from tqdm import tqdm

		one_score = 0
		two_score = 0
		draws = 0

		for i in tqdm(range(setNum)):
			board = chess.Board()
			count = 0

			#Make the openings noisy for diverse set
			dirichlet = True
			temp = 5.0
			while not board.is_game_over():
				if count > 6: #Cutoff at 6 ply
					dirichlet = True
					temp = 0.5

				if count % 2 == 0:
					if i % 2 == 0:
						self.move(modelOne, board, temp=temp, dirichlet=dirichlet)
					else:
						self.move(modelTwo, board, temp=temp, dirichlet=dirichlet)
				else:
					if i % 2 == 0:
						self.move(modelTwo, board, temp=temp, dirichlet=dirichlet)
					else:
						self.move(modelOne, board, temp=temp, dirichlet=dirichlet)

				count += 1

			if board.is_checkmate():
				if board.turn: #If its white's turn
					if i % 2 == 0: #model one is white
						two_score += 1
					else:
						one_score += 1
				else:
					if i % 2 == 0: #model two is white
						one_score += 1
					else:
						two_score += 1
			else:
				draws += 1

			game = chess.pgn.Game().from_board(board)
			game.headers["Event"] = "Game " + str(i + 1)
			game.headers["Site"] = "RTX Titan"
			game.headers["Round"] = "Training"
			if i % 2 == 0:
				game.headers["White"] = "model one"
				game.headers["Black"] = "model two"
			else:
				game.headers["White"] = "model two"
				game.headers["Black"] = "model one"

			name = "games/" + str(int(self.train_count))+ "/setPlay_Game" + str(i) + ".pgn"
			new_pgn = open(name, "w", encoding="utf-8")
			exporter = chess.pgn.FileExporter(new_pgn)
			game.accept(exporter)
	
		print("New model: ", two_score)
		print("Old_model: ", one_score)
		print("Draws: ", draws)

		win_rate = (two_score + 0.5*draws)/setNum

		if win_rate < 1:
			elo_gain = -400 * np.log10((1/win_rate) - 1)
		else:
			elo_gain = 1000

		return elo_gain















