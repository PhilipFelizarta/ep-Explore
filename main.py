
#This is our multiprocessing loop!
def play_from(board, positions, moves, values, lock): #Possible improvement could be placing our data into tuples?? post-processing required tho
	import chess
	import chess.uci
	import numpy as np
	from helper import conversion, one_hot

	stockfish = chess.uci.popen_engine("stockfish")
	info_handler = chess.uci.InfoHandler()
	stockfish.info_handlers.append(info_handler)
	stockfish.uci()

	while not board.is_game_over():
		stockfish.position(board)
		stock_move = stockfish.go(movetime=250) #Set stockfish to 5 moves per second

		#Generate position
		turn = board.turn
		k_castle = board.has_kingside_castling_rights(turn)
		q_castle = board.has_queenside_castling_rights(turn)
		kb_castle = board.has_kingside_castling_rights(not turn)
		qb_castle = board.has_queenside_castling_rights(not turn)

		check = board.is_check()

		position = conversion(str(board), turn, k_castle, q_castle, check, kb_castle, qb_castle)#Append position to training data
		mate = info_handler.info["score"][1].mate
		#Acquire lock so that all data is appended to our lists in the correct order!
		lock.acquire()
		positions.append(position)
		moves.append(one_hot(stock_move[0])) #Add stockfish's choice to training data
		if mate is None:
			value = np.tanh(info_handler.info["score"][1].cp/300)
			values.append(value)
		else:
			values.append(mate/np.abs(mate))
		lock.release()

		board.push(stock_move[0])

#This is our simple ep-Explore algorithm
if __name__ == "__main__":
	#Import statements
	import chess
	import chess.uci

	import keras
	import numpy as np
	import tensorflow as tf

	import os
	from tqdm import tqdm
	from multiprocessing import Process, Lock, Manager

	import model
	import helper

	#Remove tensorflow initialization messages
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	tf.logging.set_verbosity(tf.logging.ERROR)

	with tf.Session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
		pass

	#Create Agent
	agent = model.Agent(6, 64, 0.001, "epAgent") #Model Architecture set here-- (blocks, filters/block, l2reg)

	#Load in from previous run
	constant = 0 #Set this constant to the iteration you want to start on (default for brand new run)
	"""
	agent.model = helper.load_model("bin/model.json", "bin/modelCycle30.h5")
	agent.running_model = helper.load_model("bin/model.json", "bin/modelRunAv30.h5")
	agent.elo = 720
	agent.train_count = constant
	"""

	#Create chess board and initialize stockfish
	board = chess.Board()
	stockfish = chess.uci.popen_engine("stockfish")
	info_handler = chess.uci.InfoHandler()
	stockfish.info_handlers.append(info_handler)
	stockfish.uci()

	#Initialize multiprocessing tools
	lock = Lock()
	manager = Manager()

	positions = manager.list()
	moves = manager.list()
	values = manager.list()

	#Training hyperparameters
	cycles = 20
	num_processes = 20000

	val_pos = None
	val_move = None
	val_val = None


	#Training Loop
	for cycle_num in tqdm(range(cycles)):
		processes = [] #Create a list for all the processes that we will be running
		board.reset()

		#Begin simulating games
		print("Simulating Games")
		for _ in tqdm(range(num_processes)):
			if board.is_game_over():
				board.reset()

			copy = board.copy()
			p = Process(target=play_from, args=(copy, positions, moves, values, lock))
			p.start()
			processes.append(p)

			agent.ep_move(board)

		#Clean up processes
		print("Cleaning up...")
		for i in tqdm(range(num_processes)):
			processes[i].join()

		#learning rate scheduler
		lr = 0.025 * (np.cos((cycle_num + 1 + constant)/17) + 1)
		#Create numpy training data and clear our multiprocessing lists
		train_pos = positions
		train_move = moves
		train_val = values

		positions = manager.list() #Create multiprocessing supported list
		moves = manager.list()
		values = manager.list()

		train_pos = np.reshape(train_pos, [-1, 8, 8, 12])
		train_move = np.reshape(train_move, [-1, 1858])
		train_val = np.reshape(train_val, [-1, 1])

		agent.train(train_pos, train_move, train_val, lr, input_val=val_pos, output_val=[val_move, val_val])

		if cycle_num == 0: #Create validation set for future iterations
			m = train_pos.shape[0]
			split = int(m/50)
			val_pos = train_pos[:split]
			val_move = train_move[:split]
			val_val = train_val[:split]

		#Clear memory
		train_pos = None
		train_move = None
		train_val = None

		#Track elo change with new model
		print("Elo: ", agent.elo)

		#Save model and running average model
		model_weights = "bin/modelCycle" + str(cycle_num + 1 + constant) + ".h5"
		modelAV_weights = "bin/modelRunAv" + str(cycle_num + 1 + constant) + ".h5"
		model_file = "bin/model.json"

		helper.save_model(agent.model, model_file, model_weights)
		helper.save_model(agent.running_model, model_file, modelAV_weights)