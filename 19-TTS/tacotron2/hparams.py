import numpy as np
import tensorflow as tf

# Default hyperparameters
hparams = tf.contrib.training.HParams(
	# Comma-separated list of cleaners to run on text prior to training and eval. For non-English
	# text, you may want to use "basic_cleaners" or "transliteration_cleaners".
	cleaners='english_cleaners',

	tacotron_num_gpus = 1, #Determines the number of gpus in use for Tacotron training.
	split_on_cpu = True,
	###########################################################################################################################################

	#Audio
	num_mels = 80, #Number of mel-spectrogram channels and local conditioning dimensionality
	num_freq = 1025, # (= n_fft / 2 + 1) only used when adding linear spectrograms post processing network
	rescale = True, #Whether to rescale audio prior to preprocessing
	rescaling_max = 0.999, #Rescaling value

	#train samples of lengths between 3sec and 14sec are more than enough to make a model capable of generating consistent speech.
	clip_mels_length = True, #For cases of OOM (Not really recommended, only use if facing unsolvable OOM errors, also consider clipping your samples to smaller chunks)
	max_mel_frames = 900,  #Only relevant when clip_mels_length = True, please only use after trying output_per_steps=3 and still getting OOM errors.

	#Mel spectrogram
	n_fft = 2048, #Extra window size is filled with 0 paddings to match this parameter
	hop_size = 275, #For 22050Hz, 275 ~= 12.5 ms (0.0125 * sample_rate)
	win_size = 1100, #For 22050Hz, 1100 ~= 50 ms (If None, win_size = n_fft) (0.05 * sample_rate)
	sample_rate = 22050, #22050 Hz (corresponding to ljspeech dataset) (sox --i <filename>)
	frame_shift_ms = None, #Can replace hop_size parameter. (Recommended: 12.5)
	magnitude_power = 2., #The power of the spectrogram magnitude (1. for energy, 2. for power)

	#M-AILABS (and other datasets) trim params (there parameters are usually correct for any data, but definitely must be tuned for specific speakers)
	trim_silence = True, #Whether to clip silence in Audio (at beginning and end of audio only, not the middle)
	trim_fft_size = 2048, #Trimming window size
	trim_hop_size = 512, #Trimmin hop length
	trim_top_db = 40, #Trimming db difference from reference db (smaller==harder trim.)

	#Mel and Linear spectrograms normalization/scaling and clipping
	signal_normalization = True, #Whether to normalize mel spectrograms to some predefined range (following below parameters)
	allow_clipping_in_normalization = True, #Only relevant if mel_normalization = True
	symmetric_mels = True, #Whether to scale the data to be symmetric around 0. (Also multiplies the output range by 2, faster and cleaner convergence)
	max_abs_value = 4., #max absolute value of data. If symmetric, data will be [-max, max] else [0, max] (Must not be too big to avoid gradient explosion, 
																										  #not too small for fast convergence)
	#Contribution by @begeekmyfriend
	#Spectrogram Pre-Emphasis (Lfilter: Reduce spectrogram noise and helps model certitude levels. Also allows for better G&L phase reconstruction)
	preemphasize = True, #whether to apply filter
	preemphasis = 0.97, #filter coefficient.
	pad_sides = 1,

	#Limits
	min_level_db = -100,
	ref_level_db = 20,
	fmin = 55, #Set this to 55 if your speaker is male! if female, 95 should help taking off noise. (To test depending on dataset. Pitch info: male~[65, 260], female~[100, 525])
	fmax = 7600, #To be increased/reduced depending on data.

	#Griffin Lim
	power = 1.5, #Only used in G&L inversion, usually values between 1.2 and 1.5 are a good choice.
	griffin_lim_iters = 60, #Number of G&L iterations, typically 30 is enough but we use 60 to ensure convergence.
	###########################################################################################################################################

	#Tacotron
	#Model general type
	outputs_per_step = 1, #number of frames to generate at each decoding step (increase to speed up computation and allows for higher batch size, decreases G&L audio quality)
	stop_at_any = True, #Determines whether the decoder should stop when predicting <stop> to any frame or to all of them (True works pretty well)
	batch_norm_position = 'after', #Can be in ('before', 'after'). Determines whether we use batch norm before or after the activation function (relu). Matter for debate.
	clip_outputs = True, #Whether to clip spectrograms to T2_output_range (even in loss computation). ie: Don't penalize model for exceeding output range and bring back to borders.
	lower_bound_decay = 0.1, #Small regularizer for noise synthesis by adding small range of penalty for silence regions. Set to 0 to clip in Tacotron range.

	#Input parameters
	embedding_dim = 512, #dimension of embedding space

	#Encoder parameters
	enc_conv_num_layers = 3, #number of encoder convolutional layers
	enc_conv_kernel_size = (5, ), #size of encoder convolution filters for each layer
	enc_conv_channels = 512, #number of encoder convolutions filters for each layer
	encoder_lstm_units = 256, #number of lstm units for each direction (forward and backward)

	#Attention mechanism
	smoothing = False, #Whether to smooth the attention normalization function
	attention_dim = 128, #dimension of attention space
	attention_filters = 32, #number of attention convolution filters
	attention_kernel = (31, ), #kernel size of attention convolution
	cumulative_weights = True, #Whether to cumulate (sum) all previous attention weights or simply feed previous weights (Recommended: True)

	#Attention synthesis constraints
	#"Monotonic" constraint forces the model to only look at the forwards attention_win_size steps.
	#"Window" allows the model to look at attention_win_size neighbors, both forward and backward steps.
	synthesis_constraint = False,  #Whether to use attention windows constraints in synthesis only (Useful for long utterances synthesis)
	synthesis_constraint_type = 'window', #can be in ('window', 'monotonic'). 
	attention_win_size = 7, #Side of the window. Current step does not count. If mode is window and attention_win_size is not pair, the 1 extra is provided to backward part of the window.

	#Decoder
	prenet_layers = [256, 256], #number of layers and number of units of prenet
	decoder_layers = 2, #number of decoder lstm layers
	decoder_lstm_units = 1024, #number of decoder lstm units on each layer
	max_iters = 10000, #Max decoder steps during inference (Just for safety from infinite loop cases)

	#Residual postnet
	postnet_num_layers = 5, #number of postnet convolutional layers
	postnet_kernel_size = (5, ), #size of postnet convolution filters for each layer
	postnet_channels = 512, #number of postnet convolution filters for each layer

	#CBHG mel->linear postnet
	cbhg_kernels = 8, #All kernel sizes from 1 to cbhg_kernels will be used in the convolution bank of CBHG to act as "K-grams"
	cbhg_conv_channels = 128, #Channels of the convolution bank
	cbhg_pool_size = 2, #pooling size of the CBHG
	cbhg_projection = 256, #projection channels of the CBHG (1st projection, 2nd is automatically set to num_mels)
	cbhg_projection_kernel_size = 3, #kernel_size of the CBHG projections
	cbhg_highwaynet_layers = 4, #Number of HighwayNet layers
	cbhg_highway_units = 128, #Number of units used in HighwayNet fully connected layers
	cbhg_rnn_units = 128, #Number of GRU units used in bidirectional RNN of CBHG block. CBHG output is 2x rnn_units in shape

	#Loss params
	mask_encoder = True, #whether to mask encoder padding while computing attention. Set to True for better prosody but slower convergence.
	mask_decoder = False, #Whether to use loss mask for padded sequences (if False, <stop_token> loss function will not be weighted, else recommended pos_weight = 20)
	cross_entropy_pos_weight = 1, #Use class weights to reduce the stop token classes imbalance (by adding more penalty on False Negatives (FN)) (1 = disabled)
	###########################################################################################################################################

	#Tacotron Training
	#Reproduction seeds
	tacotron_random_seed = 5339, #Determines initial graph and operations (i.e: model) random state for reproducibility
	tacotron_data_random_state = 1234, #random state for train test split repeatability

	#performance parameters
	tacotron_swap_with_cpu = False, #Whether to use cpu as support to gpu for decoder computation (Not recommended: may cause major slowdowns! Only use when critical!)

	#train/test split ratios, mini-batches sizes
	tacotron_batch_size = 64, #number of training samples on each training steps
	#Tacotron Batch synthesis supports ~16x the training batch size (no gradients during testing). 
	#Training Tacotron with unmasked paddings makes it aware of them, which makes synthesis times different from training. We thus recommend masking the encoder.
	tacotron_synthesis_batch_size = 1, #DO NOT MAKE THIS BIGGER THAN 1 IF YOU DIDN'T TRAIN TACOTRON WITH "mask_encoder=True"!!
	tacotron_test_size = 0.05, #% of data to keep as test data, if None, tacotron_test_batches must be not None. (5% is enough to have a good idea about overfit)
	tacotron_test_batches = None, #number of test batches.

	#Learning rate schedule
	tacotron_decay_learning_rate = True, #boolean, determines if the learning rate will follow an exponential decay
	tacotron_start_decay = 40000, #Step at which learning decay starts
	tacotron_decay_steps = 18000, #Determines the learning rate decay slope (UNDER TEST)
	tacotron_decay_rate = 0.5, #learning rate decay rate (UNDER TEST)
	tacotron_initial_learning_rate = 1e-3, #starting learning rate
	tacotron_final_learning_rate = 1e-4, #minimal learning rate

	#Optimization parameters
	tacotron_adam_beta1 = 0.9, #AdamOptimizer beta1 parameter
	tacotron_adam_beta2 = 0.999, #AdamOptimizer beta2 parameter
	tacotron_adam_epsilon = 1e-6, #AdamOptimizer Epsilon parameter

	#Regularization parameters
	tacotron_reg_weight = 1e-6, #regularization weight (for L2 regularization)
	tacotron_scale_regularization = False, #Whether to rescale regularization weight to adapt for outputs range (used when reg_weight is high and biasing the model)
	tacotron_zoneout_rate = 0.1, #zoneout rate for all LSTM cells in the network
	tacotron_dropout_rate = 0.5, #dropout rate for all convolutional layers + prenet
	tacotron_clip_gradients = True, #whether to clip gradients

	#Decoder RNN learning can take be done in one of two ways:
	#	Teacher Forcing: vanilla teacher forcing (usually with ratio = 1). mode='constant'
	#	Scheduled Sampling Scheme: From Teacher-Forcing to sampling from previous outputs is function of global step. (teacher forcing ratio decay) mode='scheduled'
	#The second approach is inspired by:
	#Bengio et al. 2015: Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.
	#Can be found under: https://arxiv.org/pdf/1506.03099.pdf
	tacotron_teacher_forcing_mode = 'constant', #Can be ('constant' or 'scheduled'). 'scheduled' mode applies a cosine teacher forcing ratio decay. (Preference: scheduled)
	tacotron_teacher_forcing_ratio = 1., #Value from [0., 1.], 0.=0%, 1.=100%, determines the % of times we force next decoder inputs, Only relevant if mode='constant'
	tacotron_teacher_forcing_init_ratio = 1., #initial teacher forcing ratio. Relevant if mode='scheduled'
	tacotron_teacher_forcing_final_ratio = 0., #final teacher forcing ratio. (Set None to use alpha instead) Relevant if mode='scheduled'
	tacotron_teacher_forcing_start_decay = 10000, #starting point of teacher forcing ratio decay. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_steps = 40000, #Determines the teacher forcing ratio decay slope. Relevant if mode='scheduled'
	tacotron_teacher_forcing_decay_alpha = None, #teacher forcing ratio decay rate. Defines the final tfr as a ratio of initial tfr. Relevant if mode='scheduled'

	#Speaker adaptation parameters
	tacotron_fine_tuning = False, #Set to True to freeze encoder and only keep training pretrained decoder. Used for speaker adaptation with small data.
	###########################################################################################################################################

	sentences = [
	'Scientists at the CERN laboratory say they have discovered a new particle.',
	'There\'s a way to measure the acute emotional intelligence that has never gone out of style.',
	'President Trump met with other leaders at the Group of 20 conference.',
	'The Senate\'s bill to repeal and replace the Affordable Care Act is now imperiled.',
	'Generative adversarial network or variational auto-encoder.',
	'Basilar membrane and otolaryngology are not auto-correlations.',
	'He has read the whole thing.',
	'He reads books.',
	'He thought it was time to present the present.',
	'Thisss isrealy awhsome.',
	'The big brown fox jumps over the lazy dog.',
	'Did the big brown fox jump over the lazy dog?',
	"Peter Piper picked a peck of pickled peppers. How many pickled peppers did Peter Piper pick?",
	"She sells sea-shells on the sea-shore. The shells she sells are sea-shells I'm sure.",
	"Tajima Airport serves Toyooka.",
	'Thank you so much for your support!',
	],
	
	)

def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values) if name != 'sentences']
	return 'Hyperparameters:\n' + '\n'.join(hp)
