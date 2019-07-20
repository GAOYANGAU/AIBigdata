import argparse
import os
from warnings import warn
from time import sleep
import re
import time
import wave
from datetime import datetime
import platform
import numpy as np
import tensorflow as tf

from hparams import hparams, hparams_debug_string
from infolog import log
from tqdm import tqdm
from tacotron.models import Tacotron

import audio

from librosa import effects

from tacotron.utils import plot
from tacotron.utils.text import text_to_sequence

class Synthesizer:
	def load(self, checkpoint_path, hparams, model_name='Tacotron'):
		log('Constructing model: %s' % model_name)
		#Force the batch size to be known in order to use attention masking in batch synthesis
		inputs = tf.placeholder(tf.int32, (None, None), name='inputs')
		input_lengths = tf.placeholder(tf.int32, (None), name='input_lengths')
		targets = tf.placeholder(tf.float32, (None, None, hparams.num_mels), name='mel_targets')
		split_infos = tf.placeholder(tf.int32, shape=(hparams.tacotron_num_gpus, None), name='split_infos')
		with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
			self.model = Tacotron( hparams)

			self.model.initialize(inputs, input_lengths, split_infos=split_infos)

			self.mel_outputs = self.model.tower_mel_outputs
			self.alignments = self.model.tower_alignments
			self.stop_token_prediction = self.model.tower_stop_token_prediction
			self.targets = targets


			self.GLGPU_mel_inputs = tf.placeholder(tf.float32, (None, hparams.num_mels), name='GLGPU_mel_inputs')
			self.GLGPU_mel_outputs = audio.inv_mel_spectrogram_tensorflow(self.GLGPU_mel_inputs, hparams)

		self._hparams = hparams
		#pad input sequences with the <pad_token> 0 ( _ )
		self._pad = 0
		#explicitely setting the padding to a value that doesn't originally exist in the spectogram
		#to avoid any possible conflicts, without affecting the output range of the model too much
		if hparams.symmetric_mels:
			self._target_pad = -hparams.max_abs_value
		else:
			self._target_pad = 0.

		self.inputs = inputs
		self.input_lengths = input_lengths
		self.targets = targets
		self.split_infos = split_infos

		log('Loading checkpoint: %s' % checkpoint_path)
		#Memory allocation on the GPUs as needed
		config = tf.ConfigProto()
		config.gpu_options.allow_growth = True
		config.allow_soft_placement = True

		self.session = tf.Session(config=config)
		self.session.run(tf.global_variables_initializer())

		saver = tf.train.Saver()
		saver.restore(self.session, checkpoint_path)


	def synthesize(self, texts, basenames,  mel_dir, wav_dir, plot_dir, mel_filenames):
		hparams = self._hparams
		cleaner_names = [x.strip() for x in hparams.cleaners.split(',')]
		#[-max, max] or [0,max]
		T2_output_range = (-hparams.max_abs_value, hparams.max_abs_value) if hparams.symmetric_mels else (0, hparams.max_abs_value)

		#Repeat last sample until number of samples is dividable by the number of GPUs (last run scenario)
		while len(texts) % hparams.tacotron_synthesis_batch_size != 0:
			texts.append(texts[-1])
			basenames.append(basenames[-1])
			if mel_filenames is not None:
				mel_filenames.append(mel_filenames[-1])

		assert 0 == len(texts) % self._hparams.tacotron_num_gpus
		seqs = [np.asarray(text_to_sequence(text, cleaner_names)) for text in texts]
		input_lengths = [len(seq) for seq in seqs]
		size_per_device = len(seqs) // self._hparams.tacotron_num_gpus

		#Pad inputs according to each GPU max length
		input_seqs = None
		split_infos = []
		for i in range(self._hparams.tacotron_num_gpus):
			device_input = seqs[size_per_device*i: size_per_device*(i+1)]
			device_input, max_seq_len = self._prepare_inputs(device_input)
			input_seqs = np.concatenate((input_seqs, device_input), axis=1) if input_seqs is not None else device_input
			split_infos.append([max_seq_len, 0, 0, 0])
		feed_dict = {
			self.inputs: input_seqs,
			self.input_lengths: np.asarray(input_lengths, dtype=np.int32),
		}
		feed_dict[self.split_infos] = np.asarray(split_infos, dtype=np.int32)
		mels, alignments, stop_tokens = self.session.run([self.mel_outputs, self.alignments, self.stop_token_prediction], feed_dict=feed_dict)

		#Linearize outputs (n_gpus -> 1D)
		mels = [mel for gpu_mels in mels for mel in gpu_mels]
		alignments = [align for gpu_aligns in alignments for align in gpu_aligns]
		stop_tokens = [token for gpu_token in stop_tokens for token in gpu_token]

		#Natural batch synthesis
		#Get Mel lengths for the entire batch from stop_tokens predictions
		target_lengths = self._get_output_lengths(stop_tokens)

		#Take off the batch wise padding
		mels = [mel[:target_length, :] for mel, target_length in zip(mels, target_lengths)]
		assert len(mels) == len(texts)

		mels = np.clip(mels, T2_output_range[0], T2_output_range[1])

		saved_mels_paths = []
		for i, mel in enumerate(mels):

			# Write the spectrogram to disk
			# Note: outputs mel-spectrogram files and target ones have same names, just different folders
			mel_filename = os.path.join(mel_dir, 'mel-{}.npy'.format(basenames[i]))
			np.save(mel_filename, mel, allow_pickle=False)
			saved_mels_paths.append(mel_filename)

			#save wav (mel -> wav)

			wav = self.session.run(self.GLGPU_mel_outputs, feed_dict={self.GLGPU_mel_inputs: mel})
			wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

			audio.save_wav(wav, os.path.join(wav_dir, 'wav-{}-mel.wav'.format(basenames[i])), sr=hparams.sample_rate)

			#save alignments
			plot.plot_alignment(alignments[i], os.path.join(plot_dir, 'alignment-{}.png'.format(basenames[i])),
				title='{}'.format(texts[i]), split_title=True, max_len=target_lengths[i])

			#save mel spectrogram plot
			plot.plot_spectrogram(mel, os.path.join(plot_dir, 'mel-{}.png'.format(basenames[i])),
				title='{}'.format(texts[i]), split_title=True)

		return saved_mels_paths

	def _prepare_inputs(self, inputs):
		max_len = max([len(x) for x in inputs])
		return np.stack([self._pad_input(x, max_len) for x in inputs]), max_len

	def _pad_input(self, x, length):
		return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=self._pad)

	def _get_output_lengths(self, stop_tokens):
		#Determine each mel length by the stop token predictions. (len = first occurence of 1 in stop_tokens row wise)
		output_lengths = [row.index(1) if 1 in row else len(row) for row in np.round(stop_tokens).tolist()]
		return output_lengths


def _get_sentences(args):
	if args.text_list != '':
		with open(args.text_list, 'rb') as f:
			sentences = list(map(lambda l: l.decode("utf-8")[:-1], f.readlines()))
	else:
		sentences = hparams.sentences
	return sentences

def _run_eval(args, checkpoint_path, output_dir, hparams, sentences):
	mel_dir = os.path.join(output_dir, 'mel')
	wav_dir = os.path.join(output_dir, 'wav')
	plot_dir = os.path.join(output_dir, 'plot')

	#Create output path if it doesn't exist
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)

	log(hparams_debug_string())
	synth = Synthesizer()
	synth.load(checkpoint_path, hparams)

	#Set inputs batch wise
	sentences = [sentences[i: i+hparams.tacotron_synthesis_batch_size] for i in range(0, len(sentences), hparams.tacotron_synthesis_batch_size)]

	log('Starting Synthesis')

	for i, texts in enumerate(tqdm(sentences)):
		basenames = ['batch_{}_sentence_{}'.format(i, j) for j in range(len(texts))]
		synth.synthesize(texts, basenames, mel_dir, wav_dir, plot_dir, None)

	log('synthesized mel spectrograms at {}'.format(mel_dir))
	log('plot mel spectrograms at {}'.format(wav_dir))
	log('synthesized wavs at {}'.format(wav_dir))
	return mel_dir, wav_dir

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument('--input_dir', default='tacotron_log', help='folder to contain inputs sentences/targets')
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--text_list', default='', help='Text file contains list of texts to be synthesized. Valid if mode=eval')
	parser.add_argument('--output_dir', default='taco_output/', help='folder to contain synthesized mel spectrograms')
	args = parser.parse_args()

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
	modified_hp = hparams.parse(args.hparams)
	sentences = _get_sentences(args)

	#try:
	#	checkpoint_path = tf.train.get_checkpoint_state(os.path.join(args.input_dir,'taco_pretrained')).model_checkpoint_path
	#	log('loaded model at {}'.format(checkpoint_path))
	#except:
	#	raise RuntimeError('Failed to load checkpoint at {}'.format(args.checkpoint))
	checkpoint_path = "tacotron_log/taco_pretrained/tacotron_model.ckpt-7500"
	_run_eval(args, checkpoint_path, args.output_dir, modified_hp, sentences)

if __name__ == '__main__':
	main()
