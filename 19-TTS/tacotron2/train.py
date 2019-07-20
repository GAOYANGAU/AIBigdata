import argparse
import os
from time import sleep
from datetime import datetime
import numpy as np
import time
import traceback

import infolog
import tensorflow as tf
from hparams import hparams,hparams_debug_string
from infolog import log
from tacotron.feeder import Feeder
from tacotron.models import Tacotron
import audio
from tacotron.utils import ValueWindow, plot
from tacotron.utils.text import sequence_to_text
log = infolog.log

def _add_train_stats(model, hparams):
	with tf.variable_scope('stats') as scope:
		for i in range(hparams.tacotron_num_gpus):
			tf.summary.histogram('mel_outputs %d' % i, model.tower_mel_outputs[i])
			tf.summary.histogram('mel_targets %d' % i, model.tower_mel_targets[i])
		tf.summary.scalar('before_loss', model.before_loss)
		tf.summary.scalar('after_loss', model.after_loss)
		
		tf.summary.scalar('regularization_loss', model.regularization_loss)
		tf.summary.scalar('stop_token_loss', model.stop_token_loss)
		tf.summary.scalar('loss', model.loss)
		tf.summary.scalar('learning_rate', model.learning_rate) #Control learning rate decay speed
		if hparams.tacotron_teacher_forcing_mode == 'scheduled':
			tf.summary.scalar('teacher_forcing_ratio', model.ratio) #Control teacher forcing ratio decay when mode = 'scheduled'
		gradient_norms = [tf.norm(grad) for grad in model.gradients]
		tf.summary.histogram('gradient_norm', gradient_norms)
		tf.summary.scalar('max_gradient_norm', tf.reduce_max(gradient_norms)) #visualize gradients (in case of explosion)
		return tf.summary.merge_all()

def train(log_dir, args, hparams):
	save_dir = os.path.join(log_dir, 'taco_pretrained')
	plot_dir = os.path.join(log_dir, 'plots')
	wav_dir = os.path.join(log_dir, 'wavs')
	mel_dir = os.path.join(log_dir, 'mel-spectrograms')
	tensorboard_dir = os.path.join(log_dir, 'tacotron_events')

	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(plot_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(tensorboard_dir, exist_ok=True)


	checkpoint_path = os.path.join(save_dir, 'tacotron_model.ckpt')
	input_path = args.input_dir


	log('Checkpoint path: {}'.format(checkpoint_path))
	log('Loading training data from: {}'.format(input_path))
	log(hparams_debug_string())

	#Start by setting a seed for repeatability
	tf.set_random_seed(hparams.tacotron_random_seed)

	#Set up data feeder
	coord = tf.train.Coordinator()
	with tf.variable_scope('datafeeder') as scope:
		feeder = Feeder(coord, input_path, hparams)

	#Set up model:
	global_step = tf.Variable(0, name='global_step', trainable=False)
	with tf.variable_scope('Tacotron_model', reuse=tf.AUTO_REUSE) as scope:
		model = Tacotron(hparams)
		model.initialize(feeder.inputs, feeder.input_lengths, feeder.mel_targets, feeder.token_targets,
			targets_lengths=feeder.targets_lengths, global_step=global_step,
			is_training=True, split_infos=feeder.split_infos)
		model.add_loss()
		model.add_optimizer(global_step)
		stats = _add_train_stats(model, hparams)


	GLGPU_mel_inputs = tf.placeholder(tf.float32, (None, hparams.num_mels), name='GLGPU_mel_inputs')
	GLGPU_mel_outputs = audio.inv_mel_spectrogram_tensorflow(GLGPU_mel_inputs, hparams)

	#Book keeping
	step = 0
	time_window = ValueWindow(100)
	loss_window = ValueWindow(100)
	saver = tf.train.Saver(max_to_keep=20)

	log('Tacotron training set to a maximum of {} steps'.format(args.tacotron_train_steps))

	#Memory allocation on the GPU as needed
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	#Train
	with tf.Session(config=config) as sess:
		try:
			summary_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)

			sess.run(tf.global_variables_initializer())

			#saved model restoring
			if args.restore:
				# Restore saved model if the user requested it, default = True
				try:
					checkpoint_state = tf.train.get_checkpoint_state(save_dir)

					if (checkpoint_state and checkpoint_state.model_checkpoint_path):
						log('Loading checkpoint {}'.format(checkpoint_state.model_checkpoint_path), slack=True)
						saver.restore(sess, checkpoint_state.model_checkpoint_path)

					else:
						log('No model to load at {}'.format(save_dir), slack=True)
						saver.save(sess, checkpoint_path, global_step=global_step)

				except tf.errors.OutOfRangeError as e:
					log('Cannot restore checkpoint: {}'.format(e), slack=True)
			else:
				log('Starting new training!', slack=True)
				saver.save(sess, checkpoint_path, global_step=global_step)

			#initializing feeder
			feeder.start_threads(sess)

			#Training loop
			while not coord.should_stop() and step < args.tacotron_train_steps:
				start_time = time.time()
				step, loss, opt = sess.run([global_step, model.loss, model.optimize])
				time_window.append(time.time() - start_time)
				loss_window.append(loss)
				message = 'Step {:7d} [{:.3f} sec/step, loss={:.5f}, avg_loss={:.5f}]'.format(
					step, time_window.average, loss, loss_window.average)
				log(message, end='\r', slack=(step % args.checkpoint_interval == 0))

				if np.isnan(loss) or loss > 100.:
					log('Loss exploded to {:.5f} at step {}'.format(loss, step))
					raise Exception('Loss exploded')

				if step % args.summary_interval == 0:
					log('\nWriting summary at step: {}'.format(step))
					summary_writer.add_summary(sess.run(stats), step)

				if step % args.checkpoint_interval == 0 or step == args.tacotron_train_steps or step == 300:
					#Save model and current global step
					saver.save(sess, checkpoint_path, global_step=global_step)

					log('\nSaving alignment, Mel-Spectrograms and griffin-lim inverted waveform..')

					input_seq, mel_prediction = sess.run([
						model.tower_inputs[0][0],
						model.tower_mel_outputs[0][0],
						])

					#save predicted mel spectrogram to disk (debug)
					mel_filename = 'mel-prediction-step-{}.npy'.format(step)
					np.save(os.path.join(mel_dir, mel_filename), mel_prediction.T, allow_pickle=False)

					#save griffin lim inverted wav for debug (mel -> wav)

					wav = sess.run(GLGPU_mel_outputs, feed_dict={GLGPU_mel_inputs: mel_prediction})
					wav = audio.inv_preemphasis(wav, hparams.preemphasis, hparams.preemphasize)
					audio.save_wav(wav, os.path.join(wav_dir, 'step-{}-wave-from-mel.wav'.format(step)), sr=hparams.sample_rate)

					log('Input at step {}: {}'.format(step, sequence_to_text(input_seq)))

			log('Tacotron training complete after {} global steps!'.format(args.tacotron_train_steps), slack=True)
			return save_dir

		except Exception as e:
			log('Exiting due to exception: {}'.format(e), slack=True)
			traceback.print_exc()
			coord.request_stop(e)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--input_dir', default='training_data/train.txt', help='folder to contain inputs sentences/targets')
	parser.add_argument('--log_dir', default='tacotron_log', help='folder to contain synthesized mel spectrograms')
	parser.add_argument('--restore', type=bool, default=True, help='Set this to False to do a fresh training')
	parser.add_argument('--checkpoint_interval', type=int, default=2500,
		help='Steps between writing checkpoints')
	parser.add_argument('--eval_interval', type=int, default=5000,
		help='Steps between eval on test data')
	parser.add_argument('--summary_interval', type=int, default=250,
		help='Steps between writing checkpoints')
	parser.add_argument('--tacotron_train_steps', type=int, default=100000, help='total number of tacotron training steps')
	args = parser.parse_args()


	modified_hp = hparams.parse(args.hparams)
	os.makedirs(args.log_dir, exist_ok=True)
	train( args.log_dir, args, modified_hp)


if __name__ == '__main__':
	main()
