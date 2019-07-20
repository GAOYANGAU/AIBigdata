import argparse
import os
import numpy as np
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial

from hparams import hparams
from tqdm import tqdm
import audio

def _build_from_path(hparams, input_dirs, mel_dir, wav_dir, n_jobs=12, tqdm=lambda x: x):
	"""
	Preprocesses the speech dataset from a gven input path to given output directories

	Args:
		- hparams: hyper parameters
		- input_dir: input directory that contains the files to prerocess
		- mel_dir: output directory of the preprocessed speech mel-spectrogram dataset
		- linear_dir: output directory of the preprocessed speech linear-spectrogram dataset
		- wav_dir: output directory of the preprocessed speech audio dataset
		- n_jobs: Optional, number of worker process to parallelize across
		- tqdm: Optional, provides a nice progress bar

	Returns:
		- A list of tuple describing the train examples. this should be written to train.txt
	"""

	# We use ProcessPoolExecutor to parallelize across processes, this is just for
	# optimization purposes and it can be omited
	executor = ProcessPoolExecutor(max_workers=n_jobs)
	futures = []
	index = 1
	for input_dir in input_dirs:
		with open(os.path.join(input_dir, 'metadata.csv'), encoding='utf-8') as f:
			for line in f:
				parts = line.strip().split('|')
				basename = parts[0]
				wav_path = os.path.join(input_dir, 'wavs', '{}.wav'.format(basename))
				text = parts[2]
				futures.append(executor.submit(partial(_process_utterance, mel_dir, wav_dir, basename, wav_path, text, hparams)))
				index += 1

	return [future.result() for future in tqdm(futures) if future.result() is not None]


def _process_utterance(mel_dir, wav_dir, index, wav_path, text, hparams):
	"""
	Preprocesses a single utterance wav/text pair

	this writes the mel scale spectogram to disk and return a tuple to write
	to the train.txt file

	Args:
		- mel_dir: the directory to write the mel spectograms into
		- linear_dir: the directory to write the linear spectrograms into
		- wav_dir: the directory to write the preprocessed wav into
		- index: the numeric index to use in the spectogram filename
		- wav_path: path to the audio file containing the speech input
		- text: text spoken in the input audio file
		- hparams: hyper parameters

	Returns:
		- A tuple: (audio_filename, mel_filename, linear_filename, time_steps, mel_frames, linear_frames, text)
	"""
	try:
		# Load the audio as numpy array
		wav = audio.load_wav(wav_path, sr=hparams.sample_rate)
	except FileNotFoundError: #catch missing wav exception
		print('file {} present in csv metadata is not present in wav folder. skipping!'.format(
			wav_path))
		return None

	#Trim lead/trail silences
	if hparams.trim_silence:
		wav = audio.trim_silence(wav, hparams)

	#Pre-emphasize
	preem_wav = audio.preemphasis(wav, hparams.preemphasis, hparams.preemphasize)

	#rescale wav
	if hparams.rescale:
		wav = wav / np.abs(wav).max() * hparams.rescaling_max
		preem_wav = preem_wav / np.abs(preem_wav).max() * hparams.rescaling_max

		#Assert all audio is in [-1, 1]
		if (wav > 1.).any() or (wav < -1.).any():
			raise RuntimeError('wav has invalid value: {}'.format(wav_path))
		if (preem_wav > 1.).any() or (preem_wav < -1.).any():
			raise RuntimeError('wav has invalid value: {}'.format(wav_path))

	#[-1, 1]
	out = wav
	constant_values = 0.
	out_dtype = np.float32

	# Compute the mel scale spectrogram from the wav
	mel_spectrogram = audio.melspectrogram(preem_wav, hparams).astype(np.float32)
	mel_frames = mel_spectrogram.shape[1]

	if mel_frames > hparams.max_mel_frames and hparams.clip_mels_length:
		return None

	#Compute the linear scale spectrogram from the wav
	linear_spectrogram = audio.linearspectrogram(preem_wav, hparams).astype(np.float32)
	linear_frames = linear_spectrogram.shape[1]

	#sanity check
	assert linear_frames == mel_frames

	#Ensure time resolution adjustement between audio and mel-spectrogram
	l_pad, r_pad = audio.librosa_pad_lr(wav, hparams.n_fft, audio.get_hop_size(hparams), hparams.pad_sides)

	#Reflect pad audio signal on the right (Just like it's done in Librosa to avoid frame inconsistency)
	out = np.pad(out, (l_pad, r_pad), mode='constant', constant_values=constant_values)

	assert len(out) >= mel_frames * audio.get_hop_size(hparams)

	#time resolution adjustement
	#ensure length of raw audio is multiple of hop size so that we can use
	#transposed convolution to upsample
	out = out[:mel_frames * audio.get_hop_size(hparams)]
	assert len(out) % audio.get_hop_size(hparams) == 0
	time_steps = len(out)

	# Write the spectrogram and audio to disk
	audio_filename = 'audio-{}.npy'.format(index)
	mel_filename = 'mel-{}.npy'.format(index)
	np.save(os.path.join(wav_dir, audio_filename), out.astype(out_dtype), allow_pickle=False)
	np.save(os.path.join(mel_dir, mel_filename), mel_spectrogram.T, allow_pickle=False)

	# Return a tuple describing this training example
	return (audio_filename, mel_filename, time_steps, mel_frames, text)

def _preprocess(args, input_folders, out_dir, hparams):
	mel_dir = os.path.join(out_dir, 'mels')
	wav_dir = os.path.join(out_dir, 'audio')
	os.makedirs(mel_dir, exist_ok=True)
	os.makedirs(wav_dir, exist_ok=True)
	os.makedirs(out_dir, exist_ok=True)
	metadata = _build_from_path(hparams, input_folders, mel_dir, wav_dir, args.n_jobs, tqdm=tqdm)
	_write_metadata(metadata, out_dir)

def _write_metadata(metadata, out_dir):
	with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
		for m in metadata:
			f.write('|'.join([str(x) for x in m]) + '\n')
	#print(metadata)
	#print(metadata.shape)
	mel_frames = sum([int(m[3]) for m in metadata])
	timesteps = sum([int(m[2]) for m in metadata])
	sr = hparams.sample_rate
	hours = timesteps / sr / 3600
	print('Write {} utterances, {} mel frames, {} audio timesteps, ({:.2f} hours)'.format(
		len(metadata), mel_frames, timesteps, hours))
	print('Max input length (text chars): {}'.format(max(len(m[4]) for m in metadata)))
	print('Max mel frames length: {}'.format(max(int(m[3]) for m in metadata)))
	print('Max audio timesteps length: {}'.format(max(m[2] for m in metadata)))

def main():
	print('initializing preprocessing..')
	parser = argparse.ArgumentParser()
	parser.add_argument('--hparams', default='',
		help='Hyperparameter overrides as a comma-separated list of name=value pairs')
	parser.add_argument('--dataset_dir', default='LJSpeech-1.1')
	parser.add_argument('--output_dir', default='training_data')
	parser.add_argument('--n_jobs', type=int, default=cpu_count())
	args = parser.parse_args()

	print("dataset_dir={}".format(args.dataset_dir))
	print("output_dir={}".format(args.output_dir))
	modified_hp = hparams.parse(args.hparams)

	_preprocess(args, [args.dataset_dir], args.output_dir, modified_hp)



if __name__ == '__main__':
	main()
