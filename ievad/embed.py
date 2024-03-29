# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
	# Run a WAV file through the model and print the embeddings. The model
	# checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
	# loaded from vggish_pca_params.npz in the current directory.
	$ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

	# Run a WAV file through the model and also write the embeddings to
	# a TFRecord file. The model checkpoint and PCA parameters are explicitly
	# passed in as well.
	$ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

	# Run a built-in input (a sine wav) through the model and print the
	# embeddings. Associated model files are read from the current directory.
	$ python vggish_inference_demo.py
"""

from __future__ import print_function

import tensorflow.compat.v1 as tf
import pickle
from pathlib import Path
from ievad.vggish import vggish_input
from ievad.vggish import vggish_params
from ievad.vggish import vggish_postprocess
from ievad.vggish import vggish_slim
import yaml
import os
    
with open('ievad/config.yaml', "r") as f:
    config =  yaml.safe_load(f)
    
LOAD_PATH = Path(config['raw_data_path']).joinpath(
            Path(config['preproc']['annots_path']).stem
            )    
if not LOAD_PATH.exists():
    LOAD_PATH = LOAD_PATH.parent

def main():
	"""
	Write the postprocessed embeddings as a SequenceExample, in a similar
	format as the features released in AudioSet. Each row of the batch of
	embeddings corresponds to roughly a second of audio (96 10ms frames), and
	the rows are written as a sequence of bytes-valued features, where each
	feature value contains the 128 bytes of the whitened quantized embedding.
	"""
	directory = LOAD_PATH
	destination = Path(config['pickled_data_path']).joinpath(LOAD_PATH.stem)
	destination.mkdir(exist_ok=True, parents=True)
	wavs = []

	for file in directory.iterdir():
		if not file.suffix in ['.WAV', '.wav', '.aif']:
			continue
		print('embedding file: ', file.stem)

		# Create a list of lists to store embeddings in order
		embeddingsList = []
  
		examples_batch = vggish_input.wavfile_to_examples(file)

		# Prepare a postprocessor to munge the model embeddings.
		pproc = vggish_postprocess.Postprocessor(config["pca_params"])

		with tf.Graph().as_default(), tf.Session() as sess:
			# Define the model in inference mode, load the checkpoint, and
			# locate input and output tensors.
			vggish_slim.define_vggish_slim(training=False)

			vggish_slim.load_vggish_slim_checkpoint(sess, config["checkpoint"])

			features_tensor = sess.graph.get_tensor_by_name(
					vggish_params.INPUT_TENSOR_NAME)

			embedding_tensor = sess.graph.get_tensor_by_name(
					vggish_params.OUTPUT_TENSOR_NAME)

			# Run inference and postprocessing.
			[embedding_batch] = sess.run([embedding_tensor], 
								feed_dict={features_tensor: examples_batch})

			postprocessed_batch = pproc.postprocess(embedding_batch)

			for embedding in postprocessed_batch:
				embeddingsList.append(embedding)

			file_dest = destination.joinpath(file.stem + file.suffix + '.pickle')

			with open(file_dest, 'wb') as f:
				pickle.dump(embeddingsList, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	main()