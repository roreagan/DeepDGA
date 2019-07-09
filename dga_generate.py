from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import tensorflow as tf
from tqdm import tqdm

import dga_model
from dga_reader import load_data

flags = tf.flags

# Must fill these fields
flags.DEFINE_string('load_model',  None,    'filename of the model to load')
flags.DEFINE_string('data_dir',   'dga_data',    'data directory')
flags.DEFINE_integer('num_samples', 100, 'how many words to generate')


# model params
flags.DEFINE_integer('batch_size',          1,   'number of sequences to train on in parallel')
flags.DEFINE_integer('max_word_length',     70,   'maximum word length')
flags.DEFINE_integer('rnn_size',        50,                             'size of LSTM internal state')
flags.DEFINE_integer('highway_layers',  2,                              'number of highway layers')
flags.DEFINE_integer('char_embed_size', 30,                             'dimensionality of character embeddings')
flags.DEFINE_integer('embed_dimension', 32,                             'embedding features dimensions')
flags.DEFINE_string ('kernels',         str([2] * 20 + [3] * 10),            'CNN kernel widths')
flags.DEFINE_string ('kernel_features', str([32] * 30),                      'number of features in the CNN kernel')
flags.DEFINE_integer('rnn_layers',      2,                              'number of layers in the LSTM')
flags.DEFINE_integer('random_dimension',         32,                    'dimension of random numbers input in generator')

FLAGS = flags.FLAGS

def agd_output_to_domain(agd_output, char_vocab):
    """
    Transforms an agd output and a domain name
    """
    domain_name = "".join([char_vocab.token(i) for i in agd_output[0][0]])

    # Take everything prior to the dot
    return domain_name.split('.')[0].split(' ')[0]

def main(_):
    ''' Loads trained model and evaluates it on test split '''

    if FLAGS.load_model is None:
        print('Please specify checkpoint file to load model from')
        return -1

    if not os.path.exists(FLAGS.load_model + '.meta'):
        print('Checkpoint file not found', FLAGS.load_model)
        return -1

    print('Reading the character vocabulary from the train data')
    char_vocab, _, _, max_word_length = load_data(FLAGS.data_dir, 70)

    with tf.Graph().as_default(), tf.Session() as session:

        # tensorflow seed must be inside graph
        tf.set_random_seed(FLAGS.seed)
        np.random.seed(seed=FLAGS.seed)

        print('Initializing the network graph')
        initializer = tf.contrib.layers.xavier_initializer()
        ''' build inference graph '''
        with tf.variable_scope("Model", initializer=initializer):
            m = dga_model.inference_graph(
                    char_vocab_size=char_vocab.size,
                    char_embed_size=FLAGS.char_embed_size,
                    batch_size=FLAGS.batch_size,
                    num_highway_layers=FLAGS.highway_layers,
                    num_rnn_layers=FLAGS.rnn_layers,
                    rnn_size=FLAGS.rnn_size,
                    max_word_length=max_word_length,
                    kernels=eval(FLAGS.kernels),
                    kernel_features=eval(FLAGS.kernel_features),
                    dropout=0,
                    embed_dimension=FLAGS.embed_dimension)

            m.update(dga_model.decoder_graph(
                m.embed_output,
                char_vocab_size=char_vocab.size,
                batch_size=FLAGS.batch_size,
                num_highway_layers=FLAGS.highway_layers,
                num_rnn_layers=FLAGS.rnn_layers,
                rnn_size=FLAGS.rnn_size,
                max_word_length=max_word_length,
                kernels=eval(FLAGS.kernels),
                kernel_features=eval(FLAGS.kernel_features)))



            m.update(dga_model.genearator_layer(batch_size=FLAGS.batch_size,
                                                input_dimension=FLAGS.random_dimension,
                                                max_word_length=max_word_length,
                                                embed_dimension=FLAGS.embed_dimension))




            # we need global step only because we want to read it from the model
            global_step = tf.Variable(0, dtype=tf.int32, name='global_step')


        saver = tf.train.Saver()
        saver.restore(session, FLAGS.load_model)
        print('Loaded model from', FLAGS.load_model, 'saved at global step', global_step.eval())

        output_fname = FLAGS.data_dir + "/output_agd.txt"
        print('Generating output domains and saving them to ', output_fname)
        with open(output_fname, "w") as outfile:
            for i in tqdm(range(FLAGS.num_samples)):
                # Select a psuedo-random seed
                np_random = np.random.RandomState(i)
                pseudo_random_seed = np_random.rand(FLAGS.batch_size, FLAGS.random_dimension)

                # Generater(seed) -> embedding
                domain_embedding = session.run([m.gl_output], {m.gl_input: pseudo_random_seed})

                # Decoder(embedding) -> algorithmically generated domain
                agd_ixs = session.run([m.generated_dga], {m.decoder_input: domain_embedding[0]})
                agd = agd_output_to_domain(agd_ixs, char_vocab)

                # Save result to file
                outfile.write("{}, {}\n".format(i, agd))

        print("Done")


if __name__ == "__main__":
    tf.app.run()
