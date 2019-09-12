from typing import Optional
from pathlib import Path


def run_embed(datafile: str,
              model_name: str,
              load_from: Optional[str] = None,
              task_name: Optional[str] = None):

    datapath = Path(datafile)
    if not datapath.exists():
        raise FileNotFoundError(datapath)
    elif datapath.suffix not in ['.fasta', '.tfrecord', '.tfrecords']:
        raise Exception(f"Unknown file type: {datapath.suffix}, must be .fasta or .tfrecord")

    load_path: Optional[Path] = None
    if load_from is not None:
        load_path = Path(load_from)
        if not load_path.exists():
            raise FileNotFoundError(load_path)

    import tensorflow as tf
    import tensorflow.keras.backend as K
    import numpy as np

    from tape.models import ModelBuilder

    sess = tf.InteractiveSession()
    K.set_learning_phase(0)
    embedding_model = ModelBuilder.build_model(model_name)

    if datapath.suffix == '.fasta':
        from Bio import SeqIO
        from tape.data_utils import PFAM_VOCAB
        primary = tf.placeholder(tf.int32, [None, None])
        protein_length = tf.placeholder(tf.int32, [None])
        output = embedding_model({'primary': primary, 'protein_length': protein_length})
        sess.run(tf.global_variables_initializer())
        if load_path is not None:
            embedding_model.load_weights(str(load_path))

        embeddings = []
        for record in SeqIO.parse(str(datapath), 'fasta'):
            int_sequence = np.array([PFAM_VOCAB[aa] for aa in record.seq], ndmin=2)
            encoder_output = sess.run(output['encoder_output'],
                                      feed_dict={primary: int_sequence,
                                                 protein_length: [int_sequence.shape[1]]})
            embeddings.append(encoder_output)
    else:
        import contextlib
        if task_name is not None:
            from tape.tasks import TaskBuilder
            task = TaskBuilder.build_task(task_name)
            deserialization_func = task.deserialization_func
        else:
            from tape.data_utils import deserialize_fasta_sequence
            deserialization_func = deserialize_fasta_sequence

        data = tf.data.TFRecordDataset(str(datapath)).map(deserialization_func)
        data = data.batch(1)
        iterator = data.make_one_shot_iterator()
        batch = iterator.get_next()
        output = embedding_model(batch)
        if load_path is not None:
            embedding_model.load_weights(str(load_path))

        embeddings = []
        with contextlib.suppress(tf.errors.OutOfRangeError):
            while True:
                output_batch = sess.run(output['encoder_output'])
                for encoder_output in output_batch:
                    embeddings.append(encoder_output)

    return embeddings


def main():
    import argparse
    import pickle as pkl

    parser = argparse.ArgumentParser()
    parser.add_argument('datafile', type=str, help='sequences to embed')
    parser.add_argument('model', type=str, help='which model to use')
    parser.add_argument('--load-from', type=str, default=None, help='file from which to load pretrained weights')
    parser.add_argument(
        '--task', default=None,
        help='If running a forward pass through existing task datasets, refer to the task with this flag')
    parser.add_argument('--output', default='outputs.pkl', type=str, help='file to output results to')
    args = parser.parse_args()

    embeddings = run_embed(args.datafile, args.model, args.load_from, args.task)

    with open(args.output, 'wb') as f:
        pkl.dump(embeddings, f)


if __name__ == '__main__':
    main()
