import functools
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import tensorflow.compat.v1 as tf

import t5
import seqio
import functools as ft


def get_vocab(spm_path):
    return t5.data.SentencePieceVocabulary(spm_path)


def get_output_features(spm_path):
    vocab = get_vocab(spm_path)
    return {
        "inputs": t5.data.Feature(vocabulary=vocab, add_eos=True),
        "targets": t5.data.Feature(vocabulary=vocab, add_eos=True),
    }


def tsv_dataset_fn(file_paths, split, shuffle_files=False):
    ds = tf.data.TextLineDataset(file_paths[split])
    ds = ds.map(
        functools.partial(
            tf.io.decode_csv,
            record_defaults=["", ""],
            field_delim="\t",
            use_quote_delim=False,
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.map(lambda *ex: dict(zip(["inputs", "targets"], ex)))
    return ds


def register_tsv_datasets(task_name, ds_paths, spm_path):
    example_count = {}
    for split, path in ds_paths.items():
        with tf.io.gfile.GFile(path) as f:
            example_count[split] = len(f.readlines())

    seqio.TaskRegistry.add(
        task_name,
        source=seqio.FunctionDataSource(
            dataset_fn=ft.partial(tsv_dataset_fn, ds_paths),
            splits=list(ds_paths),
            num_input_examples=example_count,
        ),
        preprocessors=[
            seqio.preprocessors.tokenize,
            seqio.CacheDatasetPlaceholder(),
            seqio.preprocessors.append_eos_after_trim,
        ],
        output_features=get_output_features(spm_path),
        metric_fns=[],
    )


def register_datasets(datasets, spm_path):
    for dataset in datasets:
        if "splits" in dataset:
            register_tsv_datasets(dataset.name, dataset.splits, spm_path)
        elif "mixture" in dataset:
            seqio.MixtureRegistry.add(dataset.name, dataset.mixture, default_rate=1.0)
        else:
            raise ValueError("Invalid dataset")
