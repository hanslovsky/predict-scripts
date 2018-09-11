from __future__ import print_function
import argparse
from gunpowder import *
from gunpowder.tensorflow import *
import json
import logging
import numpy as np
import os
import sys
import time

_raw_key  = 'RAW'
_affs_key = 'AFFS'

_logger_name = 'predict'
_logger      = logging.getLogger(_logger_name)

def predict(
        train_dir,
        iteration,
        in_file,
        read_roi,
        out_file,
        out_dataset,
        write_roi,
        raw_dataset='volumes/raw',
        affs_dataset='volumes/prediction/affinities',
        net_io_names_json='net_io_names.json',
        unet_inference_meta='unet_inference.meta',
        unet_checkpoint_pattern='unet_checkpoint_%d'):

    with open(os.path.join(setup_dir, 'test_net_config.json'), 'r') as f:
        config = json.load(f)

    raw = ArrayKey(_raw_key)
    affs = ArrayKey(_affs_key)

    input_voxel_size  = Coordinate((120, 12, 12)) * 3
    output_voxel_size = Coordinate((40, 36, 36)) * 3
    input_shape       = (91, 862, 862)
    output_shape      = (209, 214, 214)

    input_size  = Coordinate(input_shape) * input_voxel_size
    output_size = Coordinate(output_shape) * output_voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(affs, output_size)

    # for now, use hdf
    raw_source = Hdf5Source(in_container, datasets = {raw: raw_dataset}, array_specs = {raw: ArraySpec(voxel_size=input_voxel_size)})
    target     = Hdf5Write(dataset_names={affs: affs_dataset, raw: raw_dataset}, output_filename=out_container)

    pipeline = (
        raw_source +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        IntensityScaleShift(raw, 2,-1) +
        Predict(
            os.path.join(setup_dir, unet_checkpoint_pattern%iteration),
            inputs={config['raw']: raw},
            outputs={config['affinities']: affs},
            array_specs={affs: ArraySpec(voxel_size=output_voxel_size)},
            graph=os.path.join(setup_dir, unet_inference_meta)
        ) +
        target +
        PrintProfilingStats(every=10) +
        Scan(chunk_request, num_workers=1)
    )

    _logger.info("Starting prediction...")
    t0 = time.time()
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    t1 = time.time()
    _logger.info("Prediction finished in %f seconds", t1-t0)

if __name__ == "__main__":

    print("Starting prediction...")

    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)
    logging.getLogger('gunpowder.nodes.n5_write').setLevel(logging.DEBUG)
    logging.getLogger('gunpowder.nodes.n5_source').setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--train-dir', required=True)
    parser.add_argument('--read-begin', description='Comma separated numbers', required=True)
    parser.add_argument('--read-size', description='Comma separated numbers', reuqired=True)
    parser.add_argument('--write-begin', description='Comma separated numbers', required=True)
    parser.add_argument('--write-size', description='Comma separated numbers', required=True)
    parser.add_argument('--iteration', required=True)
    parser.add_argument('--in-file', required=True)
    parser.add_argument('--out-file', required=True)
    parser.add_argument('--in-dataset', required=False, default='volumes/raw')
    parser.add_argument('--out-dataset', required=False, default='volumes/prediction/affinities')
    parser.add_argument('--net-io-names', required=False, default='net_io_names.json')
    parser.add_argument('--unet-inference-meta', reuqired=False, default='unet_inference.meta')
    parser.add_argument('--unet-checkpoint-pattern', required=False, default='unet_checkpoint_%d')

    args = parser.parse_args()

    read_roi = Roi(
        tuple(int(b) for b in args.read_begin.split(',')),
        tuple(int(b) for b in args.read_size.split(',')))
    write_roi = Roi(
        tuple(int(b) for b in args.write_begin.split(',')),
        tuple(int(b) for b in args.write_size.split(',')))

    predict(
        args.train_dir,
        args.iteration,
        args.in_file,
        read_roi,
        args.out_file,
        args.out_dataset,
        write_roi,
        raw_dataset=args.in_dataset,
        affs_dataset=args.out_dataset,
        net_io_names_json=args.net_io_names_json,
        unet_inference_meta=args.unet_inference_meta,
        unet_checkpoint_pattern=args.unet_checkpoint_pattern)
