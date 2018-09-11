from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import h5py
import json
import numpy as np
import os
import sys
import logging
import time

def predict(train_dir, iteration, in_file, read_roi, out_file, write_roi):

    setup_dir = os.path.dirname(os.path.realpath(__file__))

    # TODO: change to predict graph
    with open(os.path.join(train_dir, 'net_io_names.json'), 'r') as f:
        config = json.load(f)

    raw = ArrayKey('RAW')
    affs = ArrayKey('AFFS')

    # input_voxel_size = Coordinate((360, 36, 36))
    # output_voxel_size = Coordinate((120, 108, 108))
    input_voxel_size = Coordinate((120, 12, 12)) * 3
    output_voxel_size = Coordinate((40, 36, 36)) * 3
    input_shape = (91, 862, 862)
    output_shape = (209, 214, 214)

    input_size = Coordinate(input_shape) * input_voxel_size
    output_size = Coordinate(output_shape) * output_voxel_size
    read_roi *= input_voxel_size
    write_roi *= output_voxel_size

    chunk_request = BatchRequest()
    chunk_request.add(raw, input_size)
    chunk_request.add(affs, output_size)

    raw_source = Hdf5Source(in_file, datasets = {raw: 'volumes/raw'})
    target     = Hdf5Write(dataset_names={affs: 'volumes/prediction/affinities', raw: 'volumes/raw'}, output_filename=out_file)

    logging.warn('input voxel size  %s', input_voxel_size)
    logging.warn('output voxel size %s', output_voxel_size)
    logging.warn('input shape       %s', input_shape)
    logging.warn('output shape      %s', output_shape)
    logging.warn('input size        %s', input_size)
    logging.warn('output size       %s', output_size)
    logging.warn('read roi          %s', read_roi)
    logging.warn('write roi         %s', write_roi)
    

    pipeline = (
        raw_source +
        Pad(raw, size=None) +
        Crop(raw, read_roi) +
        Normalize(raw) +
        IntensityScaleShift(raw, 2,-1) +
        Predict(
            os.path.join(train_dir, 'unet_checkpoint_%d'%iteration),
            inputs={
                config['raw']: raw
            },
            outputs={
                config['affinities']: affs
            },
            array_specs={affs: ArraySpec(voxel_size=output_voxel_size)},
            # TODO: change to predict graph
            graph=os.path.join(train_dir, 'unet_inference.meta')
        ) +
        target +
        PrintProfilingStats(every=10) +
        Scan(chunk_request)
    )

    logging.info("Starting prediction...")
    t0 = time.time()
    with build(pipeline):
        pipeline.request_batch(BatchRequest())
    t1 = time.time()
    logging.info("Prediction finished in %f seconds", t1-t0)

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    iteration = 158000
    # in_file = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A_padded_20160501-2-additional-sections.h5'
    # out_file = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A_padded_20160501-predict-2-additional-sections.h5'
    in_file = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A+_padded_20160601.hdf'
    out_file = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/predictions/sample_A+_padded_20160601-predictions-%d-%d.hdf'
    out_file = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A+_padded_20160601-predictions.hdf'
    train_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/0'

    # in voxels
    read_begin = (0, 0, 0)
    read_end   = (91, 862, 862)
    # read_end   = (200, 3072, 3072)

    # in voxels
    write_begin = (0, 0, 0)
    # only works if larger than 209, 214, 214 -- why?
    write_end   = tuple(2.0 * d for d in (209, 214, 214))
    # write_end   = (600, 1024, 1024)

    # WHY THESE NUMBERS FOR WRITE_END?

    with h5py.File(in_file, 'r') as f:
        if 'offset' in f['volumes/raw'].attrs:
            offset = f['volumes/raw'].attrs['offset']
        else:
            offset = (0.0, 0.0, 0.0)

    with h5py.File(out_file, 'w') as f:
        f.create_dataset('volumes/prediction/affinities', shape=(3,) + write_end, dtype=np.float32)
        f['volumes/prediction/affinities'].attrs['offset'] = (offset[0] - 120, offset[1] + 36, offset[2] + 36)
        f['volumes/prediction/affinities'].attrs['resolution'] = tuple(d * 3.0 for d in (40, 36, 36))

    read_roi = Roi(read_begin, read_end)
    write_roi = Roi(write_begin, write_end)

    predict(
        train_dir,
        iteration,
        in_file,
        read_roi,
        out_file,
        write_roi)
