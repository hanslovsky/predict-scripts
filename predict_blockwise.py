from __future__ import print_function
from gunpowder import *
from gunpowder.tensorflow import *
import daisy
import h5py
import json
import numpy as np
import os
import sys
import logging
import time


_logger_name = 'predict.blockwise'
_logger      = logging.getLogger(_logger_name)
logging.getLogger(_logger_name).setLevel(logging.DEBUG)

_raw_key  = 'RAW'
_affs_key = 'AFFS'

_HOME = os.path.expanduser('~')

def predict_blockwise(
        train_dir,
        iteration,
        in_container,
        input_roi_in_pixels,
        out_container,
        # output_roi_in_pixels,
        num_workers,
        client,
        block_size_in_chunks=(1, 1, 1),
        raw_dataset='volumes/raw',
        affs_dataset='volumes/prediction/affinities',
        net_io_names_json='net_io_names.json',
        unet_inference_meta='unet_inference.meta'):

    setup_dir = os.path.dirname(os.path.realpath(__file__))

    # TODO: change to predict graph
    with open(os.path.join(train_dir, net_io_names_json), 'r') as f:
        config = json.load(f)

    raw  = ArrayKey(_raw_key)
    affs = ArrayKey(_affs_key)

    raw_source = daisy.open_ds(in_container, raw_dataset)

    # input_voxel_size = Coordinate((360, 36, 36))
    # output_voxel_size = Coordinate((120, 108, 108))
    input_voxel_size  = Coordinate((120, 12, 12)) * 3
    output_voxel_size = Coordinate((40, 36, 36)) * 3
    input_shape       = (91, 862, 862)
    output_shape      = (209, 214, 214)

    net_input_chunk_size, net_output_chunk_size, context = get_chunk_sizes(
        input_shape,
        output_shape,
        input_voxel_size,
        output_voxel_size)

    
    # compute sizes of blocks
    block_output_size = net_output_chunk_size * block_size_in_chunks
    block_input_size  = block_output_size + context + context

    input_roi   = (input_roi_in_pixels * input_voxel_size).grow(context, context)
    # output_roi  = output_roi_in_pixels * output_voxel_size

    block_input_roi  = Roi((0, 0, 0), block_input_size) - context
    block_output_roi = Roi((0, 0, 0), block_output_size)

    _logger.debug('input_voxel_size  %s', input_voxel_size)
    _logger.debug('output_voxel_size %s', output_voxel_size)
    _logger.debug('input shape       %s', input_shape)
    _logger.debug('output shape      %s', output_shape)
    _logger.debug('block_input_size  %s', block_input_size)
    _logger.debug('block_output_size %s', block_output_size)
    _logger.debug('block_input_roi   %s', block_input_roi)
    _logger.debug('block_output_roi  %s', block_output_roi)
    _logger.debug('input_roi         %s', input_roi)
    # _logger.debug('output_roi        %s', output_roi)

    cwd = os.getcwd()

    def predict_in_block(block):

        from distributed import get_worker
        
        read_roi = block.read_roi
        write_roi = block.write_roi
        predict_script = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/predict/predict.py'
        cuda_visible_devices = get_worker().cuda_visible_devices

        name = 'predict-%s-%s' % (write_roi.get_begin(), write_roi.get_size())
        log_file = os.path.join(cwd, '%s.log' % name)
        pythonpath = ':'.join([
            '%s/workspace-pycharm/u-net/gunpowder' % _HOME,
            '%s/workspace-pycharm/u-net/CNNectome' % _HOME,
            '/groups/saalfeld/home/papec/Work/my_projects/z5/bld/python'
            ])
        pythonpath_export_str = 'export PYTHONPATH=%s:$PYTHONPATH' % pythonpath

        daisy.call([
            'nvidia-docker', 'run', '--rm',
            '-u', os.getuid(),
            '-v', '/groups/turaga:/groups/turaga:rshared',
            '-v', '/groups/saalfeld:/groups/saalfeld:rshared',
            '-v', '/nrs/saalfeld:/nrs/saalfeld:rshared',
            '-w', cwd,
            '--name', name,
            'neptunes5thmoon/gunpowder:v0.3-pre6-dask1'
            '/bin/bash', '-c', '"export CUDA_VISIBLE_DEVICES=%s; %s; python -u %s 2>&1 > %s"' % (
                cuda_visible_devices,
                pythonpath_export_str,
                predict_script,
                log_file
            )
            ])

    def check_block(block):
        _logger.debug("Checking if block %s is complete...", block.write_roi)
        ds = daisy.open_ds(out_container, affs_dataset)
        center_values = ds[block.write_roi.get_center()]
        s = np.sum(center_values)
        _logger.debug("Sum of center values in %s is %f", block.write_roi, s)
        return s != 0

    # TODO set client
    daisy.run_blockwise(
        input_roi,
        block_input_roi,
        block_output_roi,
        process_function=predict_in_block,
        check_function=check_block,
        num_workers=num_workers,
        processes=False,
        read_write_conflict=False,
        client=client
        )

# source/target_chunk_size in pixels
def get_chunk_sizes(input_chunk_size, output_chunk_size, input_voxel_size, output_voxel_size):
    net_input_size  = Coordinate(input_chunk_size) * input_voxel_size
    net_output_size = Coordinate(output_chunk_size) * output_voxel_size
    context         = (net_input_size - net_output_size) / 2
    return net_input_size, net_output_size, context

def startup_dask_server(gpus, scheduler_port=8786, ip='127.0.0.1', diagnostics_port=8787):

    if len(gpus) == 0:
        raise Exception('Expected at least 1 gpu but got: %s', gpus)

    from distributed import Client, LocalCluster
    cluster=LocalCluster(
        n_workers=len(gpus),
        threads_per_worker=1,
        scheduler_port=scheduler_port,
        ip=ip,
        diagnostics_port=diagnostics_port)

    for gpu, worker in zip(gpus, cluster.workers):
        worker.cuda_visible_devices = gpu
    
    # from distributed import Scheduler, Worker, Client
    # from tornado.ioloop import IOLoop
    # from threading import Thread
    
    # loop = IOLoop.current()
    # t = Thread(target=loop.start, daemon=True)
    # t.start()
    # s = Scheduler(loop=loop)
    # s.start('tcp://:8786')

    # for gpu in gpus:
    #     t = Thread(target=loop.start, daemon=True)
    #     t.start()

    #     w = Worker('tcp://127.0.0.1:8786', loop=loop)
    #     w.start()  # choose randomly assigned port
    #     w.cuda_visible_devices = gpu

    client = Client(cluster)
    return cluster, client

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=list, required=True)

    args = parser.parse_args()

    cluster, client = startup_dask_server(args.gpus)
    num_workers     = len(args.gpus)

    logging.basicConfig(level=logging.INFO)
    logging.getLogger('gunpowder.nodes.hdf5like_write_base').setLevel(logging.DEBUG)

    iteration = 158000
    # in_container = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A_padded_20160501-2-additional-sections.h5'
    # out_container = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A_padded_20160501-predict-2-additional-sections.h5'
    in_container = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A+_padded_20160601.hdf'
    out_container = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/predictions/sample_A+_padded_20160601-predictions-%d-%d.hdf'
    out_container = '/groups/saalfeld/home/hanslovskyp/data/cremi/training/sample_A+_padded_20160601-predictions.hdf'
    train_dir = '/groups/saalfeld/home/hanslovskyp/experiments/quasi-isotropic/0'

    # in voxels
    read_begin = (0, 0, 0)
    read_end   = (91, 862, 862)
    # read_end   = (200, 3072, 3072)

    # in voxels
    write_begin = (0, 0, 0)
    # only works if larger than 209, 214, 214 -- why?
    # write_end   = tuple(2.0 * d for d in (209, 214, 214))
    # write_end   = (209, 214, 214)
    write_end   = (600, 1024, 1024)
    # write_end   = (600, 1024, 1024)

    # WHY THESE NUMBERS FOR WRITE_END?

    with h5py.File(in_container, 'r') as f:
        if 'offset' in f['volumes/raw'].attrs:
            offset = f['volumes/raw'].attrs['offset']
        else:
            offset = (0.0, 0.0, 0.0)

    with h5py.File(out_container, 'w') as f:
        f.create_dataset('volumes/prediction/affinities', shape=(3,) + write_end, dtype=np.float32)
        f['volumes/prediction/affinities'].attrs['offset'] = (offset[0] - 120, offset[1] + 36, offset[2] + 36)
        f['volumes/prediction/affinities'].attrs['resolution'] = tuple(d * 3.0 for d in (40, 36, 36))

    read_roi = Roi(read_begin, read_end)
    write_roi = Roi(write_begin, write_end)

    predict_blockwise(
        train_dir,
        iteration,
        in_container,
        read_roi,
        out_container,
        # write_roi,
        num_workers=num_workers,
        client=client,
        block_size_in_chunks=(1, 1, 1))

    
        # train_dir,
        # iteration,
        # in_container,
        # input_roi_in_pixels,
        # out_container,
        # output_roi_in_pixels,
        # num_workers,
        # client,
