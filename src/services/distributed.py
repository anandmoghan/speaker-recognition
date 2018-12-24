from os.path import join as join_path, exists as file_exists
from collections import Counter

import numpy as np
import time

from constants.app_constants import LOGS_DIR, MODELS_DIR, QUEUE_DELETE_CMD, QUEUE_GPU_USAGE_CMD, QUEUE_JOB_STATUS_CMD, QUEUE_SUBMIT_CMD
from constants.tf_constants import AVERAGE_CHECKPOINTS_CMD, AVERAGE_INITIALIZATION_CMD, AVERAGE_TRAIN_CMD, DISTRIBUTED_EXTRACT_CMD, LATEST_CHECKPOINT, META_GRAPH_PATH
from services.common import make_directory, append_cwd_to_python_path, run_command


def append_ps_and_workers(cmd, ps, workers):
    ps = list(ps.keys())
    workers = list(workers.keys())
    return '{} --ps {} --workers {}'.format(cmd, ','.join(ps), ','.join(workers))


def average_parameters(model_tag, iteration, num_workers, save_loc, compute):
    meta_graph = get_meta_graph(model_tag, save_loc)
    checkpoint_dirs_ = [get_model_path(iteration, model_tag, save_loc, worker_id=w) for w in range(num_workers)]
    final_path = get_model_path(iteration, model_tag, save_loc)

    host_name, gpu = compute if compute is not None else ('*', -1)
    job_name = '{}.average.{}'.format(model_tag, iteration)
    log_path = get_log_path(iteration, model_tag, save_loc, operation='average')
    cmd = '{} --checkpoint-dirs {} --final-path {} --gpu {} --meta-graph {}' \
        .format(AVERAGE_CHECKPOINTS_CMD, ','.join(checkpoint_dirs_), final_path, gpu, meta_graph)
    job = get_gpu_queue_job(host_name, job_name, log_path, cmd)
    submit_and_watch_job(append_cwd_to_python_path(job))


def assign_nodes(num_ps, num_workers, start_port, total_nodes=5, slots_per_node=2):
    free_slots = gpu_stat(total_nodes=total_nodes, slots_per_node=slots_per_node)
    required_slots = num_ps + num_workers

    if len(free_slots) < required_slots:
        raise Exception('Cannot assign jobs. Available slots are less than {}.'.format(required_slots))

    def assign_dict(index_list):
        return dict(
            [('{}:{}'.format(free_slots[i][0], start_port + free_slots[i][1]), free_slots[i][1]) for i in index_list])

    idx = np.random.choice(len(free_slots), required_slots, replace=False)
    return assign_dict(idx[:num_ps]), assign_dict(idx[num_ps:])


def check_past_iteration(current_iteration, model_tag, save_loc):
    model_path = get_model_path(current_iteration - 1, model_tag, save_loc)
    latest_ckpt = join_path(model_path, LATEST_CHECKPOINT)
    return file_exists(latest_ckpt)


def create_and_initialize_graph(model_tag, num_classes, num_features, save_loc, compute=None):
    model_path = get_model_path(0, model_tag, save_loc)

    host_name, gpu = compute if compute is not None else ('*', -1)
    job_name = '{}.initialize'.format(model_tag)
    log_path = get_log_path(0, model_tag, save_loc, operation='initialize')
    cmd = '{} --gpu {} --model-path {} --model-tag {} --num-classes {} --num-features {}' \
        .format(AVERAGE_INITIALIZATION_CMD, gpu, model_path, model_tag, num_classes, num_features)
    job = get_gpu_queue_job(host_name, job_name, log_path, cmd)
    submit_and_watch_job(append_cwd_to_python_path(job))


def delete_jobs(pattern):
    output, _ = run_command(QUEUE_DELETE_CMD.format(pattern))
    return output


def get_gpu_queue_job(host_name, name, log_file, cmd):
    return QUEUE_SUBMIT_CMD.format(hostname='compute-0-{}'.format(host_name), name=name, log_file=log_file, cmd=cmd)


def get_log_path(iteration, model_tag, save_loc, worker_id=None, operation='train'):
    log_loc = join_path(save_loc, '{}/{}'.format(LOGS_DIR, model_tag))
    make_directory(log_loc)
    if worker_id is not None:
        log_path = '{}.{}.{}.log'.format(operation, iteration, worker_id)
    else:
        log_path = '{}.{}.log'.format(operation, iteration)
    return join_path(log_loc, log_path)


def get_model_path(iteration, model_tag, save_loc, worker_id=None):
    model_loc = join_path(save_loc, '{}/{}'.format(MODELS_DIR, model_tag))
    if worker_id is not None:
        model_path = 'iteration_{}_{}'.format(iteration, worker_id)
    else:
        model_path = 'iteration_{}'.format(iteration)
    model_path = join_path(model_loc, model_path)
    make_directory(model_path)
    return model_path


def get_meta_graph(model_tag, save_loc):
    return join_path(save_loc, '{}/{}/iteration_0/{}'.format(MODELS_DIR, model_tag, META_GRAPH_PATH))


def gpu_stat(total_nodes=5, slots_per_node=2):
    output, _ = run_command(QUEUE_GPU_USAGE_CMD)
    used_slots = Counter(output.decode("utf-8").split('\n')[:-1])
    free_slots = []
    gpu_ids = []
    for i in range(total_nodes):
        node = 'compute-0-{}.local'.format(i)
        try:
            count = used_slots[node]
            free_gpu = [-1] * (slots_per_node - count)
        except KeyError:
            count = 0
            free_gpu = list(range(0, slots_per_node))
        free_slots = free_slots + [node] * (slots_per_node - count)
        gpu_ids = gpu_ids + free_gpu
    return list(zip(free_slots, gpu_ids))


def make_parameter_servers(ps, cmd, model_tag, log_loc):
    c = 0
    jobs_list = []
    for (address, gpu) in ps.items():
        node = address.split(':')[0][-1]
        job = get_gpu_queue_job(host_name=node, name='ps_{}_{}'.format(c + 1, model_tag),
                                           log_file='{}/{}_ps_{}.log'.format(log_loc, model_tag, c + 1),
                                           cmd='{} --gpu {} --model-tag {} --task-index {} --type ps'
                                           .format(cmd, gpu, model_tag, c))
        jobs_list.append(append_cwd_to_python_path(job))
        print('Parameter Server {} - {}'.format(c + 1, address))
        c += 1
    return jobs_list


def make_workers(workers, cmd, model_tag, log_loc):
    c = 0
    jobs_list = []
    for (address, gpu) in workers.items():
        node = address.split(':')[0][-1]
        job = get_gpu_queue_job(host_name=node, name='worker_{}_{}'.format(c + 1, model_tag),
                                           log_file='{}/{}_worker_{}.log'.format(log_loc, model_tag, c + 1),
                                           cmd='{} --gpu {} --model-tag {} --task-index {} --type worker'
                                           .format(cmd, gpu, model_tag, c))
        jobs_list.append(append_cwd_to_python_path(job))
        print('Worker Node {} - {}'.format(c + 1, address))
        c += 1
    return jobs_list


def submit_and_watch_job(job):
    watch_job(submit_job(job))


def submit_job(job):
    output, _ = run_command(job)
    return output.split(' ')[2]


def submit_jobs(jobs_list):
    job_ids = []
    for job in jobs_list:
        job_ids.append(submit_job(job))


def submit_extract_worker_job(model_tag, iteration, worker_id, feats_scp, max_chunk_size, save_loc,
                              compute=None):
    model_path = get_model_path(iteration, model_tag, save_loc)

    host_name, gpu = compute if compute is not None else ('*', -1)
    job_name = '{}.extract.{}'.format(model_tag, worker_id)
    log_path = get_log_path(iteration, model_tag, save_loc, worker_id, operation='extract')
    cmd = '{} --feats-scp {} --gpu {} --max-chunk-size {} --model-path {} --model-tag {} --save {} --worker-id {}' \
        .format(DISTRIBUTED_EXTRACT_CMD, feats_scp, gpu, max_chunk_size, model_path, model_tag, save_loc, worker_id)
    job = get_gpu_queue_job(host_name, job_name, log_path, cmd)
    return submit_job(job)


def submit_train_worker_job(model_tag, iteration, worker_id, batch_size, lr, egs_index, num_classes, num_features, save_loc, compute=None):
    meta_graph = get_meta_graph(model_tag, save_loc)

    host_name, gpu = compute if compute is not None else ('*', -1)
    job_name = '{}.{}.{}'.format(model_tag, iteration, worker_id)
    log_path = get_log_path(iteration, model_tag, save_loc, worker_id)
    cmd = '{} --batch-size {} --egs-index {} --gpu {} --iteration {} --lr {} --meta-graph {} --model-tag {}  --num-classes {} --num-features {} --save {} --worker-id {}' \
        .format(AVERAGE_TRAIN_CMD, batch_size, egs_index, gpu, iteration, lr, meta_graph, model_tag, num_classes, num_features, save_loc, worker_id)
    job = get_gpu_queue_job(host_name, job_name, log_path, cmd)
    return submit_job(append_cwd_to_python_path(job))


def watch_job(job_id, wait=3):
    time.sleep(wait)
    completed = False
    while not completed:
        output, _ = run_command(QUEUE_JOB_STATUS_CMD.format(job_id))
        if len(output.split('\n')) < 2:
            completed = True
        time.sleep(3)


def watch_jobs(job_ids):
    time.sleep(3)
    for job_id in job_ids:
        watch_job(job_id, wait=0)
