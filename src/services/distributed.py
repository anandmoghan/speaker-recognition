from collections import Counter
from subprocess import Popen, PIPE

import numpy as np

Q_SUB_CMD = 'qsub -V -cwd -b y -l hostname={host} -N {name} -q gpu.q -o {log_file} -e {log_file} {cmd}'


def append_ps_and_workers(cmd, ps, workers):
    ps = list(ps.keys())
    workers = list(workers.keys())
    return '{} --ps {} --workers {}'.format(cmd, ','.join(ps), ','.join(workers))


def assign_nodes(num_ps, num_workers, start_port, total_nodes=5, slots_per_node=2):
    free_slots = gpu_stat(total_nodes=total_nodes, slots_per_node=slots_per_node)
    required_slots = num_ps + num_workers

    if len(free_slots) < required_slots:
        raise Exception('Cannot assign jobs. Available slots are less than {}.'.format(required_slots))

    def assign_dict(index_list):
        return dict([('{}:{}'.format(free_slots[i][0], start_port + free_slots[i][1]), free_slots[i][1]) for i in index_list])

    idx = np.random.choice(len(free_slots), required_slots, replace=False)
    return assign_dict(idx[:num_ps]), assign_dict(idx[num_ps:])


def gpu_stat(total_nodes=5, slots_per_node=2):
    output, _ = Popen("qstat -q gpu.q -u \* | tail -n +3 | awk -F '[ ,@]+' '{print $10}'", stdout=PIPE, shell=True).communicate()
    used_slots = Counter(output.decode("utf-8").split('\n')[:-1])
    free_slots = []
    gpu_ids = []
    for i in range(total_nodes):
        node = 'compute-0-{}.local'.format(i)
        try:
            count = used_slots[node]
        except KeyError:
            count = 0
        free_slots = free_slots + [node] * (slots_per_node - count)
        gpu_ids = gpu_ids + list(range(count, slots_per_node))
    return list(zip(free_slots, gpu_ids))


def make_parameter_servers(ps, cmd, model_tag, log_loc):
    c = 0
    jobs_list = []
    for (address, gpu) in ps.items():
        node = address.split(':')[0]
        jobs_list.append(Q_SUB_CMD.format(host=node, name='ps_{}_{}'.format(c + 1, model_tag),
                                          log_file='{}/{}_ps_{}.log'.format(log_loc, model_tag, c + 1),
                                          cmd='{} --gpu {} --model-tag {} --task-index {} --type ps'
                                          .format(cmd, gpu, model_tag, c)))
        print('Parameter Server {} - {}'.format(c + 1, address))
        c += 1
    return jobs_list


def make_workers(workers, cmd, model_tag, log_loc):
    c = 0
    jobs_list = []
    for (address, gpu) in workers.items():
        node = address.split(':')[0]
        jobs_list.append(Q_SUB_CMD.format(host=node, name='worker_{}_{}'.format(c + 1, model_tag),
                                          log_file='{}/{}_worker_{}.log'.format(log_loc, model_tag, c + 1),
                                          cmd='{} --gpu {} --model-tag {} --task-index {} --type worker'
                                          .format(cmd, gpu, model_tag, c)))
        print('Worker Node {} - {}'.format(c + 1, address))
        c += 1
    return jobs_list


def submit_jobs(jobs_list):
    for job in jobs_list:
        Popen(job, stdout=PIPE, shell=True).communicate()
