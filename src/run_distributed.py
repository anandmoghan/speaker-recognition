from os.path import abspath, join as join_path

from constants.app_constants import LOGS_DIR
from services.common import make_directory
from services.distributed import append_ps_and_workers, assign_nodes, make_parameter_servers, make_workers, submit_jobs

import argparse as ap

parser = ap.ArgumentParser()
parser.add_argument('--cmd', default='python -u distributed_train.py', help='Python script command. Eg: python -u train.py --epochs 10')
parser.add_argument('--model-tag', default='XVECTOR', help='Model Tag')
parser.add_argument('--num-ps', type=int, default=1, help='Number of Parameter Servers')
parser.add_argument('--num-workers', type=int, default=4, help='Number of Workers')
parser.add_argument('--save', default='../save', help='Save Location')
parser.add_argument('--start-port', type=int, default=7770, help='Starting port value')

args = parser.parse_args()
save_loc = abspath(args.save)
log_loc = join_path(save_loc, LOGS_DIR)
make_directory(log_loc)

print('Making {} Parameter Server(s) and {} Worker Nodes...'.format(args.num_ps, args.num_workers))
ps, workers = assign_nodes(args.num_ps, args.num_workers, args.start_port)
cmd = append_ps_and_workers(args.cmd, ps, workers)
ps_list = make_parameter_servers(ps, cmd, args.model_tag, log_loc)
workers_list = make_workers(workers, cmd, args.model_tag, log_loc)
jobs_list = ps_list + workers_list
print('Submitting jobs to parameter server(s) and worker nodes...')
submit_jobs(jobs_list)
print('Finished.')
