from os.path import abspath

import argparse as ap

from constants.app_constants import NUM_CLASSES, NUM_EGS, NUM_FEATURES, SAVE_LOC
from services.distributed import create_and_initialize_graph, submit_train_worker_job, watch_jobs, average_parameters, \
    get_meta_graph, check_past_iteration, delete_jobs


def parse_args():
    parser = ap.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=64, help='Batch Size')
    parser.add_argument('--epochs', type=int, default=3, help='Number of Epochs')
    parser.add_argument('--final-lr', type=float, default=0.0001, help='Final learning rate')
    parser.add_argument('--initial-lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--iteration', type=int, default=0, help='Iteration to start')
    parser.add_argument('--model-tag', default='HGRU_TEST', help='Model Tag')
    parser.add_argument('--num-classes', type=int, default=NUM_CLASSES, help='Number of classification labels.')
    parser.add_argument('--num-egs', type=int, default=NUM_EGS, help='Number of Example Files')
    parser.add_argument('--num-features', type=int, default=NUM_FEATURES, help='Dimension of input features.')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of Workers')
    parser.add_argument('--save', default=SAVE_LOC, help='Save Location')
    return parser.parse_args()


def run_distributed(args):
    save_loc = abspath(args.save)

    num_iterations = int(args.epochs * args.num_egs / args.num_workers + 0.5)
    all_egs = list(range(1, args.num_egs + 1)) * args.epochs

    lr_decay = (args.final_lr / args.initial_lr) ** (1 / float(num_iterations))

    if args.iteration == 0:
        print('Creating and Initializing Graph.')
        create_and_initialize_graph(args.model_tag, args.num_classes, args.num_features, save_loc, None)
        meta_graph_path = get_meta_graph(args.model_tag, save_loc)
        print('Created graph at : {}'.format(meta_graph_path))
        args.iteration += 1

    for iteration in range(args.iteration, num_iterations + 1):
        print('Starting Iteration: {}'.format(iteration))
        if not check_past_iteration(iteration, args.model_tag, save_loc):
            raise Exception('Model from iteration {} does not exists.'.format(iteration - 1))
        job_ids = []
        lr = args.initial_lr * (lr_decay ** (iteration - 1))
        for worker_id in range(args.num_workers):
            try:
                egs_index = all_egs[(iteration - 1) * args.num_workers + worker_id]
                job_id = submit_train_worker_job(args.model_tag, iteration, worker_id, args.batch_size, lr, egs_index, args.num_classes, args.num_features, save_loc, None)
                print('Submitted job {} to worker {}.'.format(job_id, worker_id))
                job_ids.append(job_id)
            except IndexError:
                pass
        watch_jobs(job_ids)
        print('Averaging parameters over workers in iteration {}.'.format(iteration))
        average_parameters(args.model_tag, iteration, len(job_ids), save_loc, None)


if __name__ == '__main__':
    args_ = parse_args()
    try:
        run_distributed(args_)
    except KeyboardInterrupt:
        print(delete_jobs('*{}.*'.format(args_.model_tag)))
