AVERAGE_CHECKPOINTS_CMD = 'python -u distributed/average_ckpt.py'
AVERAGE_INITIALIZATION_CMD = 'python -u distributed/create_and_initialize_graph.py'
AVERAGE_TRAIN_CMD = 'python -u distributed/average_train.py'

DISTRIBUTED_EXTRACT_CMD = 'python -u distributed/extract.py'

LATEST_CHECKPOINT = 'latest_ckpt'
MODEL_CHECKPOINT = 'model.ckpt'
META_GRAPH_PATH = '{}.meta'.format(MODEL_CHECKPOINT)
