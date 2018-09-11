from os.path import join as join_path, exists

from constants.app_constants import VAD_DIR, EMB_DIR, FEATS_SCP_FILE
from services.kaldi import spaced_file_to_dict


def check_embeddings(save_loc, model_tag, args_list):
    emb_loc = join_path(join_path(save_loc, EMB_DIR), model_tag)
    fails = 0
    for i, args in enumerate(args_list):
        if not exists(join_path(emb_loc, args[0] + '.npy')):
            fails = fails + 1
    return args_list.shape[0] - fails, fails


def check_mfcc(save_loc, args_list):
    feats_scp_dict = spaced_file_to_dict(join_path(save_loc, FEATS_SCP_FILE))
    fails = 0
    for key in args_list[:, 0]:
        try:
            _ = feats_scp_dict[key]
        except KeyError:
            fails = fails + 1
    return args_list.shape[0] - fails, fails


def check_sad(save_loc, args_list):
    sad_loc = join_path(save_loc, VAD_DIR)
    fails = 0
    for args in args_list:
        if not exists(join_path(sad_loc, args[0] + '.sad')):
            fails = fails + 1
    return args_list.shape[0] - fails, fails
