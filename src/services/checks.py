from os.path import join, exists

from constants.app_constants import SAD_DIR


def check_sad(save_loc, args_list):
    sad_loc = join(save_loc, SAD_DIR)
    fails = 0
    for args in args_list:
        if not exists(join(sad_loc, args[0] + '.sad')):
            fails = fails + 1
    return args_list.shape[0] - fails, fails
