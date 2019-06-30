import os
import pickle

import logger


def save_cur_iter_dynamics_model(params, saver, sess, itr):
    if params.get("save_variables") and params.get("model_save_dir"):
        save_path = saver.save(sess, os.path.join(params["model_save_dir"], "model-iter{}.ckpt".format(itr)))
        logger.log("Model saved in path {}".format(save_path))


def confirm_restoring_dynamics_model(params):
    return params.get("restore_variables", False) and params.get("model_save_dir", False)


def restore_model(params, saver, sess, itr):
    restore_path = os.path.join(params["model_save_dir"], "model-iter{}.ckpt".format(itr))
    saver.restore(sess, restore_path)
    logger.log("Model restored from {}".format(restore_path))
