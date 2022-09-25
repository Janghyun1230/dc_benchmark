import os
import sys

sys.path.append('/home/janghyun/slurm_tutorial_v12')
from sbatch_launcher import launch_tasks, srun_gpuless_task


def run_exp():
    PYTHON_CMD = 'python -u evaluator/evaluator.py --aug idc '
    pd_test = {
        '--method': ['idc'],
        '--factor': [1, 2],
        '--ipc': [1, 10, 50],
    }
    pd_test2 = {
        '--method': ['dsa', 'dm', 'random'],
        '--ipc': [1, 10, 50],
    }

    param_dict_list = [pd_test, pd_test2]
    job_name = ['dc_test', 'dc_test']
    for i, param_dict in enumerate(param_dict_list):
        launch_tasks(param_option=1,
                     base_cmd=PYTHON_CMD,
                     param_dict=param_dict,
                     partition='rtx3090',
                     timeout='1-00:00:00',
                     job_name=job_name[i],
                     exclude='kiwi,geoffrey')


if __name__ == '__main__':
    run_exp()
    # run_tensorboard()
