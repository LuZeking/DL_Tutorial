import os
import subprocess
from datetime import datetime
import sys
import itertools
# from command_line.cmd_funcs import parse_pickle_file
import yaml
import numpy as np
import argparse



def main():
    '''
    Sumbit a network to train using slurm
    '''
    # define network and training parameters
    first_epoch = 0
    # final_epoch = 99
    final_epoch = 0


    # define slurm parameters
    job_name = "zejin_test_prototype_cifar10_10142021"#'bltvnet_GN_regularize1e-6'
    account = 'KIETZMANN-SL3-GPU'
    # account = 'KIETZMANN-SL2-GPU'
    job_script = 'slurm_submit_job.sh'
    # time = '3:00:00'
    time = '12:00:00'

    # check that paths exist
    assert os.path.exists(job_script)

    start_epoch_per_job = np.arange(first_epoch, final_epoch+1, 1, dtype=int)
    total_epochs = final_epoch - first_epoch + 1

    n_jobs = len(start_epoch_per_job)
    jobid = None
    for job_idx in range(n_jobs):

        # slurm_app = 'source run_thewatch.sh {} & python task_fulltfdata.py'.format(job_idx + first_epoch)  # run_thewatch.sh is used to print stuff about memory consumption
        # slurm_app = 'python hello.py'#.format(job_idx + first_epoch)  # run_thewatch.sh is used to print stuff about memory consumption
        slurm_app = f"python complexity_cal.py" # 1_my_new_1_1.py"
        print(slurm_app)  # 1my_new_224_NxN_1.py
        # assert os.path.exists(slurm_app.split(' ')[5])  # make sure the python script we called exists
        #assert os.path.exists(slurm_app.split(' ')[1])  # make sure the python script we called exists
        assert os.path.exists(slurm_app.split(' ')[1]) 

        # prepare training call for options
        slurm_options = "" # '--start-epoch {0} '.format(job_idx+first_epoch)
        # args_str = " ".join(args)  # list  to str
        # slurm_options += args_str
        # for key,value in config.items():
        #     slurm_options += '--{0} {1} '.format(key, value)

        os.environ['CS_SLURM_APP'] = slurm_app
        os.environ['CS_SLURM_OPTIONS'] = slurm_options

        # cmd to use sbatch to run slurm experiment
        if job_idx > 0:
            sbatch_options = ' --job-name={0} --time={1} --account={2} --depend=afterok:{3} {4}'.format(
                job_name, time, account, jobid, job_script)
        else:
            sbatch_options = ' --job-name={0} --time={1} --account={2} {3}'.format(
                job_name, time, account, job_script)

        cmd = 'sbatch' + sbatch_options

        print('\nJOB: {0}'.format(job_idx))
        print(cmd)
        cmd_split = cmd.split(' ')
        sbatch_output = subprocess.run(cmd_split, stdout=subprocess.PIPE)
        sbatch_output_str = sbatch_output.stdout.decode('UTF-8')
        print(sbatch_output_str+'\n')
        print('SLURM_APP:     ', slurm_app)
        print('SLURM_OPTION: ', slurm_options, '\n')
        # get the job ID, last output from sbatch
        jobid = int(sbatch_output_str.split(' ')[-1])

    print('')
    print('FIRST_EPOCH: ', first_epoch)
    print('FINAL_EPOCH: ', final_epoch)
    print('N_EPOCHS:    ', total_epochs)
    print('N_JOBS:      ', n_jobs)

if __name__ == '__main__':
    main()