#!/usr/bin/env python
# coding: utf-8

# In[10]:


from itertools import product
import math
import numpy as np
import inspect
import os
import time
import subprocess
from IPython.display import Image, display, HTML
import importlib
import socket
import argparse

# working dir
cwd = os.getcwd()

# hostname + cleanup login node 'daint101' etc
hostname = socket.gethostname()
if hostname.startswith('daint'):
    hostname = 'daint'

# name of this script
scriptname = inspect.getframeinfo(inspect.currentframe()).filename
scriptpath = os.path.dirname(os.path.abspath(scriptname))

# summary
print(f'CWD        : {cwd} \nScriptpath : {scriptpath} \nHostname   : {hostname}')


# In[11]:


def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

if is_notebook():
    # this makes the notebook wider on a larger screen using %x of the display
    display(HTML("<style>.container { width:100% !important; }</style>"))
    # save this notebook as a raw python file as well please
    get_ipython().system('jupyter nbconvert --to script generate-ghex.ipynb')


# In[12]:


# ------------------------------------------------------------------
# Command line params
# ------------------------------------------------------------------
def get_command_line_args(notebook_args=None):
    parser = argparse.ArgumentParser(description='Generator for ghex benchmarks')
    parser.add_argument('-d', '--dir', default=cwd, action='store', help='base directory to generate job scripts in')
    if is_notebook():
        parser.add_argument('-f', help='seems to be defaulted by jupyter')
        return parser.parse_args(notebook_args)
    return parser.parse_args()

notebook_args = '--dir /home/biddisco/benchmarking-results/test'.split()
if is_notebook():
    args = get_command_line_args(notebook_args)
else:
    args = get_command_line_args()


# In[13]:


# strings with @xxx@ will be substituted by cmake
binary_dir = "@BIN_DIR@"

if args.dir:
    run_dir = args.dir
else:
    run_dir = "@RUN_DIR@"

print(f'Generating scripts in {run_dir}')


# In[14]:


#
# experimental code to try to generate a sensible number of messages given a message size
#
def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def normalized_sigmoid_fkt(center, width, x):
   '''
   Returns array of a horizontal mirrored normalized sigmoid function
   output between 0 and 1
   '''
   s = 1/(1+np.exp(width*(x-center)))
   return s
   #return 1*(s-min(s))/(max(s)-min(s)) # normalize function to 0-1

def num_messages(vmax, vmin, center, width, x):
    s = 1 / (1 + np.exp(width*(x-center)))
    return vmin + (vmax-vmin)*s #(s-vmin)/(vmax-vmin) # normalize function to 0-1


# In[15]:


cscs = {}

# jb laptop
cscs["oryx2"] = {
  "Cores": 8,
  "Threads per core": 2,
  "Allowed rpns": [1, 2],
  "thread_array": [1,2,4,8],
  "sleeptime":0,
  "Run command": "mpiexec -n {total_ranks} --oversubscribe",
  "Batch preamble": """
#!/bin/bash -l

# Env
export OMP_NUM_THREADS={threads}

# Commands
"""
}

# daint mc nodes config
cscs["daint"] = {
  "Cores": 36,
  "Threads per core": 2,
  "Allowed rpns": [1, 2],
  "thread_array": [1,2,4,8,16,32],
  "sleeptime":0.25,
  "Run command": "srun -n {total_ranks} -c {threads_per_rank}",
  "Batch preamble": """
#!/bin/bash -l
#SBATCH --job-name={run_name}_{nodes}_{size}_{inflight}_{threads}
#SBATCH --time={time_min}
#SBATCH --nodes={nodes}
#SBATCH --partition=normal
#SBATCH --account=csstaff
#SBATCH --constraint=mc
#SBATCH --output=output.txt
#SBATCH --error=error.txt

# Env
export MPICH_MAX_THREAD_SAFETY=multiple
export OMP_NUM_THREADS={threads}
export MKL_NUM_THREADS={threads}

# Debug
module list &> modules.txt
printenv > env.txt

# Commands
"""
}


# In[16]:


#
# Generate Job script preamble
#
def init_job_text(system, run_name, time_min, nodes, threads, inflight, size):
    return system["Batch preamble"].format(run_name=run_name,
                                           time_min=time_min,
                                           nodes=nodes,
                                           size=size,
                                           inflight=inflight,
                                           threads=threads).strip()
#
# create a directory name from params
#
def make_job_directory(fdir,name, transport, nodes, threads, inflight, size):
    return f'{fdir}/{name}_{transport}_{nodes}_{threads}_{inflight}_{size}'

#
# create the launch command-line
#
def run_command(system, total_ranks, cpus_per_rank):
    threads_per_rank = system["Threads per core"] * cpus_per_rank
    return system["Run command"].format(total_ranks=total_ranks, cpus_per_rank=cpus_per_rank, threads_per_rank=threads_per_rank)

#
# create dir + write final script for sbatch/shell or other job launcher
#
def write_job_file(system, launch_file, job_dir, job_text, suffix=''):
    job_path = os.path.expanduser(job_dir)
    os.makedirs(job_path, exist_ok=True)
    job_file = f"{job_path}/job_{suffix}.sh"
    with open(job_file, "w") as f:
        f.write(job_text)

    print(f"Submitting : {job_path} : {job_file}")
    launchstring  = f'sbatch --chdir={job_path} {job_file}\n'
    launchstring += 'sleep ' + str(system['sleeptime']) + '\n'
    launch_file.write(launchstring)

#
# application specific commmands/flags/options that go into the job script
#
def ghex(system, bin_dir, timeout, transport, progs, nodes, threads, msg, size, inflight, extra_flags="", env=""):
    total_ranks = 2
    whole_cmd = ""
    # mpi oes not have suffix, other transport layers use '_libfabric', '_ucx', eetc
    suffix = f'_{transport}' if transport!='mpi' else ''
    for prog in progs:

        # generate the program commmand with command line params
        cmd = f"{bin_dir}/{prog}{suffix} {msg} {size} {inflight}"

        # get the launch command (mpiexec, srun, etc)
        run_cmd = run_command(system, total_ranks, threads)

        # simple version of benchmark
        temp = "\n" + f"{env} timeout {timeout} {run_cmd} {cmd} >> {prog}_{msg}_{size}_{inflight}.out".strip()
        whole_cmd += temp +'\n'

        # for libfabric, run benchmark again with extra environment options to control execution
        if transport=='libfabric':
            lf_env = env + 'LIBFABRIC_AUTO_PROGRESS=1'
            temp = "\n" + f"{lf_env} timeout {timeout} {run_cmd} {cmd} >> {prog}_{msg}_{size}_{inflight}.out".strip()
            whole_cmd += temp +'\n'

            lf_env = env + 'LIBFABRIC_ENDPOINT_TYPE=multiple'
            temp = "\n" + f"{lf_env} timeout {timeout} {run_cmd} {cmd} >> {prog}_{msg}_{size}_{inflight}.out".strip()
            whole_cmd += temp +'\n'

            lf_env = env + 'LIBFABRIC_ENDPOINT_TYPE=threadlocal'
            temp = "\n" + f"{lf_env} timeout {timeout} {run_cmd} {cmd} >> {prog}_{msg}_{size}_{inflight}.out".strip()
            whole_cmd += temp +'\n'

            lf_env = env + 'LIBFABRIC_AUTO_PROGRESS=1 ' + 'LIBFABRIC_ENDPOINT_TYPE=multiple'
            temp = "\n" + f"{lf_env} timeout {timeout} {run_cmd} {cmd} >> {prog}_{msg}_{size}_{inflight}.out".strip()
            whole_cmd += temp +'\n'

            lf_env = env + 'LIBFABRIC_AUTO_PROGRESS=1 ' + 'LIBFABRIC_ENDPOINT_TYPE=threadlocal'
            temp = "\n" + f"{lf_env} timeout {timeout} {run_cmd} {cmd} >> {prog}_{msg}_{size}_{inflight}.out".strip()
            whole_cmd += temp +'\n'

    return whole_cmd


# In[17]:


system = cscs[hostname]
#
job_name       = 'ghex-benchmark'
timeout        = 400        # seconds per benchmark
time_min       = timeout*20 # total time estimate
timestr        = time.strftime('%H:%M:%S', time.gmtime(time_min))
ranks_per_node = 1
nodes_arr = [2]
trans_arr = ['libfabric', 'mpi']
thrd_arr  = system['thread_array']
size_arr  = [1,100,1000,10000,100000,500000,1000000]
nmsg_lut  = {1:500000,
             100:500000,
             1000:500000,
             5000:250000,
             10000:250000,
             50000:250000,
             100000:250000,
             200000:250000,
             500000:100000,
             1000000:50000,
             2000000:25000}

#for i in size_arr:
#    print(int(num_messages(1E6, 25E3, 1E5, 1E-5, i)))

flight_arr= [1,4,64]
prog_arr  = ["ghex_p2p_bi_cb_avail_mt", "ghex_p2p_bi_cb_wait_mt", "ghex_p2p_bi_ft_avail_mt", "ghex_p2p_bi_ft_wait_mt"]


# In[18]:


combos = 0

if run_dir.startswith('@'):
    print(f'Skipping creation of job launch file for {run_dir}')
else:
    job_launch = f"{run_dir}/launch.sh"
    job_launch_file = open(job_launch, "w")

# generate all combinations in one monster loop
for nodes, transport, threads, size, inflight in product(
    nodes_arr, trans_arr, thrd_arr, size_arr, flight_arr):

    extra_flags = ""
    suffix = ""
    # number of messages (niter)
    msg = int(num_messages(1E6, 25E3, 1E5, 1E-5, size))
    msg = nmsg_lut[size]
    # create the output directory for each job
    job_dir = make_job_directory(run_dir, 'ghex', transport, nodes, threads, inflight, size)

    # first part of boiler plate job script
    job_text = init_job_text(system, job_name, timestr, nodes, threads, size, msg)

    # application specific part of job script
    job_text += ghex(
        system,
        binary_dir,
        timeout,
        transport,
        prog_arr,
        nodes,
        threads,
        msg,
        size,
        inflight,
        suffix,
        extra_flags,
    )
    # debugging
    # print(job_dir, '\n', job_text, '\n\n\n\n')

    combos += 1

    if combos==1:
        print('Uncommment the following line to perform the job creation')
    # uncomment this to create job scripts
    write_job_file(system, job_launch_file, job_dir, job_text)

print(combos)


# In[ ]:




