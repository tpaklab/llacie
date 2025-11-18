import os
import re
import time
import subprocess as sub
import shlex

from os import path
from click import echo, secho
from textwrap import dedent

from ..utils import chunker, strtobool, echo_err, echo_info

def eprint(msg):
    echo(msg, err=True)
    secho("", nl=False, err=True, reset=True)

class SlurmJob(object):
    """
    Encapsulates an invocation of srun, managing its input, output, and process state, and
    polling for completion. We have to supervise srun like this if we want to start slurm/srun
    jobs in interactive mode (srun then expects to be able to read and write from pipes to a 
    parent process, and if they close, the job gets terminated.)
    """
    slurm_count = 0
    SLURM_PATH = "srun -v --pty"
    SLURM_SHELL = "/bin/bash"

    def __init__(self, slurm_opts, slurm_script, total_jobs, job_prefix="llama_", autostart=True):
        self.bin = SlurmJob.SLURM_PATH
        self.slurm_opts = slurm_opts
        self.slurm_script = slurm_script
        self.job_prefix = job_prefix

        SlurmJob.slurm_count += 1
        self.proc = None
        self.job_id = SlurmJob.slurm_count
        self.log_prefix = f"[job {self.job_id}/{total_jobs}] "
        self.slurm_job_id = None
        self.slurm_partition = None
        self.slurm_host = None

        self.stdout_lines = []
        self.stderr_lines = []
        self.stderr_log = []

        if autostart: self.start()

    def start(self):
        shell = SlurmJob.SLURM_SHELL
        self.cmd = f"{self.bin} {self.slurm_opts} -J {self.job_prefix}{self.job_id} {shell}"
        eprint(self.cmd)
        self.proc = sub.Popen(self.cmd, stdin = sub.PIPE, stdout = sub.PIPE, stderr = sub.PIPE, 
            shell = True)
        os.set_blocking(self.proc.stdout.fileno(), False)
        os.set_blocking(self.proc.stderr.fileno(), False)
        try: 
            outs, errs = self.proc.communicate(input = self.slurm_script.encode(), timeout = 1)
        except sub.TimeoutExpired as e:
            outs, errs = e.stdout, e.stderr
        self.stdout_lines = re.split(b'(?<=\n)', outs if outs is not None else b'')
        self.stderr_lines = re.split(b'(?<=\n)', errs if errs is not None else b'')
        self.process_lines()
        self.proc.stdin.close()

    def _eprint_submitted_msg(self):
        if self.slurm_partition is None or self.slurm_job_id is None: return
        eprint(f"{self.log_prefix}Submitted to slurm partition <{self.slurm_partition}> "
            f"job id {self.slurm_job_id}")

    def process_line(self, line, from_stderr=False):
        if from_stderr and (m := re.match(r'^srun: partition\s*:\s*(\w+)', line)):
            self.slurm_partition = m[1]
            self._eprint_submitted_msg()
        elif from_stderr and (m := re.match(r'^srun: job (\d+) queued and waiting', line)):
            self.slurm_job_id = int(m[1])
            self._eprint_submitted_msg()
        elif from_stderr and (m := re.match(r'^srun: jobid (\d+)', line)):
            self.slurm_job_id = int(m[1])
            self._eprint_submitted_msg()
        elif from_stderr and (m := re.match(r'^srun: launch/slurm: _task_start: Node (\w+)', line)):
            self.slurm_host = m[1]
            eprint(f"{self.log_prefix}Starting on {self.slurm_host}")
        elif not from_stderr and re.match(r'^srun: ', line) is None:
            eprint(f"{self.log_prefix}{line}")

    def process_lines(self, including_incomplete=False):
        for i, line in enumerate(self.stdout_lines):
            if not line.endswith(b"\n") and not including_incomplete: continue
            self.process_line(line.decode().strip(), False)
            self.stdout_lines.pop(i)
        for i, line in enumerate(self.stderr_lines):
            if not line.endswith(b"\n") and not including_incomplete: continue
            line = line.decode().strip()
            self.process_line(line, True)
            self.stderr_log.append(line)
            self.stderr_lines.pop(i)

    def flush_streams(self):
        while not self.proc.stdout.closed:
            line = self.proc.stdout.readline()
            if len(line) == 0: break
            if len(self.stdout_lines) > 0 and not self.stdout_lines[-1].endswith(b"\n"): 
                self.stdout_lines[-1] += line
            else: self.stdout_lines.append(line)
        while not self.proc.stderr.closed:
            line = self.proc.stderr.readline()
            if len(line) == 0: break
            if len(self.stderr_lines) > 0 and not self.stderr_lines[-1].endswith(b"\n"):
                self.stderr_lines[-1] += line
            else: self.stderr_lines.append(line)
        self.process_lines()
    
    def is_done(self):
        if self.proc.poll() is not None:
            self.flush_streams()
            if self.proc.returncode != 0:
                stderr_log = "\n".join(self.stderr_log)
                eprint(f"{self.log_prefix}Slurm salloc invocation failed.\n"
                    f"---\n{stderr_log}\n---")
                #TODO: figure out why it failed, and if we need to retry this job?
            else:
                eprint(f"{self.log_prefix}Done; success!")
            return True
        return False
    

class SlurmJobManager(object):
    """
    Manages a series of SlurmJob's over a set of IDs. These can be note IDs, episode IDs, or
    patient IDs, but in any of these cases, this Manager will kick off a series of SlurmJob's
    that run `worker_cmd` followed by sets of IDs as additional arguments, with the 
    concurrency and resource requests specified in either the constants within this class
    (defaults) or the configuration object passed in during instantiation.
    """
    SLURM_INTERACTIVE = False
    SLURM_SBATCH_PATH = "sbatch"
    SLURM_IO_DIR = os.path.join(os.environ["HOME"], "slurm")
    SLURM_IDS_PER_JOB = 100
    SLURM_CONCURRENT_JOBS = 1   # <-- Note that this only affects interactive job management
    SLURM_PARTITION = "long,bigmem,normal"
    SLURM_VENV = '.venv'
    SLURM_SBATCH_TIMEOUT = 60

    # Quick refresher on common options for Slurm's sbatch and srun:
    # -p = Slurm partition (aka queue)
    # -n = how many processes
    #      (note that the number of cores per process is set in `_create_slurm_job()` with -c)
    # -N = how many nodes (min-max or one number)
    # -t = time limit in HH:MM:SS
    # --mem = total memory limit, suffixed in K/M/G/T
    # -x = exclude certain nodes by name (e.g., if they are misbehaving)
    SLURM_OPTS = "-n 1 -N 1 -t 24:00:00 --mem=12G -x dn078"
    LLAMA_CPP_THREADS = 16
    SLURM_JOB_TEMPLATE = dedent("""\
        #!/bin/bash
        source ~/.bashrc
        conda activate llacie
        source {venv_activate}
        {run_worker} {ids}
        exit
        """)

    def __init__(self, worker_cmd, config=dict()):
        if not isinstance(worker_cmd, str):
            worker_cmd = " ".join([shlex.quote(str(piece)) for piece in worker_cmd])
        self.worker_cmd = worker_cmd
        repo_dir = path.dirname(path.dirname(path.dirname(__file__)))

        self.slurm_venv = config.get("SLURM_VENV", self.SLURM_VENV)
        self.venv_activate = path.join(repo_dir, f"{self.slurm_venv}/bin/activate")

        self.slurm_interactive = strtobool(config.get("SLURM_INTERACTIVE", self.SLURM_INTERACTIVE))  
        self.slurm_sbatch = config.get("SLURM_SBATCH_PATH", self.SLURM_SBATCH_PATH)
        self.slurm_io_dir = config.get("SLURM_IO_DIR", self.SLURM_IO_DIR)
        self.slurm_opts = config.get("SLURM_OPTS", self.SLURM_OPTS)
        self.slurm_partition = config.get("SLURM_PARTITION", self.SLURM_PARTITION)
        self.job_size = int(config.get("SLURM_IDS_PER_JOB", self.SLURM_IDS_PER_JOB))
        self.concurrent_jobs = int(config.get("SLURM_CONCURRENT_JOBS", self.SLURM_CONCURRENT_JOBS))
        self.job_template = config.get("SLURM_JOB_TEMPLATE", self.SLURM_JOB_TEMPLATE)
        self.threads = int(config.get("SLURM_THREADS_PER_JOB", 
            config.get("LLAMA_CPP_THREADS", self.LLAMA_CPP_THREADS)))

        self._slurm_jobs = []


    def _create_interactive_job(self, ids, total_jobs):
        ids = " ".join([shlex.quote(str(id_)) for id_ in ids])
        slurm_script = self.job_template.format(
            venv_activate = self.venv_activate, 
            run_worker = self.worker_cmd,
            ids = ids
        )
        slurm_opts = f"-c {str(self.threads)} -p {self.slurm_partition} {self.slurm_opts}"
        self._slurm_jobs.append(SlurmJob(slurm_opts, slurm_script, total_jobs, 'llacie_'))


    def _run_interactive(self, id_chunks):
        total_jobs = len(id_chunks)
        while True:
            # reap any jobs that have terminated
            for i, job in enumerate(self._slurm_jobs):
                job.flush_streams()
                if job.is_done(): self._slurm_jobs.pop(i)
            # pause to let things settle
            time.sleep(1.0)
            # sow any new jobs that need to be started
            while len(id_chunks) > 0 and len(self._slurm_jobs) < self.concurrent_jobs:
                ids = id_chunks.pop(0)
                self._create_interactive_job(ids, total_jobs)
            # if there are no jobs running, and no jobs left to start, we are done.
            if len(self._slurm_jobs) == 0: break

    
    def _create_sbatch_script(self):
        """Creates the shell script that is passed to `sbatch` for background slurm jobs."""
        slurm_dir = self.slurm_io_dir

        return self.job_template.format(
            venv_activate = self.venv_activate, 
            run_worker = self.worker_cmd,
            ids = f"$(cat \"{slurm_dir}/job.$SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID.ids\")"
        )


    def _run_background(self, id_chunks):
        """Runs `worker_cmd` on each chunk of ids in `id_chunks` using a Slurm job array."""
        total_jobs = len(id_chunks)
        slurm_dir = self.slurm_io_dir
        os.makedirs(slurm_dir, exist_ok = True)

        slurm_opts = (f"{self.slurm_opts} -J llacie --array=1-{total_jobs} "
            + f"-o \"{slurm_dir}/job.%A.%a.out\" -e \"{slurm_dir}/job.%A.%a.err\" "
            + f"-c {str(self.threads)} -p {self.slurm_partition}")
        slurm_script = self._create_sbatch_script()

        proc = sub.Popen(f"{self.slurm_sbatch} {slurm_opts}", stdin = sub.PIPE, 
            stdout = sub.PIPE, stderr = sub.PIPE, shell = True)
        stdout, stderr = proc.communicate(input = slurm_script.encode(), 
            timeout = self.SLURM_SBATCH_TIMEOUT)
        match = re.match(r'^Submitted batch job (\d+)', stdout.decode().strip())

        if match is None: 
            echo_err(f"Error: unexpected sbatch output: {stdout.decode()}\n{stderr.decode()}")
            return None

        array_job_id = int(match[1])
        for i, ids in enumerate(id_chunks):
            with open(f"{slurm_dir}/job.{array_job_id}.{i + 1}.ids", 'w') as f:
                f.write(" ".join([str(j) for j in ids]))
        echo_info(f"Submitted {total_jobs} jobs under Slurm array job ID {array_job_id}")
        return array_job_id


    def run(self, all_ids):
        id_chunks = chunker(all_ids, self.job_size)
        if len(id_chunks) == 0:
            return echo_err("There are no notes to process. Exiting.")
        if self.slurm_interactive:
            self._run_interactive(id_chunks)
        else:
            self._run_background(id_chunks)