import os
import re
import time
import subprocess as sub
import shlex

from os import path
from click import echo, secho
from textwrap import dedent

from ..utils import chunker

def eprint(msg):
    echo(msg, err=True)
    secho("", nl=False, err=True, reset=True)

class BsubJob(object):
    """
    Encapsulates an invocation of bsub, managing its input, output, and process state, and
    polling for completion. We have to supervise bsub like this if we want to start bsub
    jobs in interactive mode (bsub expects to be able to read and write from pipes to a 
    parent process, and if they close, the job gets terminated.)
    """
    bsub_count = 0
    BSUB_PATH = "bsub"

    def __init__(self, bsub_opts, bsub_script, total_jobs, job_prefix="llama_", autostart=True):
        self.bin = BsubJob.BSUB_PATH
        self.bsub_opts = bsub_opts
        self.bsub_script = bsub_script
        self.job_prefix = job_prefix

        BsubJob.bsub_count += 1
        self.proc = None
        self.job_id = BsubJob.bsub_count
        self.log_prefix = f"[job {self.job_id}/{total_jobs}] "
        self.lsf_job_id = None
        self.lsf_queue = None
        self.lsf_host = None

        if autostart: self.start()

    def start(self):
        self.cmd = f"{self.bin} {self.bsub_opts} -J {self.job_prefix}{self.job_id}"
        self.proc = sub.Popen(self.cmd, stdin = sub.PIPE, stdout = sub.PIPE, stderr = sub.PIPE, 
            shell = True)
        os.set_blocking(self.proc.stdout.fileno(), False)
        os.set_blocking(self.proc.stderr.fileno(), False)
        try: 
            outs, errs = self.proc.communicate(input = self.bsub_script.encode(), timeout = 1)
        except sub.TimeoutExpired as e:
            outs, errs = e.stdout, e.stderr
        self.stdout_lines = re.split(b'(?<=\n)', outs if outs is not None else b'')
        self.stderr_lines = re.split(b'(?<=\n)', errs if errs is not None else b'')
        self.process_lines()
        self.proc.stdin.close()
    
    def process_line(self, line, from_stderr=False):
        if (m := re.match(r'Job <(\d+)> is submitted to queue <(\w+)>', line)) and not from_stderr:
            self.lsf_job_id = int(m[1])
            self.lsf_queue = m[2]
            eprint(f"{self.log_prefix}Submitted to <{self.lsf_queue}>, id {self.lsf_job_id}")
        elif (m := re.match(r'<<Starting on (\w+)>>', line)) and from_stderr:
            self.lsf_host = m[1]
            eprint(f"{self.log_prefix}Starting on {self.lsf_host}")
        elif not from_stderr:
            eprint(f"{self.log_prefix}{line}")

    def process_lines(self, including_incomplete=False):
        for i, line in enumerate(self.stdout_lines):
            if not line.endswith(b"\n") and not including_incomplete: continue
            self.process_line(line.decode().strip(), False)
            self.stdout_lines.pop(i)
        for i, line in enumerate(self.stderr_lines):
            if not line.endswith(b"\n") and not including_incomplete: continue
            self.process_line(line.decode().strip(), True)
            self.stderr_lines.pop(i)

    def flush_streams(self):
        while True:
            line = self.proc.stdout.readline()
            if len(line) == 0: break
            if len(self.stdout_lines) > 0 and not self.stdout_lines[-1].endswith(b"\n"): 
                self.stdout_lines[-1] += line
            else: self.stdout_lines.append(line)
        while True:
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
                eprint(f"{self.log_prefix}Bsub invocation failed.\n---\n{self.bsub_script}\n---")
                #TODO: figure out why it failed, and if we need to retry this job?
            else:
                eprint(f"{self.log_prefix}Done; success!")
            return True
        return False
    

class BsubJobManager(object):
    """
    Manages a series of BsubJob's over a set of IDs. These can be note IDs, episode IDs, or
    patient IDs, but in any of these cases, this Manager will kick off a series of BsubJob's
    that run `worker_cmd` followed by sets of IDs as additional arguments, with the 
    concurrency and resource requests specified in either the constants within this class
    (defaults) or the configuration object passed in during instantiation.
    """
    BSUB_IDS_PER_JOB = 50
    BSUB_CONCURRENT_JOBS = 10
    BSUB_PATH = BsubJob.BSUB_PATH
    BSUB_OPTS = "-Is -q interactive"
    LLAMA_CPP_THREADS = 16
    BSUB_LSF_JOB = dedent("""\
        #!/bin/bash
        #BSUB -W 24:00
        #BSUB -R 'rusage[mem=12000]'
        #BSUB -R 'span[hosts=1]'
        #BSUB -L /bin/bash

        source ~/.bashrc
        conda activate llacie
        source {venv_activate}
        {run_worker} {ids}
        """)

    def __init__(self, worker_cmd, config=dict()):
        if not isinstance(worker_cmd, str):
            worker_cmd = " ".join([shlex.quote(str(piece)) for piece in worker_cmd])
        self.worker_cmd = worker_cmd
        repo_dir = path.dirname(path.dirname(__file__))
        self.venv_activate = path.join(repo_dir, ".venv/bin/activate")
        self.worker_cmd = worker_cmd
        
        self.bsub_path = config.get("BSUB_PATH", self.BSUB_PATH)
        self.bsub_opts = config.get("BSUB_OPTS", self.BSUB_OPTS)
        self.job_size = int(config.get("BSUB_IDS_PER_JOB", self.BSUB_IDS_PER_JOB))
        self.concurrent_jobs = int(config.get("BSUB_CONCURRENT_JOBS", self.BSUB_CONCURRENT_JOBS))
        self.job_template = config.get("BSUB_LSF_JOB", self.BSUB_LSF_JOB)
        self.threads = int(config.get("BSUB_THREADS_PER_JOB", 
            config.get("LLAMA_CPP_THREADS", self.LLAMA_CPP_THREADS)))

        self._bsub_jobs = []


    def _create_bsub_job(self, ids, total_jobs):
        ids = " ".join([shlex.quote(str(id_)) for id_ in ids])
        bsub_script = self.job_template.format(
            venv_activate = self.venv_activate, 
            run_worker = self.worker_cmd,
            ids = ids
        )
        bsub_opts = f"-n {str(self.threads)} {self.bsub_opts}"
        self._bsub_jobs.append(BsubJob(bsub_opts, bsub_script, total_jobs, 'llama_'))


    def run(self, all_ids):
        id_chunks = chunker(all_ids, self.job_size)
        total_jobs = len(id_chunks)
        while True:
            # reap any jobs that have terminated
            for i, job in enumerate(self._bsub_jobs):
                job.flush_streams()
                if job.is_done(): self._bsub_jobs.pop(i)
            # pause to let things settle
            time.sleep(1.0)
            # sow any new jobs that need to be started
            while len(id_chunks) > 0 and len(self._bsub_jobs) < self.concurrent_jobs:
                ids = id_chunks.pop(0)
                self._create_bsub_job(ids, total_jobs)
            # if there are no jobs running, and no jobs left to start, we are done.
            if len(self._bsub_jobs) == 0: break