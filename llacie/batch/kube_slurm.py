import subprocess as sub
import shlex
import time

from os import path
from textwrap import dedent

from .slurm import SlurmJobManager
from ..utils import chunker, echo_warn, echo_info, echo_err
from ..cache.abstract import AbstractFeatureWorkerCache

class KubeSlurmJobManager(SlurmJobManager):
    """
    Uses slurm in the specific manner expected by ERISXdl, in which a wrapper script is needed for 
    each job to reserve a GPU, deploy a container from Podman onto the node, and then run all 
    commands within the container.

    This is documented at https://rc.partners.org/kb/computational-resources/erisxdl?article=3719

    The wrapper script is at /data/erisxdl/kube-slurm/wrappers/kube-slurm-custom-image-job.sh
    """

    # The smallest walltime limit for all paid queues is 2 hours.
    # At about ~1 sec per ID, a 2000-ID job takes ~40min
    SLURM_IDS_PER_JOB = 2000
    SLURM_PARTITION = "Short,Medium,Long"
    SLURM_VENV = '.venv-xdl'  # <-- relative to the root of the repo
    SLURM_THREADS_PER_JOB = 10
    # Adds a random wait 
    SLURM_MAX_RANDOM_WAIT = 60

    # Configuration that is passed into the wrapper script
    KUBE_IMAGE = 'erisxdl.partners.org/bwh-tide_eris1-g/cuda:vllm-llacie' # Points to a podman image
    KUBE_DATA_VOLUME = '/data/tide'
    KUBE_INIT_TIMEOUT = 600
    KUBE_WRAPPER_SCRIPT = '/data/erisxdl/kube-slurm/wrappers/kube-slurm-custom-image-job.sh'

    # Currently, ERISXdl GPU nodes don't have any outbound networking at all, not even allowing
    # connections to the Postgres database. As a workaround, we use a filesystem-based cache as an 
    # intermediary. This WorkerCache functionality is implemented in the llacie.cache.* modules
    KUBE_WORKERS_HAVE_NETWORKING = False
    WORKER_CACHE_POLL_INTERVAL = 15

    # Quick refresher on common options for Slurm's sbatch and srun:
    # -p = Slurm partition (aka queue)
    # -n, --ntasks = how many processes
    #      (note that the number of cores per process is set in `_create_slurm_job()` with -c)
    # -N = how many nodes (min-max or one number)
    # -t = time limit in HH:MM:SS
    # --mem = total memory limit, suffixed in K/M/G/T
    # -x = exclude certain nodes by name (e.g., if they are misbehaving)
    SLURM_OPTS = "-n 1 -t 2:00:00 --gpus=1 --mem=16G"

    SLURM_JOB_TEMPLATE = dedent("""\
        #!/bin/bash
        ## Set the docker container image to be used in the job runtime.
        export KUBE_IMAGE={kube_image}

        ## Set the script to be run within the specified container - this MUST be a separate script
        export KUBE_SCRIPT={kube_script}
        ## Other vars that the script can use
        ids_file={ids_file}
        cache_file={cache_file}
        venv_activate={venv_activate}
        export KUBE_SCRIPT_VARS="$venv_activate $cache_file $ids_file {run_worker}"

        # Define group briefcase (this will provide the GID for the user at runtime)
        export KUBE_DATA_VOLUME={kube_data_volume}

        # Users can also set the following variable to change the timeout in seconds.
        # It’s 600 by default, but might be useful to change for testing.
        export KUBE_INIT_TIMEOUT={kube_init_timeout}
                                
        # Introduced a random sleep here due to issues mounting /data/tide within the container
        # See ticket #INC4206564 with Richard Kenny. Seems very like a very fragile solution...
        MAX_RANDOM_WAIT={max_random_wait}
        sleep $((RANDOM % MAX_RANDOM_WAIT))

        ## Required wrapper script. This must be included at the end of the job submission script.
        ## This wrapper script mounts /data, and your /PHShome directory into the container 
        ## It is provided and maintained by the ERISXdl admins.
        srun {kube_wrapper_script}
        """)


    def __init__(self, worker_cmd, config=dict(), worker_cache=None):
        super().__init__(worker_cmd, config)
        self.threads = int(config.get("SLURM_THREADS_PER_JOB", self.SLURM_THREADS_PER_JOB))
        self._max_rand_wait = int(config.get("SLURM_MAX_RANDOM_WAIT", self.SLURM_MAX_RANDOM_WAIT))

        self.kube_image = config.get("KUBE_IMAGE", self.KUBE_IMAGE)
        self.kube_data_volume = config.get("KUBE_DATA_VOLUME", self.KUBE_DATA_VOLUME)
        self.kube_init_timeout = int(config.get("KUBE_INIT_TIMEOUT", self.KUBE_INIT_TIMEOUT))
        self.kube_wrapper_script = config.get("KUBE_WRAPPER_SCRIPT", self.KUBE_WRAPPER_SCRIPT)

        # This JobManager does not ever support interactive jobs
        self.slurm_interactive = False
        self.array_job_id = None
        self._done_counter = 0

        # The behavior of this JobManager changes if the workers don't have any network capability
        # In this case, we use a filesystem-based cache
        self.workers_have_networking = config.get("KUBE_WORKERS_HAVE_NETWORKING", 
            self.KUBE_WORKERS_HAVE_NETWORKING)
        self.worker_cache = worker_cache
        if not self.workers_have_networking:
            if not isinstance(self.worker_cache, AbstractFeatureWorkerCache):
                raise RuntimeError("When providing a worker cache to a KubeSlurmJobManager, it "
                    "must inherit from AbstractFeatureWorkerCache")
            if self.worker_cache is None:
                raise RuntimeError("KubeSlurmJobManager requires a worker cache if the workers do "
                    "not have networking enabled!")


    # We override the creation of the sbatch script since it is more complicated with the 
    # introduction of the wrapper script.
    def _create_sbatch_script(self):
        """Creates the shell script that is passed to `sbatch`."""
        slurm_dir = self.slurm_io_dir
        repo_dir = path.dirname(path.dirname(path.dirname(__file__)))

        if self.worker_cache is not None:
            cache_file = self.worker_cache.get_path("$SLURM_ARRAY_JOB_ID", "$SLURM_ARRAY_TASK_ID",
                False)
        else: cache_file = "-"

        return self.job_template.format(
            max_random_wait = self._max_rand_wait,
            venv_activate = self.venv_activate, 
            run_worker = self.worker_cmd,
            ids_file = f"\"{slurm_dir}/job.$SLURM_ARRAY_JOB_ID.$SLURM_ARRAY_TASK_ID.ids\"",
            cache_file = cache_file,
            kube_script = shlex.quote(path.join(repo_dir, 'llacie/batch/sh/kube_slurm.sh')),
            kube_image = shlex.quote(self.kube_image),
            kube_data_volume = shlex.quote(self.kube_data_volume),
            kube_init_timeout = self.kube_init_timeout,
            kube_wrapper_script = shlex.quote(self.kube_wrapper_script)
        )
    

    def _parse_scontrol_job_state(self, stanza):
        lines = stanza.split("\n")
        if len(lines) >= 4:
            pairs = lines[3].strip().split()
            fields = {}
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    fields[key] = value
            if "JobState" in fields: return fields["JobState"] == "RUNNING"
        return None


    def is_running(self):
        if self.array_job_id is None: return None
        cmd = f"scontrol show job {self.array_job_id}"
        proc = sub.Popen(cmd, stdout=sub.PIPE, stderr=sub.PIPE, shell=True, text=True)
        stdout, stderr = proc.communicate()
        if proc.returncode != 0:
            if "slurm_load_jobs error: Invalid job id" in stderr: return False
        else:
            stanzas = stdout.strip().split("\n\n")
            statuses = [self._parse_scontrol_job_state(stanza) for stanza in stanzas]
            if not any(status is None for status in statuses):
                return any(status is True for status in statuses)
        echo_warn("Problem interrogating job status for Slurm job {self.array_job_id}")
        echo_warn(f"stdout:\n{stdout}\nstderr:{stderr}")
        return None
    

    def _sweep_worker_cache(self):
        count = self.worker_cache.sweep_into_upstream_db()
        self._done_counter += count
        if count > 0:
            plural = "s have" if self._done_counter != 1 else " has"
            echo_info(f"{self._done_counter} worker dataset{plural} now been saved "
                f"back into Postgres.")


    def run(self, all_ids):
        """
        Kicks off this job array in Slurm, and if a WorkerCache is needed, monitors the status of 
        the interim files and merges results back into Postgres until the job is completed.
        """
        id_chunks = chunker(all_ids, self.job_size)

        if len(id_chunks) == 0:
            return echo_err("There are no notes to process. Exiting.")
        if self.worker_cache is not None:
            self.worker_cache.cache_note_sections(id_chunks)

        self.array_job_id = self._run_background(id_chunks)

        if self.worker_cache is not None and self.array_job_id is not None:
            echo_info("Watching the worker cache directory for results from the workers")
            self.worker_cache.rename(self.array_job_id)
            
            time.sleep(self.WORKER_CACHE_POLL_INTERVAL)
            while self.is_running() is not False:
                self._sweep_worker_cache()
                time.sleep(self.WORKER_CACHE_POLL_INTERVAL)
            
            self._sweep_worker_cache()
            echo_info("The Slurm job has completed!")