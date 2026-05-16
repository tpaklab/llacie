from . import AbstractAntibioticsStrategy
from ...feature.antibiotics.gpt_oss_120b import AntibioticsGPTOSS120BStrategy
from ....tasks.episode_label import AntibioticsEpisodeLabelTask

class AntibioticsEpisodeLabellingGpt120bStrategy(
        AbstractAntibioticsStrategy):
    """\
    Converts the ..presenting_sx.llama3_8b_vllm note features into episode labels using vocab v2.
    This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    task = AntibioticsEpisodeLabelTask
    name = "episode_label.antibiotics_ep1.gpt_oss_120b"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = AntibioticsGPTOSS120BStrategy(db, config, **options)

        