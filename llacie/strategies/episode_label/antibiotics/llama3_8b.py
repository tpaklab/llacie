from . import AbstractAntibioticsStrategy
from ...feature.antibiotics.llama3_8b import AntibioticsSpacyStrategy 
from ....tasks.episode_label import AntibioticsEpisodeLabelTask

class AntibioticsLlama3Instruct8BVllmStrategy(
        AbstractAntibioticsStrategy):
    """\
    Converts the ..presenting_sx.llama3_8b_vllm note features into episode labels using vocab v2.
    This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    name = "episode_label.antibiotics_ep1.llama3_8b"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = AntibioticsSpacyStrategy(db, config, **options)