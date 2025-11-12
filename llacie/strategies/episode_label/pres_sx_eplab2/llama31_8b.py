from . import AbstractPresentingSymptomsEpisodeLabelV2Strategy
from ...feature.presenting_sx.llama31_8b import PresentingSxFeatureLlama31Instruct8BStrategy
from ....tasks.episode_label import PresentingSymptomsEpisodeLabelV2Task

class PresentingSymptomsEpisodeLabelV2Llama31Instruct8BStrategy(
        AbstractPresentingSymptomsEpisodeLabelV2Strategy):
    """\
    Converts the ..presenting_sx.llama3_1_8b_vllm note features into episode labels using vocab v2.
    This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    task = PresentingSymptomsEpisodeLabelV2Task
    name = "episode_label.pres_sx_eplab2.llama31_8b"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = PresentingSxFeatureLlama31Instruct8BStrategy(db, config, **options)