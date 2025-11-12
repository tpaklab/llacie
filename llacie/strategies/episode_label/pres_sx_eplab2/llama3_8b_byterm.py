from . import AbstractPresentingSymptomsEpisodeLabelV2Strategy
from ...feature.presenting_sx.llama3_8b_byterm import PresentingSxFeatureLlama3Instruct8BByTermStrategy
from ....tasks.episode_label import PresentingSymptomsEpisodeLabelV2Task


SECS_IN_24H = 60 * 60 * 24

class PresentingSymptomsEpisodeLabelV2Llama3Instruct8BByTermStrategy(
        AbstractPresentingSymptomsEpisodeLabelV2Strategy):
    """\
    Converts the ..presenting_sx.llama3_8b_vllm_byterm note features into episode labels using vocab v2.
    This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    task = PresentingSymptomsEpisodeLabelV2Task
    name = "episode_label.pres_sx_eplab2.llama3_8b_byterm"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = PresentingSxFeatureLlama3Instruct8BByTermStrategy(db, config, **options)