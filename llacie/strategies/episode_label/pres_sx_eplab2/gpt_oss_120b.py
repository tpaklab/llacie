from . import AbstractPresentingSymptomsEpisodeLabelV2Strategy
from ...feature.presenting_sx.GPT_OSS_120B import PresentingSxFeatureGPTOSS120BStrategy
from ....tasks.episode_label import PresentingSymptomsEpisodeLabelV2Task

class PresentingSymptomsEpisodeLabelV2Llama3Instruct8BVllmStrategy(
        AbstractPresentingSymptomsEpisodeLabelV2Strategy):
    """Gpt oss 120B
    """
    task = PresentingSymptomsEpisodeLabelV2Task
    name = "episode_label.pres_sx_eplab2.gpt_oss_120B"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = PresentingSxFeatureGPTOSS120BStrategy(db, config, **options)