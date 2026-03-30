from . import AbstractPresentingSymptomsEpisodeLabelV2Strategy
from ...feature.presenting_sx.mistral import PresentingSXMistralStrategy
from ....tasks.episode_label import PresentingSymptomsEpisodeLabelV2Task

class PresentingSymptomsEpisodeLabelV2Llama31Instruct8BStrategy(
        AbstractPresentingSymptomsEpisodeLabelV2Strategy):
    """\
    Converts the ..presenting_sx.mistral note features into episode labels using vocab v2.
    This vocab is UPDATED since the K08 submission after manual examination of the
    misclassifications.
    """
    task = PresentingSymptomsEpisodeLabelV2Task
    name = "episode_label.pres_sx_eplab2.mistral"
    version = "0.0.1"
    prereq_tasks = []

    BATCH_SIZE = 1000


    def __init__(self, db, config, **options):
        super().__init__(db, config, **options)
        self.feat_strat = PresentingSXMistralStrategy(db, config, **options)