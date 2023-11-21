from mmengine.config import read_base
from opencompass.models import AI360GPT
from opencompass.partitioners import NaivePartitioner
from opencompass.runners.local_api import LocalAPIRunner
from opencompass.tasks import OpenICLInferTask


with read_base():
    from .summarizers.national_standard import summarizer
    # Coding
    from .datasets.national_standard.coding_generation import humaneval_datasets
    # Information Retrieval
    from .datasets.national_standard.information_retrieval_entity_recognition import ner_datasets
    # Knowledge
    from .datasets.national_standard.knowledge_ceval import ceval_datasets
    # Reasoning
    from .datasets.national_standard.reasoning_common import reasoning_common_datasets
    from .datasets.national_standard.reasoning_math import reasoning_math_datasets
    # Text Analysis
    from .datasets.national_standard.text_analysis_text_classification import textcls_datasets
    from .datasets.national_standard.text_analysis_sentiment_analysis import sentimentanalysis_datasets
    from .datasets.national_standard.text_analysis_intention_recognition import intentionrecognition_datasets
    # Translation
    from .datasets.national_standard.translation_en2cn import translation_datasets

datasets = [
    *humaneval_datasets,
    *ner_datasets,
    *ceval_datasets,
    *reasoning_common_datasets,
    *reasoning_math_datasets,
    *textcls_datasets,
    *sentimentanalysis_datasets,
    *intentionrecognition_datasets,
    *translation_datasets
]


models = [
    dict(
        abbr='360GPT_S2_V9',
        type=AI360GPT,
        path='360GPT_S2_V9',
        key = "xxxxxxxxxxxxxx",
        query_per_second=1,
        max_out_len=2048,
        max_seq_len=2048,
        batch_size=8),
]

infer = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalAPIRunner,
        max_num_workers=4,
        concurrent_users=4,
        task=dict(type=OpenICLInferTask)),
)

# work_dir ="./output/360GPT_S2_V9"

work_dir = "outputs/national_standard"
