# from lawbench_one_shot_gen_002588
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets.nation_standard import LawBenchNERDataset

names = [
    # ["1-1", "article_recitation"],
    # ["1-2", "knowledge_question_answering"],
    # ["2-1", "document_proofreading"],
    # ["2-2", "dispute_focus_identification"],
    # ["2-3", "marital_disputes_identification"],
    # ["2-4", "issue_topic_identification"],
    # ["2-5", "reading_comprehension"],
    ["2-6", "named_entity_recognition"],
    # ["2-7", "opinion_summarization"],
    # ["2-8", "argument_mining"],
    # ["2-9", "event_detection"],
    # ["2-10", "trigger_word_extraction"],
    # ["3-1", "fact_based_article_prediction"],
    # ["3-2", "scene_based_article_prediction"],
    # ["3-3", "charge_prediction"],
    # ["3-4", "prison_term_prediction_wo_article"],
    # ["3-5", "prison_term_prediction_w_article"],
    # ["3-6", "case_analysis"],
    # ["3-7", "criminal_damages_calculation"],
    # ["3-8", "consultation"],
]

ner_datasets = []
index="2-6"

lawbench_reader_cfg = dict(
    input_columns=['instruction', 'question'],
    output_column='answer')

lawbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role="HUMAN", prompt="{instruction}\n{question}"),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=1024)
)

lawbench_eval_cfg = dict(
    evaluator=dict(type='LawBenchEvaluator_' + index.replace('-', '_'))
)

ner_datasets.append(
    dict(
        # abbr='lawbench-' + index + '-' + name + '-1-shot',
        abbr="information_retrieval-entity-recognition",
        type=LawBenchNERDataset,
        path='./data_na/information_retrieval/named_entity_recognition.json',
        reader_cfg=lawbench_reader_cfg,
        infer_cfg=lawbench_infer_cfg,
        eval_cfg=lawbench_eval_cfg
    )
)
