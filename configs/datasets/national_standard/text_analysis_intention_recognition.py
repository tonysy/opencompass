# from CLUE_C3_gen_8c358f
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets import C3Dataset_V2
from opencompass.utils.text_postprocessors import first_capital_postprocess

C3_reader_cfg = dict(
    input_columns=[
        "question",
        "content",
        "choice0",
        "choice1",
        "choice2",
        "choice3",
        "choices",
    ],
    output_column="label",
)

C3_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[
            dict(
                role="HUMAN",
                prompt=
                "{content}\n问：{question}\nA. {choice0}\nB. {choice1}\nC. {choice2}\nD. {choice3}\n请从“A”，“B”，“C”，“D”中进行选择。\n答：",
            ),
        ]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

C3_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=first_capital_postprocess),
)

intentionrecognition_datasets = [
    dict(
        # abbr="C3",
        abbr="text_analysis-intention-recognition",
        type=C3Dataset_V2,
        # sampled from "./data/CLUE/C3/dev_0.json",
        path="./data_na/text_analysis/intention_recognition.json",
        reader_cfg=C3_reader_cfg,
        infer_cfg=C3_infer_cfg,
        eval_cfg=C3_eval_cfg,
    )
]
