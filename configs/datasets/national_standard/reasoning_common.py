from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import FixKRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import first_capital_postprocess
from opencompass.datasets.nation_standard import TwoOptionDataset, ThreeOptionDataset

reasonbench_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_postprocessor=dict(type=first_capital_postprocess)
)

reader_cfgs = []
for i in range(2, 5):
    choices = ["A", "B", "C", "D"][:i]

    reader_cfgs.append(dict(
    input_columns=["prompt_ppl"],
    output_column="label_ppl")
    )

infer_cfg=dict(
    ice_template=dict(
        type=PromptTemplate,
        template=dict(
            begin="</E>",
            round=[
                dict(
                    role="HUMAN",
                    prompt="</E>{prompt_ppl}"
                ),
                dict(role="BOT", prompt="Answer: {label_ppl}"),
            ]),
        ice_token="</E>",
        ),
    retriever=dict(type=FixKRetriever, fix_id_list=[]),
    inferencer=dict(type=GenInferencer)
)

CN_CommonsenseReasoningDataset = [
    dict(
        abbr="reasoning-cn_commonsense",
        type=ThreeOptionDataset,
        path="data_na/reasoning/cleva_commonsense.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

CN_DeductiveReasoningDataset = [
    dict(
        abbr="reasoning-cn_deductive",
        type=ThreeOptionDataset,
        path="data_na/reasoning/cleva_deductive.jsonl",
        reader_cfg=reader_cfgs[1],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

CN_InductiveReasoningDataset = [
    dict(
        abbr="reasoning-cn_inductive",
        type=TwoOptionDataset,
        path="data_na/reasoning/cleva_inductive.jsonl",
        reader_cfg=reader_cfgs[0],
        infer_cfg=infer_cfg,
        eval_cfg=reasonbench_eval_cfg),
]

reasoning_common_datasets = CN_CommonsenseReasoningDataset + CN_DeductiveReasoningDataset + CN_InductiveReasoningDataset