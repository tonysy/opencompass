# from flores_gen_806ede
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import TopkRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import FloresFirst100Dataset
from opencompass.openicl.icl_retriever import FixKRetriever

_flores_lang_map = [
    ["eng", "eng_Latn", "English", "Indo-European-Germanic"],
    ["zho_simpl", "zho_Hans", "Chinese (Simpl)", "Sino-Tibetan"],
    ["zho_trad", "zho_Hant", "Chinese (Trad)", "Sino-Tibetan"],
]
flores_lang_map = {i[0]: i for i in _flores_lang_map}
_flores_subtasks = [f"eng-{i}" for i in flores_lang_map if i != "eng"
                    ] + [f"{i}-eng" for i in flores_lang_map if i != "eng"]

translation_datasets = []
for _flores_subtask in _flores_subtasks:
    _src, _tgt = _flores_subtask.split("-")
    _, _flores_source, _src_inst, _ = flores_lang_map[_src]
    _, _flores_target, _tgt_inst, _ = flores_lang_map[_tgt]

    flores_reader_cfg = dict(
        input_columns=f"sentence_{_flores_source}",
        output_column=f"sentence_{_flores_target}",
        train_split="dev",
        test_split="devtest"
    )
    flores_infer_cfg = dict(
        ice_template=dict(
            type=PromptTemplate,
            template=dict(
                begin="</E>",
                round=[
                    dict(
                        role="HUMAN",
                        prompt=
                        f"Translate the following {_src_inst} statements to {_tgt_inst}.\n{{sentence_{_flores_source}}}"
                    ),
                    dict(role="BOT", prompt=f"{{sentence_{_flores_target}}}"),
                ],
            ),
            ice_token="</E>",
        ),
        retriever=dict(type=FixKRetriever, fix_id_list=[0, 1, 2, 3, 4, 5, 6, 7]),
        inferencer=dict(type=GenInferencer),
    )
    flores_eval_cfg = dict(
        evaluator=dict(type=BleuEvaluator),
        pred_role="BOT",
    )
    if _tgt == "zho_simpl":
        flores_eval_cfg["pred_postprocessor"] = dict(type="flores")
        flores_eval_cfg["dataset_postprocessor"] = dict(type="flores")
    
    translation_datasets.append(
        dict(
            abbr=f"flores_cn_100_{_src}-{_tgt}",
            type=FloresFirst100Dataset,
            path='./data_na/translation/flores_first100',
            name=f"{_flores_source}-{_flores_target}",
            reader_cfg=flores_reader_cfg.copy(),
            infer_cfg=flores_infer_cfg.copy(),
            eval_cfg=flores_eval_cfg.copy(),
        ))
