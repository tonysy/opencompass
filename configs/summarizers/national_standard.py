from mmengine.config import read_base

with read_base():
    from .groups.agieval import agieval_summary_groups
    from .groups.mmlu import mmlu_summary_groups
    from .groups.cmmlu import cmmlu_summary_groups
    from .groups.ceval import ceval_summary_groups
    from .groups.bbh import bbh_summary_groups
    from .groups.GaokaoBench import GaokaoBench_summary_groups
    # from .groups.flores import flores_summary_groups
    from .groups.flores_cn import flores_cn_summary_groups
    from .groups.jigsaw_multilingual import jigsaw_multilingual_summary_groups
    from .groups.tydiqa import tydiqa_summary_groups
    from .groups.xiezhi import xiezhi_summary_groups

summarizer = dict(
    dataset_abbrs=[
        '--------- 信息抽取 Information Retrivieval----',
        # 命名实体识别
        'information_retrieval-entity-recognition',
        '--------- 文本分析 Text Analysisi ------------',  # category
        # 文本分类
        'text_analysis-text-classification',
        # 情感分析
        'text_analysis-sentiment-analysis',
        # 意图识别
        'text_analysis-intention-recognition',
        '------------- 翻译 Translation --------------',  # category
        # '翻译', # subcategory
        'flores_cn_100',
        '--------------- 知识 Knowledge --------------',  # category
        'ceval',
        'ceval-stem', # 理工科
        'ceval-social-science', # 社会科学
        'ceval-humanities', # 人文
        'ceval-other', # 其他
        '--------------- 推理 Reasoning --------------',  # category
        # 常识推理
        'reasoning-cn_commonsense',
        # 演绎推理
        'reasoning-cn_deductive',
        # 归纳推理
        'reasoning-cn_deductive',
        # 数学推理
        'reasoning_math-primary-cloze_cn',
        '----------------- 代码 Coding ---------------',  # category
        # '代码', # subcategory
        'openai_humaneval',
    ],
    summary_groups=sum(
        [v for k, v in locals().items() if k.endswith("_summary_groups")], []),
)
