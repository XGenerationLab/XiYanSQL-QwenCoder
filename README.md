#  XiYanSQL-QwenCoder Models

### Important Links

💻[HuggingFace](https://huggingface.co/collections/XGenerationLab/xiyansql-models-67c9844307b49f87436808fc) |
🤖[ModelScope](https://modelscope.cn/collections/XiYanSQL-Models-4483337b614241) |
📖[XiYan-SQL](https://github.com/XGenerationLab/XiYan-SQL) |
🌕[析言GBI](https://bailian.console.aliyun.com/xiyan) |
🤗[Modelscope Space](https://www.modelscope.cn/studios/XGenerationLab/XiYanSQL-QwenCoder-32B)


## News🔥
We have updated the model links on the Hugging Face platform.

We are excited to open source the XiYanSQL-QwenCoder series model, dedicated to advancing the development of LLMs in the Text-to-SQL domain. 
Building on our previous release of the powerful **32B** model, this release introduces three model sizes: **3B**, **7B**, and **14B**. As of now, XiYanSQL-QwenCoder covers a variety of mainstream model sizes to meet the needs of different developers.

## Introduction

We open-source the first XiYanSQL-QwenCoder-32B model on January 22, 2025, and we look forward to contributing to the Text-to-SQL community.
**XiYanSQL-QwenCoder-32B**, a SQL model fine-tuned on the Qwen2.5Coder-32B model, achieves an EX score of **69.03%** on the BIRD test set, setting a new SOTA under only a single fine-tuned model.


## Model Downloads


| **Model** | **Download Latest** |
|-----------|------------------|
|XiYanSQL-QwenCoder-3B  |[🤗 Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-3B-2502)|
|XiYanSQL-QwenCoder-7B  |[🤗 Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-7B-2502)|
|XiYanSQL-QwenCoder-14B |[🤗 Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-14B-2502)|
|XiYanSQL-QwenCoder-32B |[🤗 Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-32B-2412)|



## Performance
The XiYanSQL-QwenCoder models, as multi-dialect SQL base models, demonstrating robust SQL generation capabilities. The following presents the evaluation results at the time of release. We conducted a comprehensive evaluation of the model's performance under two schema formats, M-Schema, and original DDL, using the BIRD and Spider benchmarks in the Text-to-SQL domain.

| Model name|BIRD Dev@M-Schema |BIRD Dev@DDL|Spider Test@M-Schema|Spider Test@DDL|
|-----------|:------------------:|:---------------:|:-------------------:|:---------------:|
|Codellama-34b              | 33.05%     | -          | 67.74%      | -           |
|Deepseek-coder-33b         | 47.52%     | 44.72%     | 72.39%      | -           |
|TableGPT2                  | 46.35%     | 47.07%     | 74.76%      | 77.28%      |
|Codestral 22b              | 50.52%     | 47.00%     | 78.45%      | 75.47%      |
|GLM-4-plus                 | 54.37%     | -          | 79.40%      | -           |
|Claude35_sonnet-1022       | 53.32%     | 50.46%     | 76.27%      | 73.04%      |
|Deepseek(v2.5-1210)        | 55.74%     | 55.61%     | 82.08%      | 80.57%      |
|Gemini-1.5-pro             | 61.34%     | 57.89%     | 85.11%      | 84.00%      |
|GPT-4o-0806                | 58.47%     | 54.82%     | 82.89%      | 78.45%      |
|XiYanSQL-QwenCoder-3B      | 54.11%     | 53.19%     | 82.69%      | 78.85%      |
|XiYanSQL-QwenCoder-7B      | 59.78%     | 56.58%     | 84.86%      | 80.31%      |
|XiYanSQL-QwenCoder-14B     | 63.10%     | 60.37%     | 85.76%      | 82.79%      |
|XiYanSQL-QwenCoder-32B     | 67.01%     | 63.04%     | 88.39%      | 85.46%      |



## Requirements

transformers >= 4.37.0

## Quickstart

> NOTE: XiYanSQL-QwenCoder models can be used directly for text-to-SQL tasks or serve as a better starting point for fine-tuning SQL models.


Here is a simple code snippet for quickly using **XiYanSQL-QwenCoder** model. We provide a Chinese version of the prompt, and you just need to replace the placeholders for "question," "db_schema," and "evidence" to get started. We recommend using our [M-Schema](https://github.com/XGenerationLab/M-Schema) format for the schema; other formats such as DDL are also acceptable, but they may affect performance.
Currently, we mainly support mainstream dialects like SQLite, PostgreSQL, and MySQL.

```

nl2sqlite_template_cn = """你是一名{dialect}专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用{dialect}知识生成sql语句回答【用户问题】。
【用户问题】
{question}

【数据库schema】
{db_schema}

【参考信息】
{evidence}

【用户问题】
{question}

```sql"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "XGenerationLab/XiYanSQL-QwenCoder-32B-2412"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

## dialects -> ['SQLite', 'PostgreSQL', 'MySQL']
prompt = nl2sqlite_template_cn.format(dialect="", db_schema="", question="", evidence="")
message = [{'role': 'user', 'content': prompt}]

text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    max_new_tokens=1024,
    temperature=0.1,
    top_p=0.8,
    do_sample=True,
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

```



## Acknowledgments
If you find our work useful, please give us a citation or a star, so we can make a greater contribution to the open-source community!











