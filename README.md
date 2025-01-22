#  XiYanSQL-QwenCoder-32B

### Important Links


🤖[ModelScope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-32B-2412) |
📖[XiYan-SQL](https://github.com/XGenerationLab/XiYan-SQL) |
🌕[析言GBI](https://bailian.console.aliyun.com/xiyan) |
🤗[Modelscope Space](https://www.modelscope.cn/studios/XGenerationLab/XiYanSQL-QwenCoder-32B)


## Introduction
We open-source the first XiYanSQL-QwenCoder-32B model on January 22, 2025, and we look forward to contributing to the text-to-SQL community.
**XiYanSQL-QwenCoder-32B**, a SQL model fine-tuned on the Qwen2.5Coder-32B model, achieves an EX score of **69.03%** on the BIRD test set, setting a new SOTA under only a single fine-tuned model.
In the future, we will release more SQL-related models.


## Requirements

transformers >= 4.37.0

## Quickstart

Here is a simple code snippet for quickly using **XiYanSQL-QwenCoder-32B** model. We provide a Chinese version of the prompt, and you just need to replace the placeholders for "question," "db_schema," and "evidence" to get started. We recommend using our [M-Schema](https://github.com/XGenerationLab/M-Schema) format for the schema; other formats such as DDL are also acceptable, but they may affect performance.


```

nl2sqlite_template_cn = """你是一名SQLite专家，现在需要阅读并理解下面的【数据库schema】描述，以及可能用到的【参考信息】，并运用SQLite知识生成sql语句回答【用户问题】。
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

prompt = nl2sqlite_template_cn.format(db_schema="", question="", evidence="")
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

## Performance


## Acknowledgments
If you find our work useful, please give us a citation or a thumbs up, so we can make a greater contribution to the open-source community!











