#  XiYanSQL-QwenCoder Models

### Important Links

🤗[HuggingFace](https://huggingface.co/collections/XGenerationLab/xiyansql-models-67c9844307b49f87436808fc) |
🤖[ModelScope](https://modelscope.cn/collections/XiYanSQL-Models-4483337b614241) |
📖[XiYan-SQL](https://github.com/XGenerationLab/XiYan-SQL) |
📄[Arxiv](https://arxiv.org/abs/2507.04701)| 
🌕[析言GBI](https://bailian.console.aliyun.com/xiyan) |
💻[Modelscope Space](https://www.modelscope.cn/studios/XGenerationLab/XiYanSQL-QwenCoder-32B)


## News🔥
+ `Apr. 29, 2025` 🌟 We are excited to open source our latest SQL generation models, **XiYanSQL-QwenCoder-2504**. This version continues to optimize upon the previous release, representing new SOTA performance in Text-to-SQL models.
+ `Feb. 2025` 🌟 We have updated the model links on the Hugging Face platform.
+ `Feb. 2025` 🌟 We are excited to open source the XiYanSQL-QwenCoder series model, dedicated to advancing the development of LLMs in the Text-to-SQL domain. 
Building on our previous release of the powerful **32B** model, this release introduces three model sizes: **3B**, **7B**, and **14B**. As of now, XiYanSQL-QwenCoder covers a variety of mainstream model sizes to meet the needs of different developers.
+ `Dec. 2024` 🌟 We are excited to open source the XiYanSQL-QwenCoder-32B model: XiYanSQL-QwenCoder-32B achieves an EX score of **69.03%** on the BIRD test set, setting a new SOTA under only a single fine-tuned model.
  
## Introduction

We are excited to release the **XiYanSQL-QwenCoder-2504** version, our latest SQL generation model. This version continues to optimize upon the previous version, delivering enhanced performance.
- Our model incorporates important explorations combining **fine-tuning and GRPO training**, leveraging the post-training strategies of GRPO without a thinking process, achieving both efficiency and accuracy in SQL generation.
- It demonstrates **impressive performance** and supports **multiple dialects**, ready to use out of the box.
- Improved generalization capabilities, excelling on different dialects and **out-of-domain datasets**.

In this evaluation, we have also added **a real-world SQL benchmark (the DW test set)**, which serves as an important internal evaluation baseline. This test set includes thousands of complex queries from real scenarios in both PostgreSQL and MySQL dialects, effectively reflecting the model's performance across multiple dialects and out-of-domain data.

---



> We are excited to open source the XiYanSQL-QwenCoder series model, dedicated to advancing the development of LLMs in the text-to-SQL domain. As of now, XiYanSQL-QwenCoder covers four mainstream model sizes: 3B, 7B, 14B, and 32B parameters, to meet the needs of different developers.
> - The XiYanSQL-QwenCoder model demonstrates strong performance in SQL generation, with the **XiYanSQL-QwenCoder-32B** achieving a **69.03%** EX score on the BIRD TEST set, setting a new SOTA with a single fine-tuned model. Other models in the series also maintain a leading position at their respective sizes.
> - The XiYanSQL-QwenCoder model supports multiple SQL dialects, such as SQLite, PostgreSQL, and MySQL.
>- The XiYanSQL-QwenCoder model can be used directly for text-to-SQL tasks or serve as a better starting point for fine-tuning SQL models.


## Model Downloads


| **Model** | **Download Latest** |
|-----------|------------------|
|**XiYanSQL-QwenCoder-3B-2504**  | 🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-3B-2504) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-3B-2504)   |
|**XiYanSQL-QwenCoder-7B-2504**  | 🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-7B-2504) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-7B-2504)   |
|**XiYanSQL-QwenCoder-14B-2504**  | 🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-14B-2504) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-14B-2504)   |
|**XiYanSQL-QwenCoder-32B-2504** | 🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-32B-2504) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-32B-2504) |
|XiYanSQL-QwenCoder-3B-2502  |🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-3B-2502) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-3B-2502)|
|XiYanSQL-QwenCoder-7B-2502  |🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-7B-2502) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-7B-2502)|
|XiYanSQL-QwenCoder-14B-2502 |🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-14B-2502) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-14B-2502)|
|XiYanSQL-QwenCoder-32B-2412 |🤗[HuggingFace](https://huggingface.co/XGenerationLab/XiYanSQL-QwenCoder-32B-2412) 🤖[Modelscope](https://www.modelscope.cn/models/XGenerationLab/XiYanSQL-QwenCoder-32B-2412)|



## Performance
The XiYanSQL-QwenCoder models, as multi-dialect SQL base models, demonstrating robust SQL generation capabilities. The following presents the evaluation results at the time of release. We conducted a comprehensive evaluation of the model's performance under two schema formats, M-Schema, and original DDL, using the BIRD and Spider as SQLite benchmarks in the Text-to-SQL domain, as well as DW benchmarks for PostgreSQL and MySQL dialects.

| Model name                   |  Size  | BIRD Dev@M-Schema | BIRD Dev@DDL | Spider Test@M-Schema | Spider Test@DDL | DW PostgreSQL@M-Schema | DW MySQL@M-Schema |
|------------------------------|:------:|:-----------------:|:------------:|:--------------------:|:---------------:|:----------------------:|:-----------------:|
| GPT-4o-0806                  |  UNK   |      58.47%       |    54.82%    |        82.89%        |     78.45%      |         46.79%         |      57.77%       |
| GPT-4.1-0414                 |  UNK   |      59.39%       |    54.11%    |        84.45%        |     79.86%      |         54.29%         |      63.18%       |
| Claude3.5-sonnet-1022        |  UNK   |      53.32%       |    50.46%    |        76.27%        |     73.04%      |         55.22%         |      52.84%       |
| Claude3.7-sonnet             |  UNK   |      54.82%       |    49.22%    |        78.04%        |     74.66%      |         53.23%         |      54.61%       |
| Gemini-1.5-Pro               |  UNK   |      61.34%       |    57.89%    |        85.11%        |     84.00%      |         52.78%         |      62.78%       |
| DeepSeek-V2.5-1210           |  236B  |      55.74%       |    55.61%    |        82.08%        |     80.57%      |         45.74%         |      52.18%       |
| DeepSeek-V3                  |  685B  |      59.58%       |    56.71%    |        81.52%        |     79.91%      |         52.56%         |      55.95%       |
| DeepSeek-R1                  |  685B  |      58.15%       |    55.61%    |        80.72%        |     78.85%      |         60.56%         |      62.00%       |
| DeepSeek-R1-Distill-Qwen-32B |  32B   |      50.65%       |    48.31%    |        78.65%        |     77.33%      |         37.22%         |      44.72%       |
| Deepseek-Coder-33B-Instruct  |  33B   |      47.52%       |    44.72%    |        72.39%        |      62.0%      |         31.48%         |      36.17%       |
| OmniSQL-32B                  |  32B   |      60.37%       |    55.87%    |        85.16%        |     83.19%      |         38.19%         |      42.34%       |
| XiYanSQL-QwenCoder-7B-2502   |  7B    |      59.65%       |    56.32%    |        84.15%        |     80.01%      |         39.38%         |      42.10%       |
| XiYanSQL-QwenCoder-7B-2504   |  7B    |      62.13%       |    57.43%    |        85.97%        |     82.48%      |         42.08%         |      44.67%       |
| XiYanSQL-QwenCoder-32B-2412  |  32B   |      67.07%       |    63.04%    |        88.39%        |     85.46%      |         45.07%         |      52.84%       |
| XiYanSQL-QwenCoder-32B-2504  |  32B   |      67.14%       |    62.26%    |        89.20%        |     86.17%      |         53.52%         |      57.74%       |


## Requirements

transformers >= 4.37.0
vllm >= 0.7.2

## Quickstart

> NOTE: XiYanSQL-QwenCoder models can be used directly for text-to-SQL tasks or serve as a better starting point for fine-tuning SQL models.


Here is a simple code snippet for quickly using **XiYanSQL-QwenCoder** model. We provide a Chinese version of the prompt, and you just need to replace the placeholders for "question," "db_schema," and "evidence" to get started. We recommend using our [M-Schema](https://github.com/XGenerationLab/M-Schema) format for the schema; other formats such as DDL are also acceptable, but they may affect performance.
Currently, we mainly support mainstream dialects like SQLite, PostgreSQL, and MySQL.

In response to community demands, we have also included example code for inference using vLLM.

### Prompt Template
```python
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
```

### Inference with Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
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

### Inference with vLLM
```python
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
model_path = "XGenerationLab/XiYanSQL-QwenCoder-32B-2504"
llm = LLM(model=model_path, tensor_parallel_size=8)
tokenizer = AutoTokenizer.from_pretrained(model_path)
sampling_params = SamplingParams(
    n=1,
    temperature=0.1,
    max_tokens=1024
)

## dialects -> ['SQLite', 'PostgreSQL', 'MySQL']
prompt = nl2sqlite_template_cn.format(dialect="", db_schema="", question="", evidence="")
message = [{'role': 'user', 'content': prompt}]
text = tokenizer.apply_chat_template(
    message,
    tokenize=False,
    add_generation_prompt=True
)
outputs = llm.generate([text], sampling_params=sampling_params)
response = outputs[0].outputs[0].text
```

## Contact us:

If you are interested in our research or products, please feel free to contact us.

#### Contact Information:

Yifu Liu, zhencang.lyf@alibaba-inc.com

#### Join Our DingTalk Group

<a href="https://github.com/XGenerationLab/XiYan-SQL/blob/main/xiyansql_dingding.png">Ding Group钉钉群</a> 



## Acknowledgments
If you find our work useful, please give us a citation or a star, so we can make a greater contribution to the open-source community!


## Citation
```bibtex
@article{XiYanSQL,
      title={XiYan-SQL: A Novel Multi-Generator Framework For Text-to-SQL}, 
      author={Yifu Liu and Yin Zhu and Yingqi Gao and Zhiling Luo and Xiaoxia Li and Xiaorong Shi and Yuntao Hong and Jinyang Gao and Yu Li and Bolin Ding and Jingren Zhou},
      year={2025},
      eprint={2507.04701},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.04701}, 
}
```

