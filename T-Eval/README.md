## 介绍
这是在Teval的代码框架的基础上构建的金融智能体benchmark. 我们评估模型在以下六个维度的能力: instruct, plan, reason, retrieve, understand 以及 review.
## 🛠️ 安装

```bash
$ cd T-Eval
$ pip install -r requirements.txt
$ cd lagent && pip install -e .
```

##  🛫️ 开始

### 🤖 API Models

1. 设置OPENAI_API_KEY和OPENAI_API_BASE
```bash
export OPENAI_API_KEY=xxxxxxxxx
export OPENAI_API_BASE=xxxxxxxxx
```
2. 使用以下脚本运行评测(model_name 可以是OpenAI模型或者支持openai库调用的模型，例如deepseek-chat)
```bash
sh test.sh model_name output_path
```

### 🤗 Local Models

1.使用vllm部署你的模型
```bash
CUDA_VISIBLE_DEVICES=0,1 python -m vllm.entrypoints.openai.api_server \
    --model model_path \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --served-model-name model_name \
    --block-size 16  \
    --trust-remote-code \
    --chat-template chat_templates_path \
    --port 8081
```
2.对于不同的模型，可以从T-Eval/chat_templates目录下选择相应的模板文件；使用以下脚本运行评测
```bash
export MKL_THREADING_LAYER=GNU \
export MKL_SERVICE_FORCE_INTEL=1 \
export OPENAI_API_KEY="EMPTY" \
export OPENAI_API_BASE=http://0.0.0.0:8081/v1

sh test.sh model_name output_path
```

### 💫 最终结果
一旦你测试完了所有的数据，结果的细节会放在 `output_path/teval_output/result.json`
