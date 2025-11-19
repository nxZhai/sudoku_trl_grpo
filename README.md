# sudoku_trl_grpo

## 📖简介

最近在补充NLP任务领域的GRPO强化学习训练任务，我们希望用GRPO实现一个简单的数独游戏。

本次实验我们使用GRPO的方法，用lora来做微调，框架选择trl，我们分别在GPU、NPU的AI训练卡上训练，同时我们也对比了3B模型、7B模型的训练效果，并且通过不断地调整参数实现最终准确度达到89%。

**详细教程和SwanLab观测结果链接如下：**

[![知乎](https://img.shields.io/static/v1?label=📖&message=教程&color=blue)](https://zhuanlan.zhihu.com/p/1974529769119962764)
[![SwanLab](https://img.shields.io/static/v1?label=📈&message=SwanLab&color=green)](https://swanlab.cn/@LiXinYu/sudoku-grpo-qwen2.5/overview)

## ⚙️环境安装

**GPU环境安装**

安装命令：

```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
pip install -r requirements.txt
```

对硬件条件的要求（per_device_train_batch_size<=2）：

- 3B模型：普通训练2块5090，vllm的话3块
- 7B模型：普通训练4块5090，vllm的话5块

如果per_device_train_batch_size超过2，上面的资源撑不住，需要再多几块5090。其他GPU具体没试过，因为本次教程我使用AutoDL上的算力实现的，用的更多还是5090。

除了GPU，昇腾的910B2我们也进行了实验，下面我们补充了下NPU的环境配置👇：

**NPU环境安装**

参考文献：[官方文档](https://www.hiascend.com/document/detail/zh/Pytorch/600/configandinstg/instg/insg_0002.html)

建议前置安装包
```bash
apt update  -y
apt install -y gcc g++ libnuma-dev
```

建议前置安装如下包
```bash
pip install attrs cython numpy==1.24.0 decorator sympy cffi pyyaml pathlib2 psutil protobuf==3.20 scipy requests absl-py ml-dtypes tornado cloudpickle jinja2
```

可能出现如下错误，暂时忽略
```bash
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
te 0.4.0 requires cloudpickle, which is not installed.
op-compile-tool 0.1.0 requires getopt, which is not installed.
op-compile-tool 0.1.0 requires inspect, which is not installed.
op-compile-tool 0.1.0 requires multiprocessing, which is not installed.
dataflow 0.0.1 requires jinja2, which is not installed.
```

安装pytorch 2.4.0和torch_npu 6.0.0

```bash
# 下载PyTorch安装包
wget https://download.pytorch.org/whl/cpu/torch-2.4.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 下载torch_npu插件包
wget https://gitee.com/ascend/pytorch/releases/download/v6.0.0-pytorch2.4.0/torch_npu-2.4.0.post2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
# 安装命令
pip install torch-2.4.0-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch_npu-2.4.0.post2-cp311-cp311-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
```

## 📊数据生成

训练数据样例：

```json
{"question": "_24___1234212134", "answer": "1243431234212134", "label": "simple"}
```

运行下面的命令，可以直接本地生成数据集：

```bash
python datacreate.py
```

具体可以生成下面的文件👇

```python
# 生成的数据
├──data
│   ├── sudoku_4x4_answer.jsonl  # 288条4*4只有答案的数据集
│   └── sudoku_4x4_qa.jsonl      # 随机挖空后的包含问题和答案的数据集（训练用）  
```

## 训练启动命令

由于我们在训练代码中未添加各种参数config设置，因此我们需要另外的文件去配置这些超参数，然后我们的实验都使用`accelerate`分布式训练来实现训练的加速。

```python
# 训练文件
├──configs/   # 参数配置文件
│   ├── deepspeed_zero3.yaml  # deepspeed参数配置
│   ├── grpo_qwen2.5-7b-it_lora.yaml  # 7B模型的参数配置
│   └── grpo_qwen2.5-3b-it_lora.yaml  # 3B模型的参数配置
├──scripts/   # 训练启动脚本
│   └── train_grpo.sh
└── train_grpo.py
```

如果不用`vllm`做推理加速，那么直接运行下面的代码就可以训练：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash ./scripts/train_grpo.sh
```

**使用vllm做推理加速**

首先我们选择一个块作为推理使用，要注意单卡确保能跑模型，不然显存不够。假如我们是3卡，我们令第三块卡为推理卡：

```bash
CUDA_VISIBLE_DEVICES=2 trl vllm-serve /your/path/of/model
```

然后在剩下两块卡上跑训练任务

```bash
CUDA_VISIBLE_DEVICES=0,1 bash ./scripts/train_grpo.sh
```

**合并模型**

```bash
python ./merge/merge-lora.py \
        --lora_path /root/autodl-tmp/models/outputs/grpo_qwen2.5-3b-it_lora \
        --base_model_path /root/autodl-tmp/models/qwen/qwen2.5-3b-it \
        --merge_path /root/autodl-tmp/models/outputs/merged-model-gpu
```

- `lora_path`:训练完成后保存的lora参数地址
- `base_model_path`:原模型保存地址
- `merge_path`:合并后模型保存地址


## 结果评测

**生成结果数据**

```bash
bash ./eval/generate.sh
```

如果需要`vllm`加速推理，请在运行上述代码前，开启`vllm serve`：

```bash
vllm serve /your/path/of/model
```

**评估结果**

```python
python ./eval/eval.py
```
