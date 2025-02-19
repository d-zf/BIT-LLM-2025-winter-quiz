import torch
import numpy as np
from transformers import PreTrainedTokenizerFast

# 加载模型
model = torch.load('pretrained_model.pt',weights_only=False)
model.eval()
print("模型加载成功")

# 启用隐藏状态输出
model.config.output_hidden_states = True

# 获取模型最大位置嵌入数
MAX_LEN = model.config.max_position_embeddings  # 通常为1024
print(f"模型最大支持长度：{MAX_LEN}")

# 初始化分词器
tokenizer = PreTrainedTokenizerFast(tokenizer_file='addgene_trained_dna_tokenizer.json')
tokenizer.add_special_tokens({'additional_special_tokens': ['[PROMPT]', '[PROMPT2]']})

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def process_sequence(sequence: str) -> np.ndarray:
    """处理单个DNA序列"""
    # 添加特殊标记并编码
    prefix = [3]*10 + [2]
    encoded = tokenizer.encode(sequence)
    
    # 计算总长度
    total_len = len(prefix) + len(encoded)
    
    # 长度校验与截断
    if total_len > MAX_LEN:
        print(f"警告：序列长度{total_len}超过最大值{MAX_LEN}，将截断")
        # 优先保留原始序列内容
        available_len = MAX_LEN - len(prefix)
        encoded = encoded[:available_len]
    
    # 合并标记
    tokenized_seq = prefix + encoded
    
    # 生成嵌入
    with torch.no_grad():
        input_ids = torch.tensor([tokenized_seq], dtype=torch.long).to(device)
        outputs = model(input_ids)
        hidden_states = outputs.hidden_states[-1].cpu().numpy()
        return np.mean(hidden_states, axis=1).reshape(-1)

all_embeddings = []

with open('plasmids.fasta', 'r') as f:
    current_seq = []
    for line in f:
        line = line.strip()
        if line.startswith('>'):
            if current_seq:
                seq = ''.join(current_seq)
                try:
                    emb = process_sequence(seq)
                    all_embeddings.append(emb)
                except Exception as e:
                    print(f"处理序列时出错（长度{len(seq)}）：{str(e)}")
                current_seq = []
        else:
            current_seq.append(line)
    
    # 处理最后一个序列
    if current_seq:
        seq = ''.join(current_seq)
        try:
            emb = process_sequence(seq)
            all_embeddings.append(emb)
        except Exception as e:
            print(f"处理序列时出错（长度{len(seq)}）：{str(e)}")

# 保存结果
if all_embeddings:
    embeddings_array = np.array(all_embeddings)
    np.save('sequence_embeddings.npy', embeddings_array)
    print(f"成功保存{len(all_embeddings)}个嵌入")
    np.savetxt('sequence_embeddings.txt', embeddings_array)
else:
    print("没有生成有效嵌入")