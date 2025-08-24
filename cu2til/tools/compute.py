
#compute encoder FLOPS
def compute_Wq(seq_len, hidden_size,  hidden_size_out):
    return 2 * seq_len * hidden_size * hidden_size_out

def compute_Wk(seq_len, hidden_size, hidden_size_out):
    return 2 * seq_len * hidden_size * hidden_size_out

def compute_Wv(seq_len, hidden_size, hidden_size_out):
    return 2 * seq_len * hidden_size * hidden_size_out

def compute_Wo(seq_len, hidden_size, hidden_size_out):
    return 2 * seq_len * hidden_size * hidden_size_out

# softmax(QK^T / sqrt(d_k)) * V
def compute_attention(Q_left, Q_right,  K_right, K_left, V_left, V_right):
    assert Q_right == K_right
    assert K_left == V_left
    return 2 * Q_left * Q_right * K_left + 2 * Q_left * V_left * V_right

def visual_mlp(seq_len, hidden_size,  linear1_left, linear1_right, linear2_left, linear2_right):
    assert hidden_size == linear1_left
    assert linear1_right == linear2_left

    return 2 * seq_len * hidden_size * linear1_right + 2 * seq_len * linear1_right * linear2_right

def visual_encoder_FLOPS(seq_len, hidden_size):
    blocks_number = 32 
    hidden_size_out = hidden_size
    mlp_linear1_left , mlp_linear1_right = hidden_size, hidden_size * 4 
    mlp_linear2_left , mlp_linear2_right = hidden_size * 4, hidden_size 

    one_block_FLOPS = (compute_Wq(seq_len, hidden_size, hidden_size_out) + compute_Wk(seq_len, hidden_size, hidden_size_out) + compute_Wv(seq_len, hidden_size, hidden_size_out) +
                        compute_attention(seq_len, hidden_size, hidden_size, seq_len, seq_len, hidden_size) + 
                        compute_Wo(seq_len, hidden_size, hidden_size_out) + 
                        visual_mlp(seq_len, hidden_size, mlp_linear1_left, mlp_linear1_right, mlp_linear2_left, mlp_linear2_right)
                        )
    
    print(f"encoder FLOPS is {(blocks_number * one_block_FLOPS) / (1024 **3)} GFLOPS")
    return (blocks_number * one_block_FLOPS) / (1024 **3)

def visual_encoder_Memory(seq_len, hidden_size = 1280, data_type_size = 2):
    input = seq_len * hidden_size * data_type_size
    wq = data_type_size * hidden_size * hidden_size
    wk = data_type_size * hidden_size * hidden_size
    wv = data_type_size * hidden_size * hidden_size
    wo = data_type_size * hidden_size * hidden_size
    mlp = data_type_size * hidden_size * hidden_size * 4 + data_type_size * hidden_size * 4 * hidden_size
    sum_bytes = 32 * (wq + wk + wv + wo + mlp)
    print(f"memory usage : {sum_bytes / (1024**3)}GB")
    return sum_bytes / (1024**3)
import matplotlib.pyplot as plt
def plot_encoder():
    x = [i for i in range(1, 200, 10)]
    compute_div_memory = [visual_encoder_FLOPS(i, 1280) / visual_encoder_Memory(i, 1280) for i in x]
    plt.figure(figsize=(10, 6))
    plt.plot(x, compute_div_memory, label='FLOPS/Memory')
    plt.xlabel('Input Size')  # X轴标签
    plt.ylabel('FLOPS / Memory')  # Y轴标签
    plt.title('FLOPS to Memory Ratio of Visual Encoder')  # 图标题
    plt.legend()  # 显示图例
    plt.grid(True)  # 显示网格
    plt.savefig("./compute_memory_encoder.jpg")

plot_encoder()

visual_encoder_FLOPS(1000, 1280)

visual_encoder_Memory(1000, 1280)


    
def compute_Qwen2MLP(seq_len, hidden_size, gate_in, gate_out, up_in, up_out, down_in, down_out):
    return 2 * seq_len * hidden_size * gate_out + 2 * seq_len * hidden_size * up_out + 2 * seq_len * down_in * down_out


def Qwen2VLModel_FLOPS(seq_len, hidden_size):
    blocks_number = 28 
    hidden_size_out = 512 
    mlp_linear1_left , mlp_linear1_right = hidden_size, hidden_size * 4 
    mlp_linear2_left , mlp_linear2_right = hidden_size * 4, hidden_size 

    one_block_FLOPS = (compute_Wq(seq_len, hidden_size, hidden_size) + compute_Wk(seq_len, hidden_size, hidden_size_out) + compute_Wv(seq_len, hidden_size, hidden_size_out) +
                        compute_attention(seq_len, hidden_size, hidden_size, seq_len, seq_len, hidden_size) + 
                        compute_Wo(seq_len, hidden_size, hidden_size_out) + 
                        compute_Qwen2MLP(seq_len, hidden_size, hidden_size, 18944, hidden_size, 18944, 18944, hidden_size)
                        )
    
    print(f"qwen2llm FLOPS is {(blocks_number * one_block_FLOPS) / (1000 **3)} GFLOPS")

Qwen2VLModel_FLOPS(600, 3584)
