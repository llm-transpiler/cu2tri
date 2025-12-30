from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="totemic-fact-469314-n2",
      location="global",
  )

  msg1_text1 = types.Part.from_text(text="""// Load Q tile from Shared to Registers
      serial_range_for(i, 0, kWarpTileSeqLenQ, 1) { // 1
        IndexInt warp_smem_Q_Br = warp_QP * (kMmaAtomM * kWarpTileSeqLenQ) + i * kMmaAtomM;
        IndexInt lane_smem_Q_Br = warp_smem_Q_Br + lane_id % 16; // lane在warp中的smem行坐标, lane本身排布列优先, 所以%16, 符合ldmatrix标准
        IndexInt lane_smem_Q_d = tile_K_d * kMmaAtomK + (lane_id / 16) * 8; // lane在warp中的smem列坐标
        SmemAddr lane_smem_Q_addr = smem_Q_base_addr + (lane_smem_Q_Br * kHeadDim + lane_smem_Q_d) * sizeof(fp16);
        warp_copy_sync<fp16, 128, CopyTask::S2R, Layout::ROW_MAJOR>(lane_smem_Q_addr, R_Q[i][0], R_Q[i][1], R_Q[i][2], R_Q[i][3]);
      }

这一段代码拷贝的数据有多大，注意warp_copy_sync<dtype, bits, task, target_layout>(src_addr, target_addr_list)""")

  model = "gemini-2.5-pro"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 1,
    top_p = 0.95,
    seed = 0,
    max_output_tokens = 65535,
    safety_settings = [types.SafetySetting(
      category="HARM_CATEGORY_HATE_SPEECH",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_DANGEROUS_CONTENT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_SEXUALLY_EXPLICIT",
      threshold="OFF"
    ),types.SafetySetting(
      category="HARM_CATEGORY_HARASSMENT",
      threshold="OFF"
    )],
    thinking_config=types.ThinkingConfig(
      thinking_budget=-1,
    ),
  )

  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    print(chunk.text, end="")

generate()