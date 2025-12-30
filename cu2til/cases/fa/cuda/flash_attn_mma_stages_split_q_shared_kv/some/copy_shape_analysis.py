from google import genai
from google.genai import types
import base64

def generate():
  client = genai.Client(
      vertexai=True,
      project="totemic-fact-469314-n2",
      location="global",
  )

  msg1_text1 = types.Part.from_text(text="""IndexInt load_gmem_V_Bc = tile_K_seqlen * Bc + load_smem_V_Bc;
   IndexInt load_gmem_V_d = load_smem_V_d;
   IndexInt load_gmem_V_addr = V_gmem_offset + load_gmem_V_Bc * kHeadDim + load_gmem_V_d;
   SmemAddr load_smem_V_addr = smem_V_base_addr + (V_tile_size + load_smem_V_Bc * kHeadDim + load_smem_V_d) * sizeof(fp16);
   serial_range_for(i, 0, (kHeadDim / (kNumThreads / Bc)), 8) {
    warp_copy_async<fp16, 128, CopyTask::G2S>(&V[load_gmem_V_addr + i], load_smem_V_addr + i * sizeof(fp16));
   }
   warp_copy_async_commit_group();
这一段代码拷贝的数据tile_shape是多大，请用(a, b)表示""")

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