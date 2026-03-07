from pathlib import Path
from google import genai
from google.genai import types
import base64
import os
import dotenv
dotenv.load_dotenv()
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY"),
)
def generate(folder, test_name="./"):
  if folder is None:
    return
  folder = Path(folder)
  test_path = Path(test_name)
  if not folder.exists():
    return
  with open(f"{folder / test_path}/kernel_cleaned.cu", "r") as f:
    code = f.read()
  with open(f"{folder}/prompt/macro2code.md", "r") as f:
    prompt = f.read()
  msg1_text1 = types.Part.from_text(text=f"""{prompt}
```cuda
{code}
```
  """)

  model = "gemini-2.5-flash-lite"
  contents = [
    types.Content(
      role="user",
      parts=[
        msg1_text1
      ]
    ),
  ]

  generate_content_config = types.GenerateContentConfig(
    temperature = 0.35,
    top_p = 0.95,
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
  print(contents)
  res = ""
  for chunk in client.models.generate_content_stream(
    model = model,
    contents = contents,
    config = generate_content_config,
    ):
    res += chunk.text
    print(chunk.text, end="")
  # 提取代码块中的CUDA/C++代码
  import re
  
  # 查找```cuda或```cpp代码块
  code_pattern = r'```(?:cuda|cpp|c\+\+)\n(.*?)\n```'
  matches = re.findall(code_pattern, res, re.DOTALL | re.IGNORECASE)
  
  if matches:
    # 如果找到代码块，使用第一个匹配的代码块
    extracted_code = matches[0]
    print(f"\n找到 {len(matches)} 个代码块，使用第一个")
  else:
    # 如果没有找到特定的代码块，尝试查找任何```代码块
    general_pattern = r'```\w*\n(.*?)\n```'
    general_matches = re.findall(general_pattern, res, re.DOTALL)
    if general_matches:
      extracted_code = general_matches[0]
      print(f"\n找到 {len(general_matches)} 个通用代码块，使用第一个")
    else:
      # 如果完全没有代码块，使用原始响应
      extracted_code = res
      print("\n未找到代码块，使用原始响应")
  
  res = extracted_code

  with open(folder / test_path / "kernel_expanded.cu", "w") as f:
    f.write(res)

generate("/workspace/test_cu2til", "hgemm_mma_m16n8k16_mma2x4_warp4x4_stages")