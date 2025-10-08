import os
import sys
import dotenv
import requests
import csv
import json

# --- 0. 切换到脚本所在目录 ---
# 获取脚本的绝对路径
script_dir = os.path.dirname(os.path.abspath(__file__))
# 切换到脚本所在目录
os.chdir(script_dir)
print(f"Current working directory: {os.getcwd()}")

# --- 1. 设置 ---
# 加载 .env 文件中的环境变量
dotenv.load_dotenv()

# 从环境变量中获取你的 Notion Integration Secret
INTERNAL_INTEGRATION_SECRET = os.getenv("INTERNAL_INTEGRATION_SECRET")

# 你的 Database ID
database_id = "284bc4b7a7db806098d2c7d4931b606a"

# API 请求头
headers = {
    "Authorization": f"Bearer {INTERNAL_INTEGRATION_SECRET}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28"  # 建议使用一个稳定的API版本
}

# --- 2. 辅助函数：从复杂的 Notion 属性对象中提取值 ---
def get_property_value(prop):
    """
    根据 Notion 属性的类型提取其值。
    """
    prop_type = prop.get('type')

    if prop_type == 'title':
        return prop['title'][0]['plain_text'] if prop['title'] else ''
    if prop_type == 'rich_text':
        return prop['rich_text'][0]['plain_text'] if prop['rich_text'] else ''
    if prop_type == 'number':
        return prop['number']
    if prop_type == 'select':
        return prop['select']['name'] if prop['select'] else ''
    if prop_type == 'multi_select':
        return ', '.join([item['name'] for item in prop['multi_select']])
    if prop_type == 'status':
        return prop['status']['name'] if prop['status'] else ''
    if prop_type == 'date':
        return prop['date']['start'] if prop['date'] else ''
    if prop_type == 'checkbox':
        return prop['checkbox']
    if prop_type == 'url':
        return prop['url']
    if prop_type == 'email':
        return prop['email']
    if prop_type == 'phone_number':
        return prop['phone_number']
    # 对于更复杂的类型如 'relation' 或 'formula'，你可能需要添加更多逻辑
    # 比如 formula 的值在 prop['formula'][prop['formula']['type']]
    if prop_type == 'formula':
        formula_data = prop.get('formula', {})
        formula_type = formula_data.get('type')
        if formula_type:
            return formula_data.get(formula_type)
        return ''
        
    return 'Unsupported Property Type'

# --- 3. 主函数：查询 Database 并处理所有页面 ---
def query_notion_database(db_id):
    """
    查询 Notion Database 并处理分页，返回所有页面的数据。
    """
    url = f"https://api.notion.com/v1/databases/{db_id}/query"
    all_results = []
    has_more = True
    start_cursor = None

    while has_more:
        # 构造请求体，如果有下一页，则包含 start_cursor
        payload = {}
        if start_cursor:
            payload['start_cursor'] = start_cursor

        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()  # 如果请求失败 (如 4xx or 5xx)，则抛出异常
            data = response.json()

            all_results.extend(data.get('results', []))
            
            has_more = data.get('has_more', False)
            start_cursor = data.get('next_cursor')

        except requests.exceptions.RequestException as e:
            print(f"Error querying Notion API: {e}")
            break
            
    return all_results

# --- 4. 运行和导出 ---
if __name__ == "__main__":
    print(f"正在从 Notion Database (ID: {database_id}) 读取数据...")
    
    # 获取所有页面数据
    pages = query_notion_database(database_id)

    if not pages:
        print("未能获取到任何数据，请检查你的 Database ID 和 Integration Secret 是否正确，以及 Integration 是否有权限访问该 Database。")
    else:
        print(f"成功获取到 {len(pages)} 行数据。")
        
        # 解析数据为简单的 key-value 格式
        processed_data = []
        for page in pages:
            row = {}
            # 遍历每个页面的所有属性
            for prop_name, prop_data in page['properties'].items():
                row[prop_name] = get_property_value(prop_data)
            processed_data.append(row)

        # 写入 CSV 文件
        if processed_data:
            # 使用第一行数据的键作为 CSV 的表头
            headers_csv = processed_data[0].keys()
            # 使用相对路径指向 price 目录下的 openrouter_api.csv
            output_file = '../price/openrouter_api.csv'
            
            # 确保目标目录存在
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)

            try:
                with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=headers_csv)
                    writer.writeheader()
                    writer.writerows(processed_data)
                print(f"数据已成功导出到文件: {os.path.abspath(output_file)}")
            except IOError as e:
                print(f"写入文件时发生错误: {e}")