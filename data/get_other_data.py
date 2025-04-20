# import json
# import pandas as pd
# from nba_api.stats.endpoints import boxscoretraditionalv2, boxscoreadvancedv2

# # 设置 Game ID
# game_id = '0022300013'  # 替换成你的 Game ID

# # 获取比赛基础数据
# boxscore_basic = boxscoretraditionalv2.BoxScoreTraditionalV2(game_id=game_id)
# df_basic = boxscore_basic.get_data_frames()[0]

# # 获取比赛高级数据
# boxscore_advanced = boxscoreadvancedv2.BoxScoreAdvancedV2(game_id=game_id)
# df_advanced = boxscore_advanced.get_data_frames()[0]

# # 保存基础数据
# basic_json_file = "game_basic_data.json"
# basic_csv_file = "game_basic_data.csv"
# df_basic.to_json(basic_json_file, orient='records', indent=4)
# df_basic.to_csv(basic_csv_file, index=False, encoding='utf-8')

# # 保存高级数据
# advanced_json_file = "game_advanced_data.json"
# advanced_csv_file = "game_advanced_data.csv"
# df_advanced.to_json(advanced_json_file, orient='records', indent=4)
# df_advanced.to_csv(advanced_csv_file, index=False, encoding='utf-8')

# print(f"✅ 基础数据已保存: {basic_json_file}, {basic_csv_file}")
# print(f"✅ 高级数据已保存: {advanced_json_file}, {advanced_csv_file}")
import pandas as pd
import json

# 文件路径
file_path = "/Users/kehangchen/Library/Mobile Documents/com~apple~CloudDocs/All_code/shotershub/data/NBA_2024_Shots.csv"
output_json_path = "shot_location.json"
# 读取 CSV 文件
df = pd.read_csv(file_path)

# 筛选 GAME_ID 为 22300003 的数据
filtered_df = df[df['GAME_ID'] == 22300003]

# 转换为 JSON 格式并保存
filtered_df.to_json(output_json_path, orient='records', indent=4)

print(f"✅ 筛选数据已保存为 JSON 文件: {output_json_path}")
