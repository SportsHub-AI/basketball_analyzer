from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.pyplot as plt
import json
import numpy as np

# 读取本地 JSON 文件
with open("./data/MIN_WAS_22300003/shot_location.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 过滤出 GAME_ID=22300003 的数据
shooting_data = [shot for shot in data if shot["GAME_ID"] == 22300003]

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30, -7.5), 60, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-80, -47.5), 160, 190, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-60, -47.5), 120, 190, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 142.5), 120, 120, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 142.5), 120, 120, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-220, -47.5), 0, 140, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((220, -47.5), 0, 140, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 475, 475, theta1=22, theta2=158, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 422.5), 120, 120, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 422.5), 40, 40, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-250, -47.5), 500, 470, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

# 画投篮点
x_coords_scaled = []
y_coords_scaled = []

for shot in shooting_data:
    zone = shot["ZONE_RANGE"]
    
    # 定义缩放规则
    if zone == "Less Than 8 ft.":
        scale = 3
    elif zone == "8-16 ft.":
        scale = 8
    elif zone == "24+ ft.":
        scale = 10
    else:
        # 处理未定义区域（默认缩放因子）
        scale = 5  # 例如：16-24ft 区域
        print(f"警告：未知区域 {zone}，使用默认缩放因子 {scale}")

    # 应用缩放
    x_scaled = shot["LOC_X"] * scale
    y_scaled = shot["LOC_Y"] * scale
    
    x_coords_scaled.append(x_scaled)
    y_coords_scaled.append(y_scaled)

colors = ['red' if shot["EVENT_TYPE"] == "Missed Shot" else 'blue' for shot in shooting_data]

# 转换为 numpy 数组
x_coords_scaled = np.array(x_coords_scaled)
y_coords_scaled = np.array(y_coords_scaled)

plt.figure(figsize=(12,11))

plt.scatter(
    x_coords_scaled, 
    y_coords_scaled,
    c=colors,          
    alpha=0.7,         
    edgecolors='none'   
)
draw_court(outer_lines=True)
plt.xlim(300, -300)
plt.ylim(-100, 500)
plt.axis('off')
plt.title("Shot Chart - Made (Green) vs Missed (Red)")
plt.show()
