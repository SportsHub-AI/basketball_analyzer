from matplotlib.patches import Circle, Rectangle, Arc
import matplotlib.pyplot as plt
import json
import numpy as np
from floodlight.core.xy import XY
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap

FEET_TO_M  = 0.3048

def draw_court(ax=None, color='black', lw=2, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    # Create the various parts of an NBA basketball court

    # Create the basketball hoop
    # Diameter of a hoop is 18" so it has a radius of 9", which is a value
    # 7.5 in our coordinate system
    hoop = Circle((0, 0), radius=7.5 * FEET_TO_M, linewidth=lw, color=color, fill=False)

    # Create backboard
    backboard = Rectangle((-30 * FEET_TO_M, -7.5 * FEET_TO_M), 60 * FEET_TO_M, -1, linewidth=lw, color=color)

    # The paint
    # Create the outer box 0f the paint, width=16ft, height=19ft
    outer_box = Rectangle((-24.5, -15.75), 49, 58, linewidth=lw, color=color,
                          fill=False)
    # Create the inner box of the paint, widt=12ft, height=19ft
    inner_box = Rectangle((-36.75/2, -15.75), 36.75, 58, linewidth=lw, color=color,
                          fill=False)

    # Create free throw top arc
    top_free_throw = Arc((0, 42.25), 36.75, 36.75, theta1=0, theta2=180,
                         linewidth=lw, color=color, fill=False)
    # Create free throw bottom arc
    bottom_free_throw = Arc((0, 42.25), 36.75, 36.75, theta1=180, theta2=0,
                            linewidth=lw, color=color, linestyle='dashed')
    # Restricted Zone, it is an arc with 4ft radius from center of the hoop
    restricted = Arc((0, 0), 25, 25, theta1=0, theta2=180, linewidth=lw,
                     color=color)

    # Three point line
    # Create the side 3pt lines, they are 14ft long before they begin to arc
    corner_three_a = Rectangle((-66, -15.75), 0, 29.89, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((66, -15.75), 0, 29.89, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 135, 135, theta1=11.83, theta2=168.17, linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 124.25), 36, 36, theta1=180, theta2=0,
                           linewidth=lw, color=color)
    center_inner_arc = Arc((0, 124.25), 18, 18, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, inner_box, top_free_throw,
                      bottom_free_throw, restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc,
                      center_inner_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-75, -15.75), 150, 140, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)

    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)

    return ax

'''
inner_circle = Arc((0, 0), 80, 80, theta1=0, theta2=180, linewidth=lw,
                     color=color)
outer_circle = Circle((0, 0), radius=163.42, linewidth=lw, color=color, fill=False)

'''

def generate_gaussian_movement(n_points=1000):
    """
    生成篮球场上的球员移动数据点
    
    参数:
    n_points: int, 生成的数据点总数
    
    返回:
    x_coords: numpy array, x坐标数组
    y_coords: numpy array, y坐标数组
    """
    areas = {
        '左侧三分角': {
            'center': (-200, 20), 
            'std': (30, 30), 
            'weight': 0.15
        },
        '右侧三分角': {
            'center': (200, 20), 
            'std': (30, 30), 
            'weight': 0.15
        },
        '左侧中距离': {
            'center': (-150, 120), 
            'std': (30, 30), 
            'weight': 0.1
        },
        '右侧中距离': {
            'center': (150, 120), 
            'std': (30, 30), 
            'weight': 0.1
        },
        '罚球线': {
            'center': (0, 140), 
            'std': (30, 30), 
            'weight': 0.15
        },
        '篮下': {
            'center': (0, 20), 
            'std': (30, 30), 
            'weight': 0.2
        },
        '顶弧': {
            'center': (0, 250), 
            'std': (30, 30), 
            'weight': 0.1
        },
        '左翼': {
            'center': (-120, 200), 
            'std': (30, 30), 
            'weight': 0.025
        },
        '右翼': {
            'center': (120, 200), 
            'std': (30, 30), 
            'weight': 0.025
        }
    }
    
    x_coords = []
    y_coords = []
    
    for area, params in areas.items():
        points = int(n_points * params['weight'])
        while True:
            x = np.random.normal(params['center'][0], params['std'][0], points)
            y = np.random.normal(params['center'][1], params['std'][1], points)
            
            # 过滤场外点
            mask = (abs(x) < 240) & (y > -40) & (y < 420)
            x = x[mask]
            y = y[mask]
            
            if len(x) >= points * 0.9:
                x_coords.extend(x[:points])
                y_coords.extend(y[:points])
                break
            
    return np.array(x_coords), np.array(y_coords)

def plot_basketball_heatmap(
    x_coords,
    y_coords,
    player_name=None,
    ax=None,
    alpha=0.7,
    pixel_size=1
):
    """
    绘制篮球场上的球员热力图
    
    参数:
    x_coords: array-like, x坐标数组
    y_coords: array-like, y坐标数组
    player_name: str, 球员名字（可选）
    ax: matplotlib axes对象（可选）
    cmap: str, 热力图颜色方案
    alpha: float, 透明度 (0-1)
    pixel_size: int, 像素大小（影响热力图精度）
    
    返回:
    fig: matplotlib figure对象（如果创建了新图）
    """

        # 创建自定义的白色到橙色渐变
    colors = ['white', '#ff8400']
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    # 如果没有传入ax，创建新的图和子图
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 11))
    
    # 1) 绘制球场
    draw_court(ax=ax, outer_lines=True)
    
    # 2) 创建网格
    x = np.arange(-250, 250, pixel_size)
    y = np.arange(-47.5, 422.5, pixel_size)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()])
    
    # 3) 使用KDE（核密度估计）拟合球员位置分布
    try:
        kde = gaussian_kde(np.vstack([x_coords, y_coords]))
        z = kde(positions)
        density = z.reshape(xx.shape)
        
        # 4) 绘制热力图
        im = ax.imshow(
            density,
            extent=(-250, 250, -47.5, 422.5),
            origin='lower',
            cmap=cmap,
            alpha=alpha
        )
        
        # 设置密度范围
        im.set_clim(vmin=0, vmax=density.max())
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Density')
        
    except np.linalg.LinAlgError:
        print("警告：数据点过少或分布过于集中，无法生成有效的密度估计")
        return
    
    # 5) 设置图表属性
    ax.set_xlim(-250, 250)
    ax.set_ylim(500, -100)
    ax.set_aspect('equal')
    ax.axis('off')
        
    plt.show()
    
    # 如果函数内部创建了图，则返回
    if ax is None:
        return fig

def plot_shot_map(
    shooting_data,
    ax=None
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 11))

    # 画投篮点
    x_coords_scaled = []
    y_coords_scaled = []
    made_x = []
    made_y = []
    missed_x = []
    missed_y = []

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
        x_scaled = shot["LOC_X"] * scale  * FEET_TO_M
        y_scaled = shot["LOC_Y"] * scale  * FEET_TO_M

        
        x_coords_scaled.append(x_scaled)
        y_coords_scaled.append(y_scaled)

        if shot["EVENT_TYPE"] == "Missed Shot":
            missed_x.append(x_scaled)
            missed_y.append(y_scaled)
        else:
            made_x.append(x_scaled)
            made_y.append(y_scaled)


    colors = ['red' if shot["EVENT_TYPE"] == "Missed Shot" else 'green' for shot in shooting_data]

    # 转换为 numpy 数组
    x_coords_scaled = np.array(x_coords_scaled)
    y_coords_scaled = np.array(y_coords_scaled) 

    draw_court(ax=ax, outer_lines=True)

    # 绘制命中的投篮（圆点）
    ax.scatter(
        made_x, 
        made_y,
        c='green',          
        alpha=0.7,         
        edgecolors='none',
        marker='o',   # 圆点标记
        s=300
    )
    
    # 绘制未命中的投篮（X标记）
    ax.scatter(
        missed_x, 
        missed_y,
        c='red',          
        alpha=0.7,         
        edgecolors='none',
        marker='x',    # X标记
        s=300,         # 增大X标记的大小，使其更明显
        linewidths=5
    )


    ax.set_xlim(-75, 75)
    ax.set_ylim(124.25, -15.75)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.show()

# 读取本地 JSON 文件
with open("./data/MIN_WAS_22300003/shot_location.json", "r", encoding="utf-8") as file:
    data = json.load(file)

# 过滤出 GAME_ID=22300003 的数据
shooting_data = [shot for shot in data if shot["GAME_ID"] == 22300003]

n_points = 10000
color_options = ['red', 'blue']
colors_named = np.random.choice(color_options, size=n_points)

x_coords, y_coords = generate_gaussian_movement(n_points)

plot_shot_map(shooting_data)
#plot_basketball_heatmap(x_coords, y_coords, player_name="Apine")