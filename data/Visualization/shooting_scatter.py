from matplotlib.patches import Circle, Rectangle, Arc, Wedge
import matplotlib.pyplot as plt
import json
import numpy as np
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
import math
from shapely.geometry import Polygon, Point, box
from matplotlib.patches import Polygon as MplPolygon
from shapely.ops import unary_union

FEET_TO_M  = 0.3048

def draw_court_zones(ax=None, color='black', lw=4, outer_lines=False):
    # If an axes object isn't provided to plot onto, just get current one
    if ax is None:
        ax = plt.gca()

    #-------------------------Draw the court elements-------------------------
    court_rect = box(-75, -15.75, 75, 124.25)
    #ax.add_patch(MplPolygon(list(court_rect.exterior.coords), facecolor='skyblue', edgecolor='white', linewidth=0.5, alpha=0.6))

    inner_circle = Point(0, 0).buffer(24.5)
    outer_circle = Point(0, 0).buffer((24.5**2 + 42.25**2)**0.5)

    # Draw Zone 14 (inner circle)
    patch_inner_circle = MplPolygon(list(inner_circle.exterior.coords), facecolor='#ffcad4', edgecolor='white', linewidth=0.5, alpha=0.6)
    ax.add_patch(patch_inner_circle)

    #patch_outer_circle = MplPolygon(list(outer_circle.exterior.coords), facecolor='skyblue', edgecolor='white', linewidth=0.5, alpha=0.6)
    #ax.add_patch(patch_outer_circle)

    # 创建弧顶扇形
    theta1 = np.arccos(66/67.5)
    theta2 = np.pi - theta1
    theta = np.linspace(theta1, theta2, 100)

    arc = np.column_stack((
        67.5 * np.cos(theta),
        67.5 * np.sin(theta)
    ))

    pt2_wedge = Polygon(np.vstack([(66, -15.75), arc, (-66, -15.75)]))

    pt2_rect = box(-66, -15.75, 66, (67.5**2 - 66**2)**0.5)

    level3 = pt2_rect.union(pt2_wedge).difference(outer_circle)

    # ------------------- Draw level 3 area. Out of level 3 area is the last area of the court---------------------
    '''
    if not level3.is_empty:
        if level3.geom_type == 'Polygon':
            level3_patch = MplPolygon(list(level3.exterior.coords), facecolor='lightgreen', edgecolor='white', linewidth=0.5, alpha=0.5)
            ax.add_patch(level3_patch)
        elif level3.geom_type == 'MultiPolygon':
            for poly in level3.geoms:
                patch = MplPolygon(list(poly.exterior.coords), facecolor='lightgreen', edgecolor='white', linewidth=0.5, alpha=0.5)
                ax.add_patch(patch)
    '''
    
    theta_edges = np.linspace(theta1 + np.radians(20), theta2 - np.radians(20), 4)  # Select 3 points

    # Zone 7, 8, 9
    zones = []
    for i in range(3):
        theta_start = theta_edges[i]
        theta_end = theta_edges[i+1]

        arc_outer = np.column_stack((
            67.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
            67.5 * np.sin(np.linspace(theta_start, theta_end, 50))
        ))
        arc_inner = np.column_stack((
            42.25 * np.cos(np.linspace(theta_end, theta_start, 50)),  # 注意逆序
            42.25 * np.sin(np.linspace(theta_end, theta_start, 50))
        ))

        ring_sector = Polygon(np.vstack([arc_outer, arc_inner]))
        zone = ring_sector.intersection(level3)
        zones.append(zone)

    for i, zone in enumerate(zones):
        if not zone.is_empty:
            if zone.geom_type == 'Polygon':
                ax.add_patch(MplPolygon(list(zone.exterior.coords), facecolor="#ffcad4", alpha=0.7, edgecolor='white', linewidth=0.5))
            elif zone.geom_type == 'MultiPolygon':
                for poly in zone.geoms:
                    ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor="#ffcad4", alpha=0.7, edgecolor='white', linewidth=0.5))
    
    # Draw Zone 6
    theta_start = theta_edges[-1]
    theta_end = np.radians(270)

    arc_outer = np.column_stack((
        67.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
        67.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        42.25 * np.cos(np.linspace(theta_end, theta_start, 50)),
        42.25 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))
    left_sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone6 = left_sector.intersection(level3)

    # === 绘制左边额外扇形区域 ===
    if not zone6.is_empty:
        if zone6.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone6.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
        elif zone6.geom_type == 'MultiPolygon':
            for poly in zone6.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
    
    
    
    # Draw Zone 10
    theta_start = np.radians(-90)
    theta_end =  theta_edges[0]

    arc_outer = np.column_stack((
        67.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
        67.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        42.25 * np.cos(np.linspace(theta_end, theta_start, 50)),
        42.25 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))
    right_sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone10 = right_sector.intersection(level3)

    # === 绘制左边额外扇形区域 ===
    if not zone10.is_empty:
        if zone10.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone10.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
        elif zone10.geom_type == 'MultiPolygon':
            for poly in zone10.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
    

    


    # Draw level 4
    court_outline = box(-75, -15.75, 75, 124.25)
    level4 = court_outline.difference(pt2_rect.union(pt2_wedge))

    # Zone 1 and 5
    zone1 = box(-75, -15.75, -66, (67.5**2 - 66**2)**0.5) # left_pt3_rect
    zone5 = box(66, -15.75, 75, (67.5**2 - 66**2)**0.5) # right_pt3_rect

    ax.add_patch(MplPolygon(list(zone1.exterior.coords), facecolor='#ffcad4', edgecolor='white', linewidth=0.5, alpha=0.5))
    ax.add_patch(MplPolygon(list(zone5.exterior.coords), facecolor='#ffcad4', edgecolor='white', linewidth=0.5, alpha=0.5))

    level4_main = level4.difference(zone1.union(zone5))

    # 角度范围定义（根据 full semicircle，约 180°）
    theta1 = np.radians(73)      # 右边
    theta2 = np.radians(107)      # 左边

    # 分成三段（4个分界点）
    theta_edges = np.linspace(theta1, theta2, 2)

    zone3_color = '#ffcad4'

    # Draw Zone 3
    theta_start = theta_edges[0]
    theta_end = theta_edges[1]

    arc_outer = np.column_stack((
        (75 ** 2 + 124.25 ** 2) ** 0.5 * np.cos(np.linspace(theta_start, theta_end, 50)),   # court半径更大
        (75 ** 2 + 124.25 ** 2) ** 0.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        67.5 * np.cos(np.linspace(theta_end, theta_start, 50)),   # 从内到外逆转
        67.5 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))
    ring_sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone3 = ring_sector.intersection(level4_main)

    if not zone3.is_empty:
        if zone3.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone3.exterior.coords), facecolor=zone3_color, alpha=0.6, edgecolor='white', linewidth=0.5))
        elif zone3.geom_type == 'MultiPolygon':
            for poly in zone3.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor=zone3_color, alpha=0.6, edgecolor='white', linewidth=0.5))


    # Draw Zone 2
    theta_start = theta_edges[1]
    theta_end =  np.radians(270)

    arc_outer = np.column_stack((
        (75 ** 2 + 124.25 ** 2) ** 0.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
        (75 ** 2 + 124.25 ** 2) ** 0.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        67.5 * np.cos(np.linspace(theta_end, theta_start, 50)),
        67.5 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))
    left_sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone2 = left_sector.intersection(level4_main)

    # === 绘制左边额外扇形区域 ===
    if not zone2.is_empty:
        if zone2.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone2.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
        elif zone2.geom_type == 'MultiPolygon':
            for poly in zone2.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))

    # Draw Zone 4
    theta_start = np.radians(-90)
    theta_end =  theta_edges[0]

    arc_outer = np.column_stack((
        (75 ** 2 + 124.25 ** 2) ** 0.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
        (75 ** 2 + 124.25 ** 2) ** 0.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        67.5 * np.cos(np.linspace(theta_end, theta_start, 50)),
        67.5 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))
    right_sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone4 = right_sector.intersection(level4_main)

    # === 绘制左边额外扇形区域 ===
    if not zone4.is_empty:
        if zone4.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone4.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
        elif zone4.geom_type == 'MultiPolygon':
            for poly in zone4.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))


    colors = ['#a9def9', '#e4c1f9', '#ff99c8']
    level2_ring = outer_circle.difference(inner_circle)
    
    # Draw Zone 11
    theta_start = np.radians(115)
    theta_end = np.radians(270)

    arc_outer = np.column_stack((
        67.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
        67.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        24.5 * np.cos(np.linspace(theta_end, theta_start, 50)),  # 注意逆序
        24.5 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))

    sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone11 = sector.intersection(level2_ring)

    if not zone11.is_empty:
        if zone11.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone11.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
        elif zone11.geom_type == 'MultiPolygon':
            for poly in zone11.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))

    # Draw Zone 12
    theta_start = np.radians(65)
    theta_end = np.radians(115)

    arc_outer = np.column_stack((
        67.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
        67.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        24.5 * np.cos(np.linspace(theta_end, theta_start, 50)),  # 注意逆序
        24.5 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))

    sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone12 = sector.intersection(level2_ring)

    if not zone12.is_empty:
        if zone12.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone12.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
        elif zone12.geom_type == 'MultiPolygon':
            for poly in zone12.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))

    # Draw Zone 13
    theta_start = np.radians(-90)
    theta_end = np.radians(65)

    arc_outer = np.column_stack((
        67.5 * np.cos(np.linspace(theta_start, theta_end, 50)),
        67.5 * np.sin(np.linspace(theta_start, theta_end, 50))
    ))
    arc_inner = np.column_stack((
        24.5 * np.cos(np.linspace(theta_end, theta_start, 50)),  # 注意逆序
        24.5 * np.sin(np.linspace(theta_end, theta_start, 50))
    ))

    sector = Polygon(np.vstack([arc_outer, arc_inner]))
    zone13 = sector.intersection(level2_ring)

    if not zone13.is_empty:
        if zone13.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(zone13.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
        elif zone13.geom_type == 'MultiPolygon':
            for poly in zone13.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor='#ffcad4', alpha=0.7, edgecolor='white', linewidth=0.5))
    zones = {
        "zone1": zone1,
        "zone2": zone2,
        "zone3": zone3,
        "zone4": zone4,
        "zone5": zone5,
        "zone6": zone6,
        "zone7": zones[0],
        "zone8": zones[1],
        "zone9": zones[2],
        "zone10": zone10,
        "zone11": zone11,
        "zone12": zone12,
        "zone13": zone13,
        "zone14": inner_circle,
    }

    return ax, zones

def draw_court(ax, color='black', lw=2, outer_lines=False):
    # Create the various parts of an FIBA basketball court

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
    corner_three_a = Rectangle((-66, -15.75), 0, 15.75 + (67.5**2 - 66**2)**0.5, linewidth=lw,
                               color=color)
    corner_three_b = Rectangle((66, -15.75), 0, 15.75 + (67.5**2 - 66**2)**0.5, linewidth=lw, color=color)
    # 3pt arc - center of arc will be the hoop, arc is 23'9" away from hoop
    # I just played around with the theta values until they lined up with the 
    # threes
    three_arc = Arc((0, 0), 135, 135, theta1=math.degrees(np.arccos(66/67.5)), theta2=180-math.degrees(np.arccos(66/67.5)), linewidth=lw,
                    color=color)

    # Center Court
    center_outer_arc = Arc((0, 124.25), 36, 36, theta1=180, theta2=0,
                           linewidth=lw, color=color)

    # List of the court elements to be plotted onto the axes
    court_elements = [hoop, backboard, outer_box, top_free_throw,
                      restricted, corner_three_a,
                      corner_three_b, three_arc, center_outer_arc]

    if outer_lines:
        # Draw the half court line, baseline and side out bound lines
        outer_lines = Rectangle((-75, -15.75), 150, 140, linewidth=lw,
                                color=color, fill=False)
        court_elements.append(outer_lines)


    # Add the court elements onto the axes
    for element in court_elements:
        ax.add_patch(element)


'''
inner_circle = Arc((0, 0), 24.5, 24.5, theta1=0, theta2=180, linewidth=lw,
                     color=color)
outer_circle = Circle((0, 0), radius=(24.5**2 + 42.25**2)**0.5, linewidth=lw, color=color, fill=False)

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

    x_coords = np.array(x_coords) * FEET_TO_M
    y_coords = np.array(y_coords) * FEET_TO_M
            
    return np.array(x_coords), np.array(y_coords)

def plot_basketball_heatmap(
    x_coords,
    y_coords,
    player_name=None,
    ax=None,
    alpha=0.7,
    pixel_size=0.5
):
    """
    绘制篮球场上的球员热力图（单位为米）
    """

    FEET_TO_M = 0.3048

    # 创建自定义的白色到橙色渐变色图
    colors = ['#f0f0f0', '#ff8400']
    cmap = LinearSegmentedColormap.from_list('custom', colors)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 11))

    # 1) 绘制球场（单位为米）
    draw_court(ax=ax, outer_lines=True)

    # 2) 创建网格（单位也转换为米）
    x_min, x_max = -250 * FEET_TO_M, 250 * FEET_TO_M
    y_min, y_max = -47.5 * FEET_TO_M, 422.5 * FEET_TO_M

    x = np.arange(x_min, x_max, pixel_size)
    y = np.arange(y_min, y_max, pixel_size)
    xx, yy = np.meshgrid(x, y)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # 3) 核密度估计
    try:
        kde = gaussian_kde(np.vstack([x_coords, y_coords]))
        z = kde(positions)
        density = z.reshape(xx.shape)

        # 4) 绘制热力图
        im = ax.imshow(
            density,
            extent=(x_min, x_max, y_min, y_max),
            origin='lower',
            cmap=cmap,
            alpha=alpha
        )

        im.set_clim(vmin=0, vmax=density.max())

    except np.linalg.LinAlgError:
        print("警告：数据点过少或分布过于集中，无法生成有效的密度估计")
        return

    # 5) 设置坐标轴范围（单位为米）
    ax.set_xlim(-75, 75)
    ax.set_ylim(124.25, -15.75)
    ax.set_aspect('equal')
    ax.axis('off')

    plt.show()

    if ax is None:
        return fig


def plot_shot_map(shooting_data, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 11))

    ax, zones = draw_court_zones(ax=ax, outer_lines=True)

    made_x, made_y = [], []
    missed_x, missed_y = [], []

    # 初始化每个区域的命中和总数
    zone_stats = {k: {'made': 0, 'total': 0} for k in zones}

    for shot in shooting_data:
        zone = shot["ZONE_RANGE"]
        if zone == "Less Than 8 ft.":
            scale = 3
        elif zone == "8-16 ft.":
            scale = 8
        elif zone == "24+ ft.":
            scale = 10
        else:
            scale = 5

        x = shot["LOC_X"] * scale * FEET_TO_M
        y = shot["LOC_Y"] * scale * FEET_TO_M
        p = Point(x, y)

        for name, shape in zones.items():
            if shape.contains(p):
                zone_stats[name]['total'] += 1
                if shot["EVENT_TYPE"] != "Missed Shot":
                    zone_stats[name]['made'] += 1
                break

        if shot["EVENT_TYPE"] == "Missed Shot":
            missed_x.append(x)
            missed_y.append(y)
        else:
            made_x.append(x)
            made_y.append(y)

    # 计算命中率并映射颜色
    zone_accuracy = {}
    for name, stat in zone_stats.items():
        total = stat['total']
        made = stat['made']
        acc = made / total if total > 0 else 0
        zone_accuracy[name] = acc

    # 设置颜色映射
    colors = ['white', '#ff8400']
    cmap = LinearSegmentedColormap.from_list('custom', colors)
    norm = plt.Normalize(vmin=0, vmax=1)

    for name, shape in zones.items():
        acc = zone_accuracy[name]
        color = cmap(norm(acc))
        if shape.geom_type == 'Polygon':
            ax.add_patch(MplPolygon(list(shape.exterior.coords), facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.8))
            centroid = shape.centroid
        elif shape.geom_type == 'MultiPolygon':
            for poly in shape.geoms:
                ax.add_patch(MplPolygon(list(poly.exterior.coords), facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.8))
            centroid = unary_union(shape).centroid
        else:
            continue

        # 文本位置微调
        dx = 0
        if name == "zone11":
            dx = -7  # 向左偏移
        elif name == "zone13":
            dx = 7   # 向右偏移

        total = zone_stats[name]['total']
        made = zone_stats[name]['made']

        if total > 0:
            percentage = f"{int(round(100 * acc))}%"
            summary = f"{made}/{total}"
            text = f"{percentage}\n{summary}"
        else:
            text = "No data"

        ax.text(
            centroid.x + dx, centroid.y,
            text,
            ha='center', va='center',
            fontsize=10, color='black', fontweight='bold'
        )

    # 绘制球场
    draw_court(ax=ax, color="#f0f0f0", lw=3, outer_lines=True)

    # 命中：绿圆
    ax.scatter(
        made_x,
        made_y,
        c='green',
        alpha=0.7,
        edgecolors='none',
        marker='o',
        s=300
    )

    # 未中：红叉
    ax.scatter(
        missed_x,
        missed_y,
        c='red',
        alpha=0.7,
        edgecolors='none',
        marker='x',
        s=300,
        linewidths=5
    )

    # 设置轴范围
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
plot_basketball_heatmap(x_coords, y_coords, player_name="Apine")