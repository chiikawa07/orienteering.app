import cv2
import numpy as np
import heapq
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import xml.etree.ElementTree as ET
from datetime import datetime

# ==========================================
# UI: ページ設定とカスタムCSS
# ==========================================
st.set_page_config(layout="wide", page_title="オリエンテーリングAI")

st.markdown("""
    <style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; max-width: 98%; }
    .stExpander { border: 1px solid #444; border-radius: 8px; }
    hr { margin-top: 0.5rem; margin-bottom: 0.5rem; }
    </style>
""", unsafe_allow_html=True)

def haversine_distance(p1, p2):
    R = 6371.0
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

def parse_time(time_str):
    if not time_str: return None
    time_str = time_str.replace('Z', '+00:00')
    try: return datetime.fromisoformat(time_str)
    except: return None

def parse_gpx_data(file_bytes):
    try:
        root = ET.fromstring(file_bytes)
        segments = []
        for trkseg in root.iter():
            if trkseg.tag.endswith('trkseg'):
                seg_points = []
                for pt in trkseg.iter():
                    if pt.tag.endswith('trkpt'):
                        lat = float(pt.attrib['lat'])
                        lon = float(pt.attrib['lon'])
                        time_str = None
                        for child in pt:
                            if child.tag.endswith('time'):
                                time_str = child.text
                                break
                        time_obj = parse_time(time_str)
                        seg_points.append((lat, lon, time_obj))
                if seg_points:
                    segments.append(seg_points)
        if not segments:
            pts = []
            for pt in root.iter():
                if pt.tag.endswith('rtept') or pt.tag.endswith('wpt'):
                    lat = float(pt.attrib['lat'])
                    lon = float(pt.attrib['lon'])
                    pts.append((lat, lon, None))
            if pts:
                segments.append(pts)
        return segments
    except Exception as e:
        st.error(f"GPXファイルの解析に失敗しました: {e}")
        return []

def get_color_for_pace(pace):
    if pace is None:
        return (0, 180, 255) 
    fast_pace = 4.0   
    slow_pace = 15.0  
    ratio = (pace - fast_pace) / (slow_pace - fast_pace)
    ratio = max(0.0, min(1.0, ratio)) 
    g = int(255 * (1 - ratio))
    r = int(255 * ratio)
    return (0, g, r) 

# ==========================================
# 画像処理をキャッシュ化
# ==========================================
@st.cache_data(show_spinner="AIが地図と地形を解析中...")
def process_map_data(file_bytes, scale, slope_weight, nav_weight):
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    h_s, w_s = small_img.shape[:2]

    Z = small_img.reshape((-1, 3))
    Z = np.float32(Z)
    K = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    labels_reshaped = label.reshape((h_s, w_s))

    isom_colors = {
        "white": np.array([245, 245, 245], dtype=np.uint8),
        "black": np.array([40, 40, 40], dtype=np.uint8),
        "yellow": np.array([80, 220, 240], dtype=np.uint8),
        "green": np.array([90, 180, 110], dtype=np.uint8),
        "brown": np.array([60, 130, 180], dtype=np.uint8),
        "blue": np.array([220, 120, 50], dtype=np.uint8)
    }
    isom_lab = {k: cv2.cvtColor(np.array([[v]]), cv2.COLOR_BGR2LAB)[0][0] for k, v in isom_colors.items()}
    center_lab = cv2.cvtColor(np.array([center]), cv2.COLOR_BGR2LAB)[0]

    masks = {k: np.zeros((h_s, w_s), dtype=np.uint8) for k in isom_colors.keys()}
    for i in range(K):
        c_lab = center_lab[i]
        min_dist = float('inf')
        closest_name = "white"
        for name, target_lab in isom_lab.items():
            dist = np.linalg.norm(np.float32(c_lab) - np.float32(target_lab))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        masks[closest_name][labels_reshaped == i] = 255

    mask_white = masks["white"]
    mask_black = masks["black"]
    mask_yellow = masks["yellow"]
    mask_green = masks["green"]
    mask_brown = masks["brown"]
    mask_blue = masks["blue"]

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_black_closed = cv2.morphologyEx(mask_black, cv2.MORPH_CLOSE, kernel_close)
    road_mask = np.zeros_like(mask_black)
    wall_mask = np.zeros_like(mask_black)

    contours, _ = cv2.findContours(mask_black_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1)
        mask_roi = mask_brown[y:y+h_rect, x:x+w_rect]
        if cv2.countNonZero(mask_roi) > 0:
            continue
        if area > 400 and ratio < 3.0:
            cv2.drawContours(wall_mask, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(road_mask, [cnt], -1, 255, -1)

    corners = cv2.goodFeaturesToTrack(road_mask, maxCorners=50, qualityLevel=0.1, minDistance=20)
    attack_points = []
    if corners is not None:
        for i in corners:
            cx, cy = i.ravel()
            attack_points.append((int(cy), int(cx)))

    brown_blur = cv2.GaussianBlur(mask_brown, (5, 5), 0)
    grad_x = cv2.Sobel(brown_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(brown_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_x = np.where(grad_mag > 0, grad_x / grad_mag, 0.0)
    grad_y = np.where(grad_mag > 0, grad_y / grad_mag, 0.0)
    
    slope_heatmap = cv2.GaussianBlur(mask_brown, (31, 31), 0) 
    slope_penalty = (slope_heatmap / 255.0) * 20.0

    inv_road = cv2.bitwise_not(road_mask)
    dist_to_road = cv2.distanceTransform(inv_road, cv2.DIST_L2, 5)
    dist_capped = np.clip(dist_to_road, 0, 80) 
    nav_penalty = (dist_capped / 80.0) * 3.0

    small_cost = np.full((h_s, w_s), 5.0)
    small_cost[mask_white > 0] = 1.0
    small_cost[mask_yellow > 0] = 0.8
    small_cost[mask_brown > 0] = 1.2
    small_cost[mask_green > 0] = 3.0
    small_cost[road_mask > 0] = 0.5
    
    small_cost = small_cost + slope_penalty + nav_penalty
    small_cost[wall_mask > 0] = 9999
    small_cost[mask_blue > 0] = 9999

    # 白フチ（余白）をAIが道と勘違いしないように、壁を厚く設定
    margin = 40 
    small_cost[0:margin, :] = 9999
    small_cost[-margin:, :] = 9999
    small_cost[:, 0:margin] = 9999
    small_cost[:, -margin:] = 9999

    return small_img, h_s, w_s, road_mask, attack_points, grad_x, grad_y, grad_mag, dist_capped, small_cost

def dijkstra(cost_map, gx_mat, gy_mat, g_mag, start, goal):
    h, w = cost_map.shape
    dist = np.full((h, w), np.inf)
    prev = np.full((h, w, 2), -1)
    dist[start] = 0
    pq = [(0, start)]
    directions = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]
    c_weight = 50.0

    while pq:
        d, (y,x) = heapq.heappop(pq)
        if (y,x) == goal: break
        for dy, dx in directions:
            ny, nx = y+dy, x+dx
            if 0 <= ny < h and 0 <= nx < w:
                if cost_map[ny,nx] >= 9999: continue
                move_weight = 1.414 if (dy != 0 and dx != 0) else 1.0
                base_step_cost = cost_map[ny,nx]
                if g_mag[ny, nx] > 10:
                    move_len = np.sqrt(dy**2 + dx**2)
                    mdy, mdx = dy / move_len, dx / move_len
                    dot_product = abs(mdy * gy_mat[ny, nx] + mdx * gx_mat[ny, nx])
                    base_step_cost += dot_product * c_weight

                nd = d + (base_step_cost * move_weight)
                if nd < dist[ny,nx]:
                    dist[ny,nx] = nd
                    prev[ny,nx] = [y,x]
                    heapq.heappush(pq, (nd, (ny,nx)))

    path = []
    cur = goal
    while tuple(cur) != tuple(start):
        path.append(cur)
        cur = prev[cur[0], cur[1]]
        if cur[0] == -1: break
    path.append(start)
    return path[::-1]

# ==========================================
# メインUI： Livelox風カラムレイアウト
# ==========================================
col_panel, col_map = st.columns([1, 3])

with col_panel:
    st.markdown("### 🧭 Livelox風 解析AI")
    
    with st.expander("📂 地図とGPSの読み込み", expanded=True):
        uploaded_file = st.file_uploader("地図画像 (必須)", type=["png", "jpg", "jpeg"])
        gpx_file = st.file_uploader("GPSログ (.gpx)", type=["gpx"])

if uploaded_file is not None:
    gpx_segments = []
    total_gpx_dist = 0.0
    total_pts = 0
    if gpx_file is not None:
        gpx_segments = parse_gpx_data(gpx_file.read())
        for seg in gpx_segments:
            total_pts += len(seg)
            for i in range(len(seg) - 1):
                total_gpx_dist += haversine_distance((seg[i][0], seg[i][1]), (seg[i+1][0], seg[i+1][1]))

    scale = 0.35
    file_bytes = bytes(uploaded_file.read())
    (small_img, h_s, w_s, road_mask, attack_points, grad_x, grad_y, grad_mag, dist_capped, small_cost) = process_map_data(file_bytes, scale, 20.0, 3.0)

    if 'start_nx' not in st.session_state:
        st.session_state.start_nx, st.session_state.start_ny = 0.53, 0.77
        st.session_state.goal_nx, st.session_state.goal_ny = 0.70, 0.50
    if 'last_click' not in st.session_state:
        st.session_state.last_click = None

    margin = 40
    sx = max(margin, min(int(st.session_state.start_nx * w_s), w_s - margin - 1))
    sy = max(margin, min(int(st.session_state.start_ny * h_s), h_s - margin - 1))
    gx = max(margin, min(int(st.session_state.goal_nx * w_s), w_s - margin - 1))
    gy = max(margin, min(int(st.session_state.goal_ny * h_s), h_s - margin - 1))
    start, goal = (sy, sx), (gy, gx)

    routes, metrics = [], []
    colors = [(0, 0, 255), (255, 0, 0), (0, 128, 0)]
    
    if small_cost[start] < 9999 and small_cost[goal] < 9999:
        path1 = dijkstra(small_cost, grad_x, grad_y, grad_mag, start, goal)
        if path1 and len(path1) > 1:
            routes.append(path1)
            metrics.append({"名前": "AI 最適解", "色": "🔴 赤", "スコア": round(sum(small_cost[p[0], p[1]] for p in path1), 1)})

        best_ap = min(attack_points, key=lambda p: np.hypot(p[0]-goal[0], p[1]-goal[1])) if attack_points else None
        if best_ap:
            path_to_ap = dijkstra(small_cost, grad_x, grad_y, grad_mag, start, best_ap)
            path_from_ap = dijkstra(small_cost, grad_x, grad_y, grad_mag, best_ap, goal)
            if path_to_ap and path_from_ap:
                path2 = path_to_ap[:-1] + path_from_ap
                routes.append(path2)
                metrics.append({"名前": "AP 経由", "色": "🔵 青", "スコア": round(sum(small_cost[p[0], p[1]] for p in path2), 1)})

        if path1:
            search_cost = small_cost.copy()
            for p in path1:
                y_min, y_max = max(0, p[0]-4), min(h_s, p[0]+5)
                x_min, x_max = max(0, p[1]-4), min(w_s, p[1]+5)
                search_cost[y_min:y_max, x_min:x_max] += 25.0
            path3 = dijkstra(search_cost, grad_x, grad_y, grad_mag, start, goal)
            if path3 and len(path3) > 1:
                routes.append(path3)
                metrics.append({"名前": "大穴ルート", "色": "🟢 緑", "スコア": round(sum(small_cost[p[0], p[1]] for p in path3), 1)})

    vis = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
    h_orig, w_orig = vis.shape[:2]
    scale_inv = 1 / scale

    orig_start = (int(st.session_state.start_nx * w_orig), int(st.session_state.start_ny * h_orig))
    orig_goal = (int(st.session_state.goal_nx * w_orig), int(st.session_state.goal_ny * h_orig))

    for ap in attack_points:
        cv2.circle(vis, (int(ap[1] * scale_inv), int(ap[0] * scale_inv)), 4, (0, 255, 255), -1)
    if best_ap:
        cv2.circle(vis, (int(best_ap[1] * scale_inv), int(best_ap[0] * scale_inv)), 15, (255, 255, 0), 4)

    with col_panel:
        st.radio("📌 地図をクリックして移動:", ["🔵 スタート", "🔴 ゴール"], key="point_type")
        
        st.markdown("### 🏃‍♂️ 競技者 (ルート比較)")
        for m in metrics:
            st.markdown(f"**{m['色']} : {m['名前']}**<br>難易度スコア: {m['スコア']}", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
        
        if gpx_segments:
            st.markdown(f"**🏃‍♂️ あなたの実走 (速度色分け)**<br>距離: {round(total_gpx_dist, 2)} km", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
            
            with st.expander("⚙️ GPS位置合わせ", expanded=True):
                st.write("※軌跡が自動で地図の大きさにフィットしています。微調整してください。")
                gpx_scale = st.slider("拡大率", 0.1, 5.0, 1.0, step=0.05)
                gpx_rot = st.slider("回転角度", -180, 180, 0, step=1)
                gpx_offset_x = st.slider("左右移動 (X)", -2000, 2000, 0, step=10)
                gpx_offset_y = st.slider("上下移動 (Y)", -2000, 2000, 0, step=10)
        else:
            gpx_scale, gpx_rot, gpx_offset_x, gpx_offset_y = 1.0, 0, 0, 0

    if gpx_segments:
        all_lats = [p[0] for seg in gpx_segments for p in seg]
        all_lons = [p[1] for seg in gpx_segments for p in seg]
        center_lat = (min(all_lats) + max(all_lats)) / 2
        center_lon = (min(all_lons) + max(all_lons)) / 2

        avg_lat_rad = np.radians(center_lat)
        m_per_deg_lat = 111320.0
        m_per_deg_lon = 40075000.0 * np.cos(avg_lat_rad) / 360.0

        lat_range = max(all_lats) - min(all_lats)
        lon_range = max(all_lons) - min(all_lons)
        
        track_w_m = lon_range * m_per_deg_lon
        track_h_m = lat_range * m_per_deg_lat
        max_track_dim = max(track_w_m, track_h_m) if max(track_w_m, track_h_m) > 0 else 1.0

        target_px = min(w_orig, h_orig) * 0.8
        pixels_per_meter = target_px / max_track_dim

        for seg in gpx_segments:
            gpx_pixels = []
            for lat, lon, time_obj in seg:
                dx_m = (lon - center_lon) * m_per_deg_lon
                dy_m = -(lat - center_lat) * m_per_deg_lat
                dx = dx_m * pixels_per_meter * gpx_scale
                dy = dy_m * pixels_per_meter * gpx_scale
                rad = np.radians(gpx_rot)
                rx = dx * np.cos(rad) - dy * np.sin(rad)
                ry = dx * np.sin(rad) + dy * np.cos(rad)
                px = int(w_orig / 2 + rx + gpx_offset_x)
                py = int(h_orig / 2 + ry + gpx_offset_y)
                gpx_pixels.append((px, py))
            
            for i in range(len(gpx_pixels) - 1):
                pt1, pt2 = gpx_pixels[i], gpx_pixels[i+1]
                lat1, lon1, t1 = seg[i]
                lat2, lon2, t2 = seg[i+1]
                pace = None
                if t1 and t2:
                    dist_km = haversine_distance((lat1, lon1), (lat2, lon2))
                    dt_sec = (t2 - t1).total_seconds()
                    if dist_km > 0.002 and dt_sec > 0:
                        pace = (dt_sec / 60.0) / dist_km
                seg_color = get_color_for_pace(pace)
                cv2.line(vis, pt1, pt2, (255, 255, 255), thickness=6)
                cv2.line(vis, pt1, pt2, seg_color, thickness=3)

    cv2.circle(vis, orig_start, 30, (255, 0, 255), 5)
    cv2.circle(vis, orig_goal, 30, (255, 0, 255), 5)
    cv2.circle(vis, orig_goal, 18, (255, 0, 255), 3)

    for i in reversed(range(len(routes))):
        color = colors[i]
        for j in range(len(routes[i]) - 1):
            pt1 = (int(routes[i][j][1] * scale_inv), int(routes[i][j][0] * scale_inv))
            pt2 = (int(routes[i][j+1][1] * scale_inv), int(routes[i][j+1][0] * scale_inv))
            cv2.line(vis, pt1, pt2, color, thickness=4)

    with col_map:
        # ★ 完全解決策：高画質のまま1600pxに制限し、use_column_width=Trueを復活
        # これにより、どんなブラウザサイズでも「画像に対する正確な相対座標」が確実に取れます
        disp_w = min(w_orig, 1600)
        disp_h = int(h_orig * (disp_w / w_orig))
        vis_disp = cv2.resize(vis, (disp_w, disp_h))
        vis_rgb = cv2.cvtColor(vis_disp, cv2.COLOR_BGR2RGB)
        
        click_val = streamlit_image_coordinates(vis_rgb, key="main_map", use_column_width=True)

        if click_val is not None and click_val != st.session_state.last_click:
            st.session_state.last_click = click_val
            
            # コンポーネントから返ってきた座標を、渡した画像の幅(disp_w)で割ることで、完璧な%を取得
            nx = click_val['x'] / disp_w
            ny = click_val['y'] / disp_h
            
            if st.session_state.point_type == "🔵 スタート":
                st.session_state.start_nx, st.session_state.start_ny = nx, ny
            else:
                st.session_state.goal_nx, st.session_state.goal_ny = nx, ny
            
            st.rerun() 
else:
    st.info("左のパネルから地図画像をアップロードしてください。")