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
        return []

def get_color_for_pace(pace):
    if pace is None: return (0, 180, 255) 
    ratio = max(0.0, min(1.0, (pace - 4.0) / (15.0 - 4.0))) 
    return (0, int(255 * (1 - ratio)), int(255 * ratio)) 

# ==========================================
# 画像処理をキャッシュ化
# ==========================================
@st.cache_data(show_spinner="AIが地図と地形を解析中...")
def process_map_data(file_bytes, scale, slope_weight, nav_weight):
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    h_s, w_s = small_img.shape[:2]

    hsv = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    lower_mag = np.array([125, 40, 40])
    upper_mag = np.array([175, 255, 255])
    mask_magenta = cv2.inRange(hsv, lower_mag, upper_mag)
    
    kernel_mag = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask_magenta_wall = cv2.dilate(mask_magenta, kernel_mag, iterations=2)

    Z = np.float32(small_img.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, 6, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
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
    center_lab = cv2.cvtColor(np.array([np.uint8(center)]), cv2.COLOR_BGR2LAB)[0]

    masks = {k: np.zeros((h_s, w_s), dtype=np.uint8) for k in isom_colors.keys()}
    for i in range(6):
        min_dist, closest = float('inf'), "white"
        for name, target_lab in isom_lab.items():
            dist = np.linalg.norm(np.float32(center_lab[i]) - np.float32(target_lab))
            if dist < min_dist: min_dist, closest = dist, name
        masks[closest][labels_reshaped == i] = 255

    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_black_closed = cv2.morphologyEx(masks["black"], cv2.MORPH_CLOSE, kernel_close)
    road_mask = np.zeros_like(masks["black"])
    wall_mask = np.zeros_like(masks["black"])

    contours, _ = cv2.findContours(mask_black_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w_rect, h_rect = cv2.boundingRect(cnt)
        ratio = max(w_rect, h_rect) / (min(w_rect, h_rect) + 1)
        if cv2.countNonZero(masks["brown"][y:y+h_rect, x:x+w_rect]) > 0: continue
        if area > 400 and ratio < 3.0: cv2.drawContours(wall_mask, [cnt], -1, 255, -1)
        else: cv2.drawContours(road_mask, [cnt], -1, 255, -1)

    corners = cv2.goodFeaturesToTrack(road_mask, maxCorners=50, qualityLevel=0.1, minDistance=20)
    attack_points = [(int(cy), int(cx)) for cx, cy in (c.ravel() for c in corners)] if corners is not None else []

    # ==========================================
    # ★NEW: モルフォロジー変換による有機的な余白カット
    # ==========================================
    # 白以外の部分（色がある部分）をベースにする
    non_white = cv2.bitwise_not(masks["white"])
    # 収縮させて、タイトル文字などの「細い線や小さな塊」を消滅させる
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    eroded = cv2.erode(non_white, kernel_erode)
    # 膨張させて、残った等高線や森を1つの巨大な塊（地図の島）にくっつける
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (41, 41))
    map_blob = cv2.dilate(eroded, kernel_dilate)
    
    # 島の内部の穴を完全に塗りつぶす
    contours_map, _ = cv2.findContours(map_blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    map_blob_filled = np.zeros_like(map_blob)
    if contours_map:
        largest_cnt = max(contours_map, key=cv2.contourArea)
        cv2.drawContours(map_blob_filled, [largest_cnt], -1, 255, -1)
    else:
        map_blob_filled.fill(255)

    brown_blur = cv2.GaussianBlur(masks["brown"], (5, 5), 0)
    grad_x = cv2.Sobel(brown_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(brown_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_x = np.where(grad_mag > 0, grad_x / grad_mag, 0.0)
    grad_y = np.where(grad_mag > 0, grad_y / grad_mag, 0.0)
    
    slope_penalty = (cv2.GaussianBlur(masks["brown"], (31, 31), 0) / 255.0) * 20.0
    nav_penalty = (np.clip(cv2.distanceTransform(cv2.bitwise_not(road_mask), cv2.DIST_L2, 5), 0, 80) / 80.0) * 3.0

    small_cost = np.full((h_s, w_s), 5.0)
    small_cost[masks["white"] > 0] = 1.0
    small_cost[masks["yellow"] > 0] = 0.8
    small_cost[masks["brown"] > 0] = 1.2
    small_cost[masks["green"] > 0] = 3.0
    small_cost[road_mask > 0] = 0.5
    small_cost = small_cost + slope_penalty + nav_penalty
    small_cost[wall_mask > 0] = 9999
    small_cost[masks["blue"] > 0] = 9999
    small_cost[mask_magenta_wall > 0] = 9999

    return h_s, w_s, attack_points, grad_x, grad_y, grad_mag, small_cost, map_blob_filled

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
                    base_step_cost += abs(mdy * gy_mat[ny, nx] + mdx * gx_mat[ny, nx]) * c_weight
                nd = d + (base_step_cost * move_weight)
                if nd < dist[ny,nx]:
                    dist[ny,nx] = nd
                    prev[ny,nx] = [y,x]
                    heapq.heappush(pq, (nd, (ny,nx)))

    path, cur = [], goal
    while tuple(cur) != tuple(start):
        path.append(cur)
        cur = prev[cur[0], cur[1]]
        if cur[0] == -1: break
    path.append(start)
    return path[::-1]

def snap_to_valid(pt, cost_map, max_r=50):
    if cost_map[pt] < 9999: return pt
    y, x = pt
    h, w = cost_map.shape
    for r in range(1, max_r):
        for dy in range(-r, r+1):
            for dx in range(-r, r+1):
                if abs(dy) == r or abs(dx) == r:
                    ny, nx = y+dy, x+dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if cost_map[ny, nx] < 9999:
                            return (ny, nx)
    return pt

# ==========================================
# セッション初期化
# ==========================================
if 'start_nx' not in st.session_state:
    st.session_state.start_nx, st.session_state.start_ny = 0.53, 0.77
    st.session_state.goal_nx, st.session_state.goal_ny = 0.70, 0.50
if 'last_click' not in st.session_state:
    st.session_state.last_click = None

# ==========================================
# メインUI
# ==========================================
col_panel, col_map = st.columns([1, 3])

with col_panel:
    st.markdown("### 🧭 Livelox風 解析AI")
    with st.expander("📂 地図とGPSの読み込み", expanded=True):
        uploaded_file = st.file_uploader("地図画像 (必須)", type=["png", "jpg", "jpeg"])
        gpx_file = st.file_uploader("GPSログ (.gpx)", type=["gpx"])

if uploaded_file is not None:
    gpx_segments, total_gpx_dist, total_pts = [], 0.0, 0
    if gpx_file is not None:
        gpx_segments = parse_gpx_data(gpx_file.read())
        for seg in gpx_segments:
            total_pts += len(seg)
            for i in range(len(seg) - 1):
                total_gpx_dist += haversine_distance((seg[i][0], seg[i][1]), (seg[i+1][0], seg[i+1][1]))

    scale = 0.35
    file_bytes = bytes(uploaded_file.read())
    (h_s, w_s, attack_points, grad_x, grad_y, grad_mag, small_cost, map_blob_filled) = process_map_data(file_bytes, scale, 20.0, 3.0)

    with col_panel:
        point_type = st.radio("📌 地図をクリックして移動:", ["🔵 スタート", "🔴 ゴール"])
        
        with st.expander("✂️ 競技エリアの制限 (余白カット)", expanded=False):
            st.write("AIが余白を「森」と勘違いするのを防ぎます。")
            use_auto_crop = st.checkbox("🤖 自動トリミングを有効にする", value=True)
            
            if use_auto_crop:
                st.info("自動トリミング適用中（地図上の暗い領域はAIが進入しません）")
            else:
                crop_top = st.slider("上部のカット (%)", 0, 50, 0)
                crop_bottom = st.slider("下部のカット (%)", 0, 50, 0)
                crop_left = st.slider("左側のカット (%)", 0, 50, 0)
                crop_right = st.slider("右側のカット (%)", 0, 50, 0)

        with st.expander("🎯 クリックが効かない場合の微調整", expanded=False):
            st.session_state.start_nx = st.slider("スタートの横位置 (X)", 0.0, 1.0, value=float(st.session_state.start_nx), step=0.01)
            st.session_state.start_ny = st.slider("スタートの縦位置 (Y)", 0.0, 1.0, value=float(st.session_state.start_ny), step=0.01)
            st.session_state.goal_nx = st.slider("ゴールの横位置 (X)", 0.0, 1.0, value=float(st.session_state.goal_nx), step=0.01)
            st.session_state.goal_ny = st.slider("ゴールの縦位置 (Y)", 0.0, 1.0, value=float(st.session_state.goal_ny), step=0.01)

    search_cost = small_cost.copy()
    if use_auto_crop:
        search_cost[map_blob_filled == 0] = 9999
    else:
        t_m = int(h_s * (crop_top / 100))
        b_m = int(h_s * (1 - crop_bottom / 100))
        l_m = int(w_s * (crop_left / 100))
        r_m = int(w_s * (1 - crop_right / 100))
        search_cost[0:t_m, :] = 9999
        search_cost[b_m:h_s, :] = 9999
        search_cost[:, 0:l_m] = 9999
        search_cost[:, r_m:w_s] = 9999

    sx = max(0, min(int(st.session_state.start_nx * w_s), w_s - 1))
    sy = max(0, min(int(st.session_state.start_ny * h_s), h_s - 1))
    gx = max(0, min(int(st.session_state.goal_nx * w_s), w_s - 1))
    gy = max(0, min(int(st.session_state.goal_ny * h_s), h_s - 1))
    
    start = snap_to_valid((sy, sx), search_cost)
    goal = snap_to_valid((gy, gx), search_cost)

    routes, metrics = [], []
    if search_cost[start] >= 9999 or search_cost[goal] >= 9999:
        st.error("⚠️ スタートまたはゴールが『完全な場外』にあります。地図の内側をクリックしてください。")
    else:
        path1 = dijkstra(search_cost, grad_x, grad_y, grad_mag, start, goal)
        if path1 and len(path1) > 1:
            routes.append((path1, (0, 0, 255)))
            metrics.append({"名前": "AI 最適解", "色": "🔴 赤", "スコア": round(sum(small_cost[p[0], p[1]] for p in path1), 1)})

        best_ap = min(attack_points, key=lambda p: np.hypot(p[0]-goal[0], p[1]-goal[1])) if attack_points else None
        if best_ap and search_cost[best_ap] < 9999:
            path_to_ap = dijkstra(search_cost, grad_x, grad_y, grad_mag, start, best_ap)
            path_from_ap = dijkstra(search_cost, grad_x, grad_y, grad_mag, best_ap, goal)
            if path_to_ap and path_from_ap:
                path2 = path_to_ap[:-1] + path_from_ap
                routes.append((path2, (255, 0, 0)))
                metrics.append({"名前": "AP 経由", "色": "🔵 青", "スコア": round(sum(small_cost[p[0], p[1]] for p in path2), 1)})

    vis = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
    h_orig, w_orig = vis.shape[:2]
    scale_inv = 1 / scale

    # ★NEW: 進入禁止エリア（トリミング部分）をグレーアウト＆黒い境界線で描画
    if use_auto_crop:
        map_blob_orig = cv2.resize(map_blob_filled, (w_orig, h_orig), interpolation=cv2.INTER_NEAREST)
        vis[map_blob_orig == 0] = (vis[map_blob_orig == 0] * 0.4).astype(np.uint8)
        contours_orig, _ = cv2.findContours(map_blob_orig, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours_orig, -1, (0, 0, 0), 4)
    else:
        t_orig, b_orig = int((t_m / h_s) * h_orig), int((b_m / h_s) * h_orig)
        l_orig, r_orig = int((l_m / w_s) * w_orig), int((r_m / w_s) * w_orig)
        if t_orig > 0: vis[0:t_orig, :] = (vis[0:t_orig, :] * 0.4).astype(np.uint8)
        if b_orig < h_orig: vis[b_orig:h_orig, :] = (vis[b_orig:h_orig, :] * 0.4).astype(np.uint8)
        if l_orig > 0: vis[:, 0:l_orig] = (vis[:, 0:l_orig] * 0.4).astype(np.uint8)
        if r_orig < w_orig: vis[:, r_orig:w_orig] = (vis[:, r_orig:w_orig] * 0.4).astype(np.uint8)
        cv2.rectangle(vis, (l_orig, t_orig), (r_orig, b_orig), (0, 0, 0), 4)

    orig_start = (int(start[1] * scale_inv), int(start[0] * scale_inv))
    orig_goal = (int(goal[1] * scale_inv), int(goal[0] * scale_inv))

    if 'best_ap' in locals() and best_ap: 
        cv2.circle(vis, (int(best_ap[1] * scale_inv), int(best_ap[0] * scale_inv)), 15, (255, 255, 0), 4)

    with col_panel:
        st.markdown("### 🏃‍♂️ ルート比較")
        for m in metrics:
            st.markdown(f"**{m['色']} : {m['名前']}**<br>スコア: {m['スコア']}", unsafe_allow_html=True)
            st.markdown("<hr>", unsafe_allow_html=True)
        
        if gpx_segments:
            st.markdown(f"**🏃‍♂️ GPS実走 (速度色分け)**<br>距離: {round(total_gpx_dist, 2)} km", unsafe_allow_html=True)
            with st.expander("⚙️ GPS位置合わせ", expanded=True):
                gpx_scale = st.slider("拡大率", 0.1, 5.0, 1.0, step=0.05)
                gpx_rot = st.slider("回転角度", -180, 180, 0, step=1)
                gpx_offset_x = st.slider("左右移動 (X)", -2000, 2000, 0, step=10)
                gpx_offset_y = st.slider("上下移動 (Y)", -2000, 2000, 0, step=10)
        else:
            gpx_scale, gpx_rot, gpx_offset_x, gpx_offset_y = 1.0, 0, 0, 0

    if gpx_segments:
        all_lats, all_lons = [p[0] for s in gpx_segments for p in s], [p[1] for s in gpx_segments for p in s]
        center_lat, center_lon = (min(all_lats) + max(all_lats)) / 2, (min(all_lons) + max(all_lons)) / 2
        m_per_deg_lat, m_per_deg_lon = 111320.0, 40075000.0 * np.cos(np.radians(center_lat)) / 360.0
        pixels_per_meter = (min(w_orig, h_orig) * 0.8) / max((max(all_lats)-min(all_lats))*m_per_deg_lat, (max(all_lons)-min(all_lons))*m_per_deg_lon)

        for seg in gpx_segments:
            gpx_pixels = []
            for lat, lon, _ in seg:
                dx, dy = (lon - center_lon) * m_per_deg_lon * pixels_per_meter * gpx_scale, -(lat - center_lat) * m_per_deg_lat * pixels_per_meter * gpx_scale
                rad = np.radians(gpx_rot)
                gpx_pixels.append((int(w_orig / 2 + (dx * np.cos(rad) - dy * np.sin(rad)) + gpx_offset_x), int(h_orig / 2 + (dx * np.sin(rad) + dy * np.cos(rad)) + gpx_offset_y)))
            
            for i in range(len(gpx_pixels) - 1):
                t1, t2 = seg[i][2], seg[i+1][2]
                pace = None
                if t1 and t2:
                    dist_km, dt_sec = haversine_distance((seg[i][0], seg[i][1]), (seg[i+1][0], seg[i+1][1])), (t2 - t1).total_seconds()
                    if dist_km > 0.002 and dt_sec > 0: pace = (dt_sec / 60.0) / dist_km
                if (0 <= gpx_pixels[i][0] < w_orig and 0 <= gpx_pixels[i][1] < h_orig):
                    cv2.line(vis, gpx_pixels[i], gpx_pixels[i+1], (255, 255, 255), thickness=6)
                    cv2.line(vis, gpx_pixels[i], gpx_pixels[i+1], get_color_for_pace(pace), thickness=3)

    cv2.circle(vis, orig_start, 30, (255, 0, 255), 5)
    cv2.circle(vis, orig_goal, 30, (255, 0, 255), 5)
    cv2.circle(vis, orig_goal, 18, (255, 0, 255), 3)

    for path, color in reversed(routes):
        for j in range(len(path) - 1):
            cv2.line(vis, (int(path[j][1]*scale_inv), int(path[j][0]*scale_inv)), (int(path[j+1][1]*scale_inv), int(path[j+1][0]*scale_inv)), color, thickness=4)

    with col_map:
        vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        click_val = streamlit_image_coordinates(vis_rgb, width=1000, key="main_map")

        if click_val is not None and click_val != st.session_state.last_click:
            st.session_state.last_click = click_val
            nx, ny = click_val['x'] / w_orig, click_val['y'] / h_orig
            if point_type == "🔵 スタート":
                st.session_state.start_nx, st.session_state.start_ny = nx, ny
            else:
                st.session_state.goal_nx, st.session_state.goal_ny = nx, ny
            st.rerun() 
else:
    st.info("左のパネルから地図画像をアップロードしてください。")