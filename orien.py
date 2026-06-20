import cv2
import numpy as np
import heapq
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
import xml.etree.ElementTree as ET

# ==========================================
# UI: タイトルとページ設定
# ==========================================
st.set_page_config(layout="wide")
st.title("オリエンテーリングAI (GPSログ重ね合わせ版)")

# 2つのファイルをアップロードできるように配置
col_file1, col_file2 = st.columns(2)
with col_file1:
    uploaded_file = st.file_uploader("1. 地図画像（PNG等）を選択してください", type=["png", "jpg", "jpeg"])
with col_file2:
    gpx_file = st.file_uploader("2. 【任意】実走GPSログ（.gpx）を選択してください", type=["gpx"])

# 2点間の球面距離(km)を計算するハバーシン公式
def haversine_distance(p1, p2):
    R = 6371.0  # 地球の半径 (km)
    lat1, lon1 = np.radians(p1[0]), np.radians(p1[1])
    lat2, lon2 = np.radians(p2[0]), np.radians(p2[1])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# GPXファイルをパースして緯度経度のリストを返す関数
def parse_gpx_data(file_bytes):
    try:
        root = ET.fromstring(file_bytes)
        points = []
        for el in root.iter():
            if el.tag.endswith('trkpt'):
                lat = float(el.attrib['lat'])
                lon = float(el.attrib['lon'])
                points.append((lat, lon))
        return points
    except Exception as e:
        st.error(f"GPXファイルの解析に失敗しました: {e}")
        return []

# ==========================================
# 画像処理をキャッシュ化
# ==========================================
@st.cache_data(show_spinner="AIが地図の色と地形を解析中...")
def process_map_data(file_bytes, scale, slope_weight, nav_weight):
    img_array = np.asarray(bytearray(file_bytes), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    h_s, w_s = small_img.shape[:2]

    # K-Means減色
    Z = small_img.reshape((-1, 3))
    Z = np.float32(Z)
    K = 6
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    center = np.uint8(center)
    labels_reshaped = label.reshape((h_s, w_s))

    # ISOM色マッチング
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

    # 道・建物の判別
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

    # アタックポイント抽出
    corners = cv2.goodFeaturesToTrack(road_mask, maxCorners=50, qualityLevel=0.1, minDistance=20)
    attack_points = []
    if corners is not None:
        for i in corners:
            cx, cy = i.ravel()
            attack_points.append((int(cy), int(cx)))

    # ベクトル・ペナルティ計算
    brown_blur = cv2.GaussianBlur(mask_brown, (5, 5), 0)
    grad_x = cv2.Sobel(brown_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(brown_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_x = np.where(grad_mag > 0, grad_x / grad_mag, 0.0)
    grad_y = np.where(grad_mag > 0, grad_y / grad_mag, 0.0)
    
    slope_heatmap = cv2.GaussianBlur(mask_brown, (31, 31), 0) 
    slope_penalty = (slope_heatmap / 255.0) * slope_weight

    inv_road = cv2.bitwise_not(road_mask)
    dist_to_road = cv2.distanceTransform(inv_road, cv2.DIST_L2, 5)
    dist_capped = np.clip(dist_to_road, 0, 80) 
    nav_penalty = (dist_capped / 80.0) * nav_weight

    # コストマップ合成
    small_cost = np.full((h_s, w_s), 5.0)
    small_cost[mask_white > 0] = 1.0
    small_cost[mask_yellow > 0] = 0.8
    small_cost[mask_brown > 0] = 1.2
    small_cost[mask_green > 0] = 3.0
    small_cost[road_mask > 0] = 0.5
    
    small_cost = small_cost + slope_penalty + nav_penalty
    small_cost[wall_mask > 0] = 9999
    small_cost[mask_blue > 0] = 9999

    margin = 20 
    small_cost[0:margin, :] = 9999
    small_cost[-margin:, :] = 9999
    small_cost[:, 0:margin] = 9999
    small_cost[:, -margin:] = 9999

    return small_img, h_s, w_s, road_mask, attack_points, grad_x, grad_y, grad_mag, dist_capped, small_cost

# ==========================================
# メイン処理開始
# ==========================================
if uploaded_file is not None:
    # 1. 走行ログの読み込み処理（存在する場合）
    gpx_points = []
    total_gpx_dist = 0.0
    if gpx_file is not None:
        gpx_points = parse_gpx_data(gpx_file.read())
        if gpx_points:
            # 実走行距離(km)の算出
            for i in range(len(gpx_points) - 1):
                total_gpx_dist += haversine_distance(gpx_points[i], gpx_points[i+1])

    # サイドバーのUI
    st.sidebar.markdown("---")
    st.sidebar.subheader("⛰️ アップダウン・沢またぎの回避設定")
    slope_weight = st.sidebar.slider("斜度の基本ペナルティ (全体の回避度)", 0.0, 50.0, 20.0, step=2.0)
    cross_weight = st.sidebar.slider("等高線を横切る移動へのペナルティ", 0.0, 100.0, 50.0, step=5.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 ナビゲーション難易度の設定")
    nav_weight = st.sidebar.slider("道から離れることへの不安度", 0.0, 10.0, 3.0, step=0.5)

    # 2. GPSキャリブレーションUIの追加（ログがある場合のみ出現）
    gpx_scale, gpx_rot, gpx_offset_x, gpx_offset_y = 1.0, 0, 0, 0
    if gpx_points:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🏃‍♂️ GPSログの位置同期（位置合わせ）")
        st.sidebar.write("地図上の道と軌跡が重なるように調整してください。")
        gpx_scale = st.sidebar.slider("GPS軌跡の拡大率", 0.1, 10.0, 1.5, step=0.05)
        gpx_rot = st.sidebar.slider("GPS軌跡の回転角度", -180, 180, 0, step=1)
        gpx_offset_x = st.sidebar.slider("左右移動 (X)", -2000, 2000, 0, step=5)
        gpx_offset_y = st.sidebar.slider("上下移動 (Y)", -2000, 2000, 0, step=5)

    # 画質設定
    scale = 0.35

    # キャッシュされた画像解析の呼び出し
    file_bytes = bytes(uploaded_file.read())
    (small_img, h_s, w_s, road_mask, attack_points, grad_x, grad_y, grad_mag, dist_capped, small_cost) = process_map_data(file_bytes, scale, slope_weight, nav_weight)

    # デバッグUI
    st.sidebar.markdown("---")
    st.sidebar.subheader("AIの脳内マップ")
    display_cost = np.clip(small_cost, 0, 10)
    cost_visual = cv2.normalize(display_cost, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    st.sidebar.image(cost_visual, use_container_width=True)

    # 経路探索アルゴリズム
    def dijkstra(cost_map, gx_mat, gy_mat, g_mag, c_weight, start, goal):
        h, w = cost_map.shape
        dist = np.full((h, w), np.inf)
        prev = np.full((h, w, 2), -1)
        dist[start] = 0
        pq = [(0, start)]
        directions = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]

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

    # クリックUI
    if 'start_nx' not in st.session_state:
        st.session_state.start_nx, st.session_state.start_ny = 0.53, 0.77
        st.session_state.goal_nx, st.session_state.goal_ny = 0.70, 0.50
    if 'last_click' not in st.session_state:
        st.session_state.last_click = None

    st.markdown("---")
    st.subheader("📍 1. ルートを設定する (地図をクリック)")
    point_type = st.radio("クリックで動かすポイントを選択:", ["🔵 スタート", "🔴 ゴール"], horizontal=True)
    
    ui_width = 800
    ui_height = int(h_s * (ui_width / w_s))

    click_map_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    click_map_ui = cv2.resize(click_map_img, (ui_width, ui_height))
    
    ui_sx, ui_sy = int(st.session_state.start_nx * ui_width), int(st.session_state.start_ny * ui_height)
    ui_gx, ui_gy = int(st.session_state.goal_nx * ui_width), int(st.session_state.goal_ny * ui_height)

    cv2.circle(click_map_ui, (ui_sx, ui_sy), 8, (255, 0, 255), -1)
    cv2.putText(click_map_ui, "S", (ui_sx + 10, ui_sy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    cv2.circle(click_map_ui, (ui_gx, ui_gy), 8, (255, 0, 255), 2)
    cv2.circle(click_map_ui, (ui_gx, ui_gy), 3, (255, 0, 255), -1)
    cv2.putText(click_map_ui, "G", (ui_gx + 10, ui_gy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
    
    click_val = streamlit_image_coordinates(click_map_ui, key="map_click")

    if click_val is not None and click_val != st.session_state.last_click:
        st.session_state.last_click = click_val
        nx, ny = click_val['x'] / ui_width, click_val['y'] / ui_height
        if point_type == "🔵 スタート":
            st.session_state.start_nx, st.session_state.start_ny = nx, ny
        else:
            st.session_state.goal_nx, st.session_state.goal_ny = nx, ny
        st.rerun()

    margin = 20
    sx = max(margin, min(int(st.session_state.start_nx * w_s), w_s - margin - 1))
    sy = max(margin, min(int(st.session_state.start_ny * h_s), h_s - margin - 1))
    gx = max(margin, min(int(st.session_state.goal_nx * w_s), w_s - margin - 1))
    gy = max(margin, min(int(st.session_state.goal_ny * h_s), h_s - margin - 1))

    start, goal = (sy, sx), (gy, gx)

    # 経路探索実行
    st.markdown("---")
    st.subheader("🗺️ 2. AIルート解析結果")

    if small_cost[start] >= 9999 or small_cost[goal] >= 9999:
        st.error("⚠️ 通行不可エリアです。別の場所をクリックしてください。")
    else:
        with st.spinner('AIがルートを探索中...'):
            routes, metrics = [], []
            colors = [(0, 0, 255), (255, 0, 0), (0, 128, 0)]
            
            path1 = dijkstra(small_cost, grad_x, grad_y, grad_mag, cross_weight, start, goal)
            if path1 and len(path1) > 1:
                routes.append(path1)
                metrics.append({"名前": "第1ルート (AI最適解)", "色": "🔴 赤", "難易度スコア": round(sum(small_cost[p[0], p[1]] for p in path1), 1), "距離": len(path1)})

            best_ap = min(attack_points, key=lambda p: np.hypot(p[0]-goal[0], p[1]-goal[1])) if attack_points else None
            if best_ap:
                path_to_ap = dijkstra(small_cost, grad_x, grad_y, grad_mag, cross_weight, start, best_ap)
                path_from_ap = dijkstra(small_cost, grad_x, grad_y, grad_mag, cross_weight, best_ap, goal)
                if path_to_ap and path_from_ap:
                    path2 = path_to_ap[:-1] + path_from_ap
                    routes.append(path2)
                    metrics.append({"名前": "第2ルート (AP経由)", "色": "🔵 青", "難易度スコア": round(sum(small_cost[p[0], p[1]] for p in path2), 1), "距離": len(path2)})

            if path1:
                search_cost = small_cost.copy()
                for p in path1:
                    y_min, y_max = max(0, p[0]-4), min(h_s, p[0]+5)
                    x_min, x_max = max(0, p[1]-4), min(w_s, p[1]+5)
                    search_cost[y_min:y_max, x_min:x_max] += 25.0
                path3 = dijkstra(search_cost, grad_x, grad_y, grad_mag, cross_weight, start, goal)
                if path3 and len(path3) > 1:
                    routes.append(path3)
                    metrics.append({"名前": "第3ルート (迂回大穴)", "色": "🟢 緑", "難易度スコア": round(sum(small_cost[p[0], p[1]] for p in path3), 1), "距離": len(path3)})

        if not routes:
            st.warning("⚠️ ルートが見つかりませんでした。")
        else:
            # 元画像の復元と描画
            vis = cv2.imdecode(np.asarray(bytearray(file_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
            h_orig, w_orig = vis.shape[:2]

            orig_start = (int(st.session_state.start_nx * w_orig), int(st.session_state.start_ny * h_orig))
            orig_goal = (int(st.session_state.goal_nx * w_orig), int(st.session_state.goal_ny * h_orig))

            scale_inv = 1 / scale
            for ap in attack_points:
                cv2.circle(vis, (int(ap[1] * scale_inv), int(ap[0] * scale_inv)), 4, (0, 255, 255), -1)
            if best_ap:
                cv2.circle(vis, (int(best_ap[1] * scale_inv), int(best_ap[0] * scale_inv)), 15, (255, 255, 0), 4)

            # --------------------------------------------------
            # 🏃‍♂️ GPSデータの2D変換と描画（アフィン変換処理）
            # --------------------------------------------------
            if gpx_points:
                lats = [p[0] for p in gpx_points]
                lons = [p[1] for p in gpx_points]
                mean_lat, mean_lon = np.mean(lats), np.mean(lons)
                
                gpx_pixels = []
                for lat, lon in gpx_points:
                    # 北緯38度付近の経度補正を掛けつつメートル単位の相対距離に変換
                    dx = (lon - mean_lon) * np.cos(np.radians(38.0)) * 111000 * gpx_scale
                    dy = -(lat - mean_lat) * 111000 * gpx_scale
                    
                    # 回転行列の適用
                    rad = np.radians(gpx_rot)
                    rx = dx * np.cos(rad) - dy * np.sin(rad)
                    ry = dx * np.sin(rad) + dy * np.cos(rad)
                    
                    # 元画像上のピクセル座標を決定
                    px = int(w_orig / 2 + rx + gpx_offset_x)
                    py = int(h_orig / 2 + ry + gpx_offset_y)
                    gpx_pixels.append((px, py))
                
                # 実際の軌跡を「鮮やかなオレンジ色」の二重線で描画
                for i in range(len(gpx_pixels) - 1):
                    pt1, pt2 = gpx_pixels[i], gpx_pixels[i+1]
                    if (0 <= pt1[0] < w_orig and 0 <= pt1[1] < h_orig and 0 <= pt2[0] < w_orig and 0 <= pt2[1] < h_orig):
                        cv2.line(vis, pt1, pt2, (0, 100, 255), thickness=6) # 縁取り
                        cv2.line(vis, pt1, pt2, (0, 180, 255), thickness=3) # 中心線

            # コントロール記号の描画
            cv2.circle(vis, orig_start, 30, (255, 0, 255), 5)
            cv2.circle(vis, orig_goal, 30, (255, 0, 255), 5)
            cv2.circle(vis, orig_goal, 18, (255, 0, 255), 3)

            # AIルートの描画
            for i in reversed(range(len(routes))):
                color = colors[i]
                for j in range(len(routes[i]) - 1):
                    pt1 = (int(routes[i][j][1] * scale_inv), int(routes[i][j][0] * scale_inv))
                    pt2 = (int(routes[i][j+1][1] * scale_inv), int(routes[i][j+1][0] * scale_inv))
                    cv2.line(vis, pt1, pt2, color, thickness=4)

            st.image(vis, channels="BGR", caption="赤:最適解 / 青:AP経由 / 緑:大穴 / オレンジ:あなたのGPS実走ログ", use_container_width=True)
            
            # パフォーマンスダッシュボードの出力
            st.subheader("📊 ルートごとのパフォーマンス比較")
            
            # GPSログがある場合はカラムを1つ増やして実走データを並べる
            num_cols = len(metrics) + (1 if gpx_points else 0)
            cols = st.columns(num_cols)
            
            for i, col in enumerate(cols[:len(metrics)]):
                with col:
                    st.markdown(f"**{metrics[i]['color']} : {metrics[i]['名前']}**")
                    st.metric(label="難易度スコア (推定タイム)", value=metrics[i]["難易度スコア"])
                    st.metric(label="移動距離 (ピクセル)", value=metrics[i]["距離"])
            
            if gpx_points:
                with cols[-1]:
                    st.markdown("**🏃‍♂️ オレンジ : あなたのGPS実走**")
                    st.metric(label="実走行距離", value=f"{round(total_gpx_dist, 2)} km")
                    st.metric(label="ログのデータ点数", value=f"{len(gpx_points)} pt")
                    st.caption("※サイドバーの『左右移動』『上下移動』『拡大率』を使って、軌跡を地図の道に重ね合わせてください。")

else:
    st.info("上のボックスから地図画像をアップロードすると自動的に解析が始まります。")