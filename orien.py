import cv2
import numpy as np
import heapq
import streamlit as st

# ==========================================
# UI: タイトルとアップローダー
# ==========================================
st.title("オリエンテーリングAI (ナビゲーション考慮版)")
uploaded_file = st.file_uploader("地図画像（PNG等）を選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # =========================
    # ① 画像読み込み & K-Meansによる完全自動減色
    # =========================
    scale = 0.2
    small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    h_s, w_s = small_img.shape[:2]

    with st.spinner("AIが地図の色を解析・完全自動分類中..."):
        Z = small_img.reshape((-1, 3))
        Z = np.float32(Z)
        K = 6
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        center = np.uint8(center)
        labels_reshaped = label.reshape((h_s, w_s))

        # =========================
        # ② ISOM基準色との自動マッチング (Lab色空間)
        # =========================
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

    # =========================
    # ③〜④ 小径（点線）と建物（壁）の判別
    # =========================
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

    # =========================
    # ⑤ 地形コスト設定 & 勾配ベクトル & 【新規】ナビゲーションペナルティ
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.subheader("⛰️ アップダウン・沢またぎの回避設定")
    slope_weight = st.sidebar.slider("斜度の基本ペナルティ (全体の回避度)", 0.0, 50.0, 20.0, step=2.0)
    cross_weight = st.sidebar.slider("等高線を横切る移動へのペナルティ (沢またぎ防止)", 0.0, 100.0, 50.0, step=5.0)

    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 ナビゲーション難易度の設定")
    st.sidebar.write("数値を上げると、現在地を見失わないよう道や小径（ハンドレール）の近くを好んで走るようになります。")
    nav_weight = st.sidebar.slider("道から離れることへの不安度 (ペナルティ)", 0.0, 10.0, 3.0, step=0.5)

    # (A) 勾配と密集度ペナルティの計算
    brown_blur = cv2.GaussianBlur(mask_brown, (5, 5), 0)
    grad_x = cv2.Sobel(brown_blur, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(brown_blur, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_x = np.where(grad_mag > 0, grad_x / grad_mag, 0.0)
    grad_y = np.where(grad_mag > 0, grad_y / grad_mag, 0.0)
    slope_heatmap = cv2.GaussianBlur(mask_brown, (21, 21), 0)
    slope_penalty = (slope_heatmap / 255.0) * slope_weight

    # (B) 【新規】ナビゲーションペナルティの計算（距離変換）
    # 道マスクを反転（道=0, それ以外=255）させて距離を測る
    inv_road = cv2.bitwise_not(road_mask)
    dist_to_road = cv2.distanceTransform(inv_road, cv2.DIST_L2, 5)
    # 最大50ピクセル離れたところで不安度はピーク（頭打ち）とする
    dist_capped = np.clip(dist_to_road, 0, 50)
    nav_penalty = (dist_capped / 50.0) * nav_weight

    # (C) 基本コストマップの合成
    small_cost = np.full((h_s, w_s), 5.0)
    small_cost[mask_white > 0] = 1.0
    small_cost[mask_yellow > 0] = 0.8
    small_cost[mask_brown > 0] = 1.2
    small_cost[mask_green > 0] = 3.0
    small_cost[road_mask > 0] = 0.5
    
    # ペナルティをすべて上乗せ！
    small_cost = small_cost + slope_penalty + nav_penalty

    # 絶対障害物の適用（上乗せ後に適用して壁を維持する）
    small_cost[wall_mask > 0] = 9999
    small_cost[mask_blue > 0] = 9999

    margin = 15
    small_cost[0:margin, :] = 9999
    small_cost[-margin:, :] = 9999
    small_cost[:, 0:margin] = 9999
    small_cost[:, -margin:] = 9999

    # =========================
    # ⑥ デバッグUI表示
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.subheader("AIの脳内マップ")
    display_cost = np.clip(small_cost, 0, 10)
    cost_visual = cv2.normalize(display_cost, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    st.sidebar.image(cost_visual, caption="黒＝速い / 白＝遅い・壁", use_container_width=True)

    with st.sidebar.expander("🔍 AIの空間認識テスト"):
        # 距離変換マップをカラー化して表示
        dist_color = cv2.normalize(dist_capped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        dist_color = cv2.applyColorMap(dist_color, cv2.COLORMAP_JET)
        st.image(dist_color, caption="🗺️ ナビゲーション不安度 (赤＝道から遠い危険地帯 / 青＝道の近くで安心)", use_container_width=True)
        st.image(slope_heatmap, caption="🔥 急斜面ヒートマップ", use_container_width=True)
        st.image(road_mask, caption="黒（道・小径と認識した場所）", use_container_width=True)

    # =========================
    # ⑦ 経路探索（異方性コスト対応ダイクストラ法）
    # =========================
    def dijkstra(cost_map, gx_mat, gy_mat, g_mag, c_weight, start, goal):
        h, w = cost_map.shape
        dist = np.full((h, w), np.inf)
        prev = np.full((h, w, 2), -1)
        dist[start] = 0
        pq = [(0, start)]
        directions = [(-1,0),(1,0),(0,-1),(0,1), (-1,-1),(-1,1),(1,-1),(1,1)]

        while pq:
            d, (y,x) = heapq.heappop(pq)
            if (y,x) == goal:
                break
            for dy, dx in directions:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h and 0 <= nx < w:
                    if cost_map[ny,nx] >= 9999:
                        continue
                        
                    move_weight = 1.414 if (dy != 0 and dx != 0) else 1.0
                    base_step_cost = cost_map[ny,nx]
                    
                    if g_mag[ny, nx] > 10:
                        move_len = np.sqrt(dy**2 + dx**2)
                        mdy, mdx = dy / move_len, dx / move_len
                        dot_product = abs(mdy * gy_mat[ny, nx] + mdx * gx_mat[ny, nx])
                        direction_penalty = dot_product * c_weight
                        base_step_cost += direction_penalty

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
            if cur[0] == -1:
                break
        path.append(start)
        return path[::-1]

    # =========================
    # ⑧ スライダーと複数ルート探索
    # =========================
    st.sidebar.header("コントロールの設定")
    sy = st.sidebar.slider("スタート Y位置 (%)", 0, 100, 77)
    sx = st.sidebar.slider("スタート X位置 (%)", 0, 100, 53)
    gy = st.sidebar.slider("ゴール Y位置 (%)", 0, 100, 50)
    gx = st.sidebar.slider("ゴール X位置 (%)", 0, 100, 70)

    start_y = min(int(h_s * (sy / 100)), h_s - 1)
    start_x = min(int(w_s * (sx / 100)), w_s - 1)
    goal_y  = min(int(h_s * (gy / 100)), h_s - 1)
    goal_x  = min(int(w_s * (gx / 100)), w_s - 1)
    start, goal = (start_y, start_x), (goal_y, goal_x)

    if small_cost[start] >= 9999 or small_cost[goal] >= 9999:
        st.error("⚠️ スタートまたはゴールが通行不可エリアです。スライダーをずらしてください。")
    else:
        with st.spinner('AIが「ナビのしやすさ」と「地形」を総合評価して探索中...'):
            routes = []
            metrics = []
            colors = [(0, 0, 255), (255, 0, 0), (0, 128, 0)]
            route_names = ["第1ルート (最適解)", "第2ルート (別ルート)", "第3ルート (大穴)"]
            
            search_cost = small_cost.copy()
            for i in range(3):
                path = dijkstra(search_cost, grad_x, grad_y, grad_mag, cross_weight, start, goal)
                if not path or len(path) <= 1:
                    break
                    
                route_dist = len(path)
                route_diff = sum(small_cost[p[0], p[1]] for p in path)
                
                routes.append(path)
                metrics.append({
                    "名前": route_names[i],
                    "色": ["🔴 赤", "🔵 青", "🟢 緑"][i],
                    "難易度スコア": round(route_diff, 1),
                    "相対距離": route_dist
                })
                
                for p in path:
                    y, x = p
                    y_min, y_max = max(0, y-4), min(h_s, y+5)
                    x_min, x_max = max(0, x-4), min(w_s, x+5)
                    search_cost[y_min:y_max, x_min:x_max] += 25.0

        if not routes:
            st.warning("⚠️ ルートが見つかりませんでした。")
        else:
            # =========================
            # ⑨ 可視化とダッシュボード
            # =========================
            vis = img.copy()
            scale_inv = int(1 / scale)
            h_orig, w_orig = img.shape[:2]
            purple = (255, 0, 255)

            orig_start = (int(w_orig * sx / 100), int(h_orig * sy / 100))
            orig_goal = (int(w_orig * gx / 100), int(h_orig * gy / 100))

            cv2.circle(vis, orig_start, 30, purple, 5)
            cv2.circle(vis, orig_goal, 30, purple, 5)
            cv2.circle(vis, orig_goal, 18, purple, 3)

            for i in reversed(range(len(routes))):
                path = routes[i]
                color = colors[i]
                for j in range(len(path) - 1):
                    pt1 = (path[j][1] * scale_inv, path[j][0] * scale_inv)
                    pt2 = (path[j+1][1] * scale_inv, path[j+1][0] * scale_inv)
                    cv2.line(vis, pt1, pt2, color, thickness=4)

            st.subheader("🗺️ AIルート解析結果")
            st.image(vis, channels="BGR", caption="赤:最適解 / 青:第2候補 / 緑:第3候補", use_container_width=True)
            
            st.subheader("📊 ルートごとのパフォーマンス比較")
            cols = st.columns(len(metrics))
            for i, col in enumerate(cols):
                m = metrics[i]
                with col:
                    st.markdown(f"**{m['色']} : {m['名前']}**")
                    st.metric(label="難易度スコア (推定タイム)", value=m["難易度スコア"])
                    st.metric(label="移動距離 (ピクセル)", value=m["相対距離"])

else:
    st.info("上のボックスから地図画像をアップロードすると自動的に解析が始まります。")