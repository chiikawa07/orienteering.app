import cv2
import numpy as np
import heapq
import streamlit as st

# ==========================================
# UI: タイトルとアップローダー
# ==========================================
st.title("オリエンテーリング ルート解析AI")
uploaded_file = st.file_uploader("地図画像（PNG等）を選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # =========================
    # ① 画像読み込み & 前処理（縮小）
    # =========================
    scale = 0.2
    small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    hsv_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    h_s, w_s = small_img.shape[:2]

    # =========================
    # ② 色マスク作成（サイドバーでチューニング）
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 AIの色認識チューニング")
    
    w_v_min = st.sidebar.slider("白: 明るさ(V)の最小値", 0, 255, 210)
    w_s_max = st.sidebar.slider("白: 鮮やかさ(S)の最大値", 0, 255, 12)
    mask_white = cv2.inRange(hsv_small, (0, 0, w_v_min), (180, w_s_max, 255))

    y_h_min = st.sidebar.slider("黄: 色合い(H)の下限", 0, 180, 10)
    y_h_max = st.sidebar.slider("黄: 色合い(H)の上限", 0, 180, 30)
    mask_yellow = cv2.inRange(hsv_small, (y_h_min, 30, 140), (y_h_max, 255, 255))

    g_h_min = st.sidebar.slider("緑: 色合い(H)の下限", 0, 180, 35)
    mask_green = cv2.inRange(hsv_small, (g_h_min, 30, 50), (85, 255, 255))

    # 茶色（等高線）と黒（道・崖）
    mask_brown = cv2.inRange(hsv_small, (10, 30, 50), (30, 150, 200))
    mask_black = cv2.inRange(hsv_small, (0, 0, 0), (180, 255, 90))

    # =========================
    # ③〜④ 破線対応と道判定
    # =========================
    kernel = np.ones((3,3), np.uint8)
    mask_black_dilated = cv2.dilate(mask_black, kernel, iterations=1)
    road_mask = np.zeros_like(mask_black)
    wall_mask = np.zeros_like(mask_black)

    contours, _ = cv2.findContours(mask_black_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = max(cw, ch) / (min(cw, ch) + 1)
        if area < 100 and ratio > 3:
            # 茶色(等高線)と被らない黒を道とする
            mask_roi = mask_brown[y:y+ch, x:x+cw]
            if cv2.countNonZero(mask_roi) == 0:
                cv2.drawContours(road_mask, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(wall_mask, [cnt], -1, 255, -1)

    # =========================
    # ⑤ コストマップ生成（理想のルートへ誘導）
    # =========================
    small_cost = np.full((h_s, w_s), 5.0)
    small_cost[mask_white > 0] = 1.0       # 白
    small_cost[mask_yellow > 0] = 0.8      # 黄（最速）
    small_cost[mask_brown > 0] = 1.5       # 茶（等高線を横切ると遅い）
    small_cost[mask_green > 0] = 3.0       # 緑
    small_cost[road_mask > 0] = 0.5        # 道
    small_cost[wall_mask > 0] = 9999       # 壁

    # ズル防止策（右端や下端の余白を通らせないため、壁を厚く設定）
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

    with st.sidebar.expander("🔍 AIの色認識テスト"):
        st.image(mask_green, caption="緑（藪）", use_container_width=True)
        st.image(mask_yellow, caption="黄色（オープン）", use_container_width=True)
        st.image(mask_white, caption="白（森）", use_container_width=True)
        st.image(mask_brown, caption="茶色（等高線）", use_container_width=True)

    # =========================
    # ⑦ 経路探索アルゴリズム
    # =========================
    def dijkstra(cost_map, start, goal):
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
                    nd = d + (cost_map[ny,nx] * move_weight)
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
        with st.spinner('AIが複数のルートオプションを探索中...'):
            routes = []
            metrics = []
            colors = [(0, 0, 255), (255, 0, 0), (0, 128, 0)] # 赤, 青, 緑
            route_names = ["第1ルート (最適解)", "第2ルート (別ルート)", "第3ルート (大穴)"]
            
            search_cost = small_cost.copy()
            for i in range(3):
                path = dijkstra(search_cost, start, goal)
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
                
                # ペナルティ付与（見つけたルート周辺のコストを上げて別ルートを探させる）
                for p in path:
                    y, x = p
                    y_min, y_max = max(0, y-4), min(h_s, y+5)
                    x_min, x_max = max(0, x-4), min(w_s, x+5)
                    search_cost[y_min:y_max, x_min:x_max] += 15.0

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