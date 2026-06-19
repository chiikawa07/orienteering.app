import cv2
import numpy as np
import heapq
import streamlit as st

# ==========================================
# ① 画面からファイルを受け取る仕様
# ==========================================
st.title("オリエンテーリング ルート解析AI")
uploaded_file = st.file_uploader("地図画像（PNG等）を選択してください", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # アップロードされたファイルをOpenCV形式に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # 既存の前処理
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, w = img.shape[:2]

    # =========================
    # ② 色マスク作成
    # =========================
    # 白（走りやすい）
    mask_white = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    # 緑（遅い）
    mask_green = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))
    # 黒（道・崖）
    mask_black = cv2.inRange(hsv, (0, 0, 0), (180, 255, 50))

    # =========================
    # ③ 破線対応（膨張）
    # =========================
    kernel = np.ones((3,3), np.uint8)
    mask_black_dilated = cv2.dilate(mask_black, kernel, iterations=1)

    # =========================
    # ④ 黒を「道 or 崖」に分類
    # =========================
    road_mask = np.zeros_like(mask_black)
    wall_mask = np.zeros_like(mask_black)

    contours, _ = cv2.findContours(mask_black_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = max(cw, ch) / (min(cw, ch) + 1)

        if area < 100 and ratio > 3:
            cv2.drawContours(road_mask, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(wall_mask, [cnt], -1, 255, -1)

    # =========================
    # ⑤ コストマップ生成
    # =========================
    cost = np.full((h, w), 5.0)  
    cost[mask_white > 0] = 1.0
    cost[mask_green > 0] = 3.0
    cost[road_mask > 0] = 0.5
    cost[wall_mask > 0] = 9999

    # =========================
    # ⑥ 軽量化（縮小）
    # =========================
    scale = 0.2
    small_cost = cv2.resize(cost, (0,0), fx=scale, fy=scale)

    # =========================
    # ⑦ ダイクストラ法
    # =========================
    def dijkstra(cost_map, start, goal):
        h_s, w_s = cost_map.shape
        dist = np.full((h_s, w_s), np.inf)
        prev = np.full((h_s, w_s, 2), -1)

        dist[start] = 0
        pq = [(0, start)]

        directions = [(-1,0),(1,0),(0,-1),(0,1),
                      (-1,-1),(-1,1),(1,-1),(1,1)]

        while pq:
            d, (y,x) = heapq.heappop(pq)
            if (y,x) == goal:
                break

            for dy, dx in directions:
                ny, nx = y+dy, x+dx
                if 0 <= ny < h_s and 0 <= nx < w_s:
                    if cost_map[ny,nx] >= 9999: # 壁はスキップ
                        continue
                    
                    # 斜め移動のコスト補正
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

   
    # ⑧ スタート・ゴール設定（スライダー化）
    # =========================
    st.sidebar.header("コントロールの設定")
    sy = st.sidebar.slider("スタート Y位置 (%)", 0, 100, 50) # 初期値を中央付近に変更
    sx = st.sidebar.slider("スタート X位置 (%)", 0, 100, 30)
    gy = st.sidebar.slider("ゴール Y位置 (%)", 0, 100, 50)
    gx = st.sidebar.slider("ゴール X位置 (%)", 0, 100, 70)

    # 探索用座標（エラーを防ぐため、最大値を画像サイズ-1に制限）
    start_y = min(int(h * scale * (sy / 100)), int(h * scale) - 1)
    start_x = min(int(w * scale * (sx / 100)), int(w * scale) - 1)
    goal_y  = min(int(h * scale * (gy / 100)), int(h * scale) - 1)
    goal_x  = min(int(w * scale * (gx / 100)), int(w * scale) - 1)
    
    start = (start_y, start_x)
    goal  = (goal_y, goal_x)

    # 現在のコストをサイドバーに表示（デバッグ用）
    st.sidebar.markdown("---")
    st.sidebar.write(f"🟢 スタート地点のコスト: `{small_cost[start]}`")
    st.sidebar.write(f"🔴 ゴール地点のコスト: `{small_cost[goal]}`")

    # 安全装置：壁（9999）の上にいる場合は計算しない
    if small_cost[start] >= 9999 or small_cost[goal] >= 9999:
        st.error("⚠️ スタートまたはゴールが通行不可エリア（黒線や枠外）に配置されています。スライダーを少しずらしてください。")
    else:
        # 処理中であることを画面に示すスピナー
        with st.spinner('AIがベストルートを探索中...'):
            path = dijkstra(small_cost, start, goal)

        if not path or len(path) <= 1:
            st.warning("⚠️ ルートが見つかりませんでした。完全に壁に囲まれている可能性があります。")
        else:
            # =========================
            # ⑨ 可視化 (記号の描画)
            # =========================
            vis = img.copy()
            scale_inv = int(1 / scale)
            purple = (255, 0, 255)

            # 実サイズ座標の算出
            orig_start = (int(w * sx / 100), int(h * sy / 100))
            orig_goal = (int(w * gx / 100), int(h * gy / 100))

            # スタート地点を円で描画
            cv2.circle(vis, orig_start, 30, purple, 5)
            # ゴール地点を二重円で描画
            cv2.circle(vis, orig_goal, 30, purple, 5)
            cv2.circle(vis, orig_goal, 18, purple, 3)

            # ルートを描画
            for i in range(len(path) - 1):
                pt1 = (path[i][1] * scale_inv, path[i][0] * scale_inv)
                pt2 = (path[i+1][1] * scale_inv, path[i+1][0] * scale_inv)
                cv2.line(vis, pt1, pt2, (0, 0, 255), thickness=4)

            st.subheader("AI算出したベストルート")
            st.image(vis, channels="BGR", caption="解析結果", use_container_width=True)
# =========================
    # ⑨ 可視化 (元の画像に太い線を引く)
    # =========================
    vis = img.copy()
    scale_inv = int(1 / scale)

    for i in range(len(path) - 1):
        pt1 = (path[i][1] * scale_inv, path[i][0] * scale_inv)
        pt2 = (path[i+1][1] * scale_inv, path[i+1][0] * scale_inv)
        cv2.line(vis, pt1, pt2, (0, 0, 255), thickness=3)

    st.subheader("AI算出したベストルート")
    st.image(vis, channels="BGR", caption="解析結果", use_container_width=True)

else:
    st.info("上のボックスから地図画像をアップロードすると自動的に解析が始まります。")