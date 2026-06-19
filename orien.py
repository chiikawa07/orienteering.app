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
    # アップロードされたファイルをOpenCV形式に変換
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # =========================
    # ① 画像読み込み & 前処理（計算速度UPのため縮小）
    # =========================
    scale = 0.2
    small_img = cv2.resize(img, (0,0), fx=scale, fy=scale)
    hsv_small = cv2.cvtColor(small_img, cv2.COLOR_BGR2HSV)
    h_s, w_s = small_img.shape[:2]

    # =========================
    # ② 色マスク作成（地図の色を認識）
    # =========================
    # 白（走りやすい森）
    mask_white = cv2.inRange(hsv_small, (0, 0, 180), (180, 25, 255))
    # 黄色（オープン・走りやすい）
    mask_yellow = cv2.inRange(hsv_small, (15, 30, 150), (35, 255, 255))
    # 緑（遅い藪）
    mask_green = cv2.inRange(hsv_small, (35, 30, 50), (85, 255, 255))
    # 黒・茶色（等高線や道など）
    mask_black = cv2.inRange(hsv_small, (0, 0, 0), (180, 255, 120))

    # =========================
    # ③ 破線対応（膨張処理）
    # =========================
    kernel = np.ones((3,3), np.uint8)
    mask_black_dilated = cv2.dilate(mask_black, kernel, iterations=1)

    # =========================
    # ④ 黒を「道」か「崖（壁）」に分類
    # =========================
    road_mask = np.zeros_like(mask_black)
    wall_mask = np.zeros_like(mask_black)

    contours, _ = cv2.findContours(mask_black_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = max(cw, ch) / (min(cw, ch) + 1)
        # 細長くて小さいものは「道」とする
        if area < 100 and ratio > 3:
            cv2.drawContours(road_mask, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(wall_mask, [cnt], -1, 255, -1)

    # =========================
    # ⑤ コストマップ（AIの脳内）生成
    # =========================
    small_cost = np.full((h_s, w_s), 5.0)  # 未知の色は遅い(5.0)とする
    
    small_cost[mask_white > 0] = 1.0       # 白：標準ペース
    small_cost[mask_yellow > 0] = 0.8      # 黄：オープン（最速）
    small_cost[mask_green > 0] = 3.0       # 緑：遅い
    small_cost[road_mask > 0] = 0.5        # 道：爆速
    small_cost[wall_mask > 0] = 9999       # 崖：通行不可

    # ズル防止策：画像の端っこに「見えない壁」を作る
    margin = 5
    small_cost[0:margin, :] = 9999
    small_cost[-margin:, :] = 9999
    small_cost[:, 0:margin] = 9999
    small_cost[:, -margin:] = 9999

    # =========================
   # =========================
    # ⑥ AIの脳内マップ可視化（サイドバー）
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.subheader("AIの脳内マップ")
    
    # 【修正】9999（壁）のせいで他の色が真っ黒に潰れるのを防ぐため、上限を10でカットする
    display_cost = np.clip(small_cost, 0, 10)
    cost_visual = cv2.normalize(display_cost, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    st.sidebar.image(cost_visual, caption="黒＝速い / 白＝遅い・壁", use_container_width=True)

    # 【追加】AIの色認識が上手くいっているか確認するデバッグパネル
    with st.sidebar.expander("🔍 AIの色認識テスト（デバッグ用）"):
        st.write("白く光っている部分が、AIがその色だと認識した場所です。")
        st.image(mask_green, caption="緑（藪）の認識", use_container_width=True)
        st.image(mask_yellow, caption="黄色（オープン）の認識", use_container_width=True)
        st.image(mask_white, caption="白（森）の認識", use_container_width=True)

    # =========================
    # ⑦ 経路探索アルゴリズム（ダイクストラ法）
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
                    if cost_map[ny,nx] >= 9999:  # 壁はスキップ
                        continue
                    # 斜め移動のコスト補正（ルート2倍）
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
    # ⑧ スタート・ゴール設定（スライダー）
    # =========================
    st.sidebar.header("コントロールの設定")
    sy = st.sidebar.slider("スタート Y位置 (%)", 0, 100, 50)
    sx = st.sidebar.slider("スタート X位置 (%)", 0, 100, 30)
    gy = st.sidebar.slider("ゴール Y位置 (%)", 0, 100, 50)
    gx = st.sidebar.slider("ゴール X位置 (%)", 0, 100, 70)

    # 探索用の座標計算
    start_y = min(int(h_s * (sy / 100)), h_s - 1)
    start_x = min(int(w_s * (sx / 100)), w_s - 1)
    goal_y  = min(int(h_s * (gy / 100)), h_s - 1)
    goal_x  = min(int(w_s * (gx / 100)), w_s - 1)
    
    start = (start_y, start_x)
    goal  = (goal_y, goal_x)

    st.sidebar.markdown("---")
    st.sidebar.write(f"🟢 スタート地点のコスト: `{small_cost[start]}`")
    st.sidebar.write(f"🔴 ゴール地点のコスト: `{small_cost[goal]}`")

    # 安全装置
    if small_cost[start] >= 9999 or small_cost[goal] >= 9999:
        st.error("⚠️ スタートまたはゴールが通行不可エリア（黒線や枠外）に配置されています。スライダーを少しずらしてください。")
    else:
        with st.spinner('AIがベストルートを探索中...'):
            path = dijkstra(small_cost, start, goal)

        if not path or len(path) <= 1:
            st.warning("⚠️ ルートが見つかりませんでした。完全に壁に囲まれている可能性があります。")
        else:
            # =========================
            # ⑨ 可視化 (画像への描画)
            # =========================
            vis = img.copy()
            scale_inv = int(1 / scale)
            h_orig, w_orig = img.shape[:2]
            purple = (255, 0, 255) # オリエンテーリング記号の色

            # 実サイズ座標の算出
            orig_start = (int(w_orig * sx / 100), int(h_orig * sy / 100))
            orig_goal = (int(w_orig * gx / 100), int(h_orig * gy / 100))

            # スタート地点を円で描画
            cv2.circle(vis, orig_start, 30, purple, 5)
            # ゴール地点を二重円で描画
            cv2.circle(vis, orig_goal, 30, purple, 5)
            cv2.circle(vis, orig_goal, 18, purple, 3)

            # AIのルートを太い赤線で描画
            for i in range(len(path) - 1):
                pt1 = (path[i][1] * scale_inv, path[i][0] * scale_inv)
                pt2 = (path[i+1][1] * scale_inv, path[i+1][0] * scale_inv)
                cv2.line(vis, pt1, pt2, (0, 0, 255), thickness=4)

            # ブラウザに表示
            st.subheader("AI算出したベストルート")
            st.image(vis, channels="BGR", caption="解析結果", use_container_width=True)

else:
    st.info("上のボックスから地図画像をアップロードすると自動的に解析が始まります。")