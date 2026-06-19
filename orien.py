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
   # =========================
    # =========================
    # ② 色マスク作成（★茶色の追加、白の厳格化、黄色の微調整）
    # =========================
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 AIの色認識チューニング")
    st.sidebar.write("デバッグ画面を見ながらスライダーを動かしてください")

    # 白（走りやすい森。彩度Sの上限を下げて、本当に白い所だけにする）
    w_v_min = st.sidebar.slider("白: 明るさ(V)の最小値", 0, 255, 210)
    w_s_max = st.sidebar.slider("白: 鮮やかさ(S)の最大値 (下げるほど白を厳しく判定)", 0, 255, 12) # 15 -> 12
    mask_white = cv2.inRange(hsv_small, (0, 0, w_v_min), (180, w_s_max, 255))

    # 黄色（オープン・走りやすい。彩度Sの下限を少し下げて拾いやすくする）
    y_h_min = st.sidebar.slider("黄: 色合い(H)の下限", 0, 180, 10)
    y_h_max = st.sidebar.slider("黄: 色合い(H)の上限", 0, 180, 30)
    mask_yellow = cv2.inRange(hsv_small, (y_h_min, 30, 140), (y_h_max, 255, 255)) # S_min 40->30, V_min 150->140

    # 緑（遅い藪）
    g_h_min = st.sidebar.slider("緑: 色合い(H)の下限", 0, 180, 35)
    mask_green = cv2.inRange(hsv_small, (g_h_min, 30, 50), (85, 255, 255))

    # ★追加：茶色（等高線。走行コストを上げる。Hは黄色と被るが、彩度Sが低く、明るさVも中程度）
    mask_brown = cv2.inRange(hsv_small, (10, 30, 50), (30, 150, 200))

    # 黒（道・崖・文字。等高線は茶色マスクで拾うため、黒は明るさVを低く設定）
    mask_black = cv2.inRange(hsv_small, (0, 0, 0), (180, 255, 90)) # V_max 120 -> 90

    # =========================
    # ③ 破線対応（膨張処理）はそのまま
    # =========================
    kernel = np.ones((3,3), np.uint8)
    mask_black_dilated = cv2.dilate(mask_black, kernel, iterations=1)

    # =========================
    # ④ 黒を「道 or 崖」に分類（★茶色（等高線）は道に入れない）
    # =========================
    road_mask = np.zeros_like(mask_black)
    wall_mask = np.zeros_like(mask_black)

    contours, _ = cv2.findContours(mask_black_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        ratio = max(cw, ch) / (min(cw, ch) + 1)
        # 破線などで小さい、かつ細長いものを道とする
        if area < 100 and ratio > 3:
            # ★茶色（等高線）と被る場合は、道としない
            mask_roi = mask_brown[y:y+ch, x:x+cw]
            if cv2.countNonZero(mask_roi) == 0:
                cv2.drawContours(road_mask, [cnt], -1, 255, -1)
        else:
            cv2.drawContours(wall_mask, [cnt], -1, 255, -1)

    # =========================
    # ⑤ コストマップ生成（★茶色のコスト追加、ズル防止の強化）
    # =========================
    small_cost = np.full((h_s, w_s), 5.0)  # 初期値（未知の色は遅い(5.0)とする）

    small_cost[mask_white > 0] = 1.0       # 白：標準ペース
    small_cost[mask_yellow > 0] = 0.8      # 黄色：オープン（走りやすい）
    small_cost[mask_green > 0] = 3.0       # 緑：遅い
    small_cost[road_mask > 0] = 0.5        # 道：爆速
    small_cost[wall_mask > 0] = 9999       # 崖・建物：通行不可
    small_cost[mask_brown > 0] = 1.5       # ★追加：茶色：等高線（斜面。少し遅い）

    # ズル防止策（画像の上下左右の端を「見えない壁」にする）
    margin = 8 # 5 -> 8
    small_cost[0:margin, :] = 9999     # 上端を壁に
    small_cost[-margin:, :] = 9999     # 下端を壁に
    small_cost[:, 0:margin] = 9999     # 左端を壁に
    small_cost[:, -margin:] = 9999     # 右端を壁に

    # ★ズル防止策の強化：地図の下の広い余白を「厚い見えない壁」にする
    # 縮小画像（small_cost）の下から10ピクセル分（実サイズだと50ピクセル分）を壁にする
    bottom_wall_thickness = 10
    small_cost[-bottom_wall_thickness:, :] = 9999
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
    # =========================
    # ⑧ ルート探索実行とペナルティ法
    # =========================
    # 安全装置
    if small_cost[start] >= 9999 or small_cost[goal] >= 9999:
        st.error("⚠️ スタートまたはゴールが通行不可エリア（黒線や枠外）に配置されています。スライダーを少しずらしてください。")
    else:
        with st.spinner('AIが複数のルートオプションを探索中...'):
            routes = []
            metrics = []
            # 描画用の色設定 (OpenCVはBGR形式)
            colors = [(0, 0, 255), (255, 0, 0), (0, 128, 0)] # 第1:赤, 第2:青, 第3:緑
            route_names = ["第1ルート (最適解)", "第2ルート (別ルート)", "第3ルート (大穴)"]
            
            # 探索用のコストマップをコピー（ここにペナルティを足していく）
            search_cost = small_cost.copy()
            
            for i in range(3):
                path = dijkstra(search_cost, start, goal)
                
                # ルートが見つからない場合はループを抜ける
                if not path or len(path) <= 1:
                    break
                    
                # 【指標計算】実際の地形コスト(small_cost)を使って難易度を計算する
                route_dist = len(path) # 距離: パスの長さ（ピクセル数）
                route_diff = sum(small_cost[p[0], p[1]] for p in path) # 難易度: 通過したピクセルのコスト合計
                
                routes.append(path)
                metrics.append({
                    "名前": route_names[i],
                    "色": ["🔴 赤", "🔵 青", "🟢 緑"][i],
                    "難易度スコア": round(route_diff, 1),
                    "相対距離": route_dist
                })
                
                # 【アルゴリズムの工夫：ペナルティ付与】
                # 見つけたルートの周辺のコストを一時的に爆上げして、強引に別のルートを探させる
                for p in path:
                    y, x = p
                    # ルートの上下左右3ピクセルに「+10.0」のペナルティを塗る
                    y_min, y_max = max(0, y-3), min(h_s, y+4)
                    x_min, x_max = max(0, x-3), min(w_s, x+4)
                    search_cost[y_min:y_max, x_min:x_max] += 10.0

        if not routes:
            st.warning("⚠️ ルートが見つかりませんでした。")
        else:
            # =========================
            # ⑨ 可視化 (画像への描画) と 比較ダッシュボード
            # =========================
            vis = img.copy()
            scale_inv = int(1 / scale)
            h_orig, w_orig = img.shape[:2]
            purple = (255, 0, 255) # オリエンテーリング記号の色

            orig_start = (int(w_orig * sx / 100), int(h_orig * sy / 100))
            orig_goal = (int(w_orig * gx / 100), int(h_orig * gy / 100))

            # コントロールポイントの描画
            cv2.circle(vis, orig_start, 30, purple, 5)
            cv2.circle(vis, orig_goal, 30, purple, 5)
            cv2.circle(vis, orig_goal, 18, purple, 3)

            # 複数ルートの描画 (後ろのルートから描画して、第1ルート(赤)が一番上に来るようにする)
            for i in reversed(range(len(routes))):
                path = routes[i]
                color = colors[i]
                for j in range(len(path) - 1):
                    pt1 = (path[j][1] * scale_inv, path[j][0] * scale_inv)
                    pt2 = (path[j+1][1] * scale_inv, path[j+1][0] * scale_inv)
                    cv2.line(vis, pt1, pt2, color, thickness=4)

            # 画像の表示
            st.subheader("🗺️ AIルート解析結果")
            st.image(vis, channels="BGR", caption="赤:最適解 / 青:第2候補 / 緑:第3候補", use_container_width=True)
            
            # =========================
            # ⑩ データ比較UIの表示
            # =========================
            st.subheader("📊 ルートごとのパフォーマンス比較")
            # 取得できたルートの数だけ画面を縦割りのカラムにする
            cols = st.columns(len(metrics))
            
            for i, col in enumerate(cols):
                m = metrics[i]
                with col:
                    st.markdown(f"**{m['色']} : {m['名前']}**")
                    # st.metric を使うと、株価や気温のようにカッコよく数値を強調表示できます
                    st.metric(label="難易度スコア (推定タイム)", value=m["難易度スコア"])
                    st.metric(label="移動距離 (ピクセル数)", value=m["相対距離"])
                    
            st.info("💡 **見方**: 「難易度スコア」が低いほど地形的に速く走れる理論値です。「距離」が短くても難易度が高い場合は、ヤブ漕ぎや急斜面が含まれていることを示します。")

else:
    st.info("上のボックスから地図画像をアップロードすると自動的に解析が始まります。")
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