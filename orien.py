
import cv2
import numpy as np
import heapq

# =========================
# ① 画像読み込み & 前処理
# =========================
img = cv2.imread("map.png")
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

    # 判定ルール
    if area < 100 and ratio > 3:
        cv2.drawContours(road_mask, [cnt], -1, 255, -1)
    else:
        cv2.drawContours(wall_mask, [cnt], -1, 255, -1)

# =========================
# ⑤ コストマップ生成
# =========================
cost = np.full((h, w), 5.0)  # 初期値

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
    h, w = cost_map.shape
    dist = np.full((h, w), np.inf)
    prev = np.full((h, w, 2), -1)

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
            if 0 <= ny < h and 0 <= nx < w:
                nd = d + cost_map[ny,nx]
                if nd < dist[ny,nx]:
                    dist[ny,nx] = nd
                    prev[ny,nx] = [y,x]
                    heapq.heappush(pq, (nd, (ny,nx)))

    # 経路復元
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
# ⑧ スタート・ゴール設定
# =========================
start = (int(h*scale*0.1), int(w*scale*0.1))
goal  = (int(h*scale*0.8), int(w*scale*0.8))

path = dijkstra(small_cost, start, goal)

# =========================
# ⑨ 可視化
# =========================
vis = cv2.resize(img, (small_cost.shape[1], small_cost.shape[0]))

for y,x in path:
    vis[y,x] = [0,0,255]

cv2.imwrite("result.png", vis)
print("result.png に保存しました")
