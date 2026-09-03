"""相机与算法常量：只有相机 1749 完成标定，其余相机接入前必须先标 ROI 与阈值。

标定来源：experiments/barrier_bin.py（预注册验收 A1-A3 通过版本），
标定分辨率 1622x905（data/input 存档帧尺寸）。
"""
from dataclasses import dataclass

# 相机 1749：主入口实时流；1750 离线且未标定，暂不接入
RTSP_1749 = "rtsp://118.140.234.166:8554/dahua1002490"
DEFAULT_CAMERA = "1749"

# 标定分辨率：ROI/形态学核/面积阈值在此尺寸测得，接入实流时等比缩放
CALIB_W, CALIB_H = 1622, 905

# 红底座阈值（水马红 a*≈150-176；土坡/锈铁 a*≈120-130）
RED_H1, RED_H2, RED_H3, RED_H4 = 0, 10, 168, 180
RED_S_MIN, RED_A_MIN = 60, 145
WHITE_V_MIN, WHITE_S_MAX = 195, 60
# 远端带放宽阈值（远端粉底座 a*≈126-146、S 低、色相不稳）
FAR_RED_S_MIN, FAR_RED_A_MIN, FAR_RED_V_MIN = 25, 134, 140

# 形态学：open 去斑点；close 补缝；(1,21)/(21,1) 线段核沿路向把屏障带连片
OPEN_K, CLOSE_K, KNIT_H, KNIT_V = (3, 3), (9, 9), (1, 21), (21, 1)
MIN_CC_AREA, FAR_MIN_CC_AREA = 12, 6
# CC 均值色体检（防灰色路面/树影混入）：偏红 或 偏白 二选一
CC_A_REDISH, CC_V_WHITEISH, CC_S_WHITEISH = 136, 160, 80


@dataclass(frozen=True)
class Camera:
    id: str
    rtsp: str
    roi: tuple
    far_band: tuple
    exclude: tuple


# ROI 矩形（仅相机 1749 有效）：左排 / 桥头远端带 / 右排
ROI_1749 = ((0, 330, 900, 930), (1040, 295, 1462, 395), (880, 370, 1468, 950))
# 远端放宽阈值生效带：主带 + 左远列（粉底座低洼段）
FAR_BAND_1749 = ((1040, 295, 1462, 395), (1040, 375, 1170, 445))
# 烧录物/车误检排除框：时间戳、"1749"章、桥头白色面包车、橙色吊车驾驶室
EXCLUDE_1749 = ((1100, 25, 1595, 135), (15, 825, 125, 900),
                (1272, 280, 1318, 306), (1420, 272, 1458, 306))

# 缺口检测 slot（标定坐标）：矩形 + 基准覆盖率（GAP=0.80 为修复后假设值，待补齐后标定）
SLOTS_1749 = {
    "GAP": ((676, 415, 766, 585), 0.80),
    "NEAR": ((300, 470, 420, 690), 0.83),
    "MID": ((560, 400, 680, 620), 0.82),
    "FAR": ((1140, 300, 1200, 390), 0.33),
}
# 路面标定 ROI（三块日晒纯路面，供 RER 逐帧拟合；阴影路面另行扩样）
ROAD_ROIS_1749 = ((700, 700, 1000, 860), (1040, 560, 1240, 660), (600, 850, 900, 900))


def scale_rect(rect, w: int, h: int) -> tuple:
    """单个标定坐标矩形 -> 实流分辨率。"""
    sx, sy = w / CALIB_W, h / CALIB_H
    x0, y0, x1, y1 = rect
    return (round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))

def scale_cam(cam: Camera, w: int, h: int) -> Camera:
    """标定坐标 -> 实流分辨率，逐分量等比缩放（首帧时调用一次）。"""
    sx, sy = w / CALIB_W, h / CALIB_H

    def rects(rs):
        return tuple((round(x0 * sx), round(y0 * sy), round(x1 * sx), round(y1 * sy))
                     for x0, y0, x1, y1 in rs)
    return Camera(cam.id, cam.rtsp, rects(cam.roi), rects(cam.far_band), rects(cam.exclude))


CAMERAS = {
    "1749": Camera("1749", RTSP_1749, ROI_1749, FAR_BAND_1749, EXCLUDE_1749),
}
