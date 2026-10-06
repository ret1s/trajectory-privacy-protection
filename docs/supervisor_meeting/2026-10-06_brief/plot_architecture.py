"""Layered, vector architecture for the next supervisor meeting."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path(__file__).resolve().parent / 'figures'
INK = '#193c49'
TEAL = '#147d7b'
GRAY = '#60727b'
ORANGE = '#b87728'


def draw():
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none'})
    fig, ax = plt.subplots(figsize=(11.2, 10.6))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.set(xlim=(0, 112), ylim=(0, 106))
    ax.axis('off')

    def box(x, y, width, height, text, *, fill='#ffffff', color=INK, size=13, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0.4,rounding_size=1.0', lw=1.3,
            facecolor=fill, edgecolor=color))
        ax.text(x+width/2, y+height/2, text, ha='center', va='center',
                fontsize=size, color=INK, weight='bold' if bold else 'normal',
                linespacing=1.4)

    def arrow(points, *, color=TEAL, dashed=False):
        for a, b in zip(points, points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.6,
                    ls='--' if dashed else '-')
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            mutation_scale=13, color=color, lw=1.6,
            linestyle='--' if dashed else '-'))

    box(9, 95, 27, 9, 'INPUT RIÊNG TƯ\nGPS thật tại thiết bị', fill='#f5f8fa', color=GRAY)
    box(42, 95, 43, 9, 'INPUT CÔNG KHAI\nMạng đường, POI, prior, đồng hồ', fill='#f5f8fa', color=GRAY)
    ax.add_patch(FancyBboxPatch((7, 12), 78, 79,
        boxstyle='round,pad=0.4,rounding_size=1.2', lw=2.0,
        facecolor='#f9fcfb', edgecolor=TEAL))
    ax.text(46, 87.5, 'MÔ HÌNH CỦA TA · CHẠY TẠI THIẾT BỊ',
            ha='center', fontsize=14, color=TEAL, weight='bold')

    bands = [(65, 19, 'LAYER 1 · BẢO VỆ GPS'),
             (46, 17, 'LAYER 2 · ƯỚC LƯỢNG VÀ CHỌN Q'),
             (25, 19, 'LAYER 3 · LẤY POI · KHÔNG WARMUP/DELAY'),
             (14, 9, 'LAYER 4 · TRẢ LỜI LOCAL')]
    for y, height, title in bands:
        ax.add_patch(FancyBboxPatch((9, y), 74, height,
            boxstyle='round,pad=0.25,rounding_size=0.7',
            facecolor='#eaf4f2', edgecolor='#c6dcda', lw=.7))
        ax.text(10.5, y+height-2, title, fontsize=10.5, color=TEAL, weight='bold')

    box(11, 67, 29, 12, 'Lịch đọc + ngân sách chung\nĐúng lịch, còn cap mới đọc', size=12)
    box(45, 67, 36, 12, 'Geo-I trên mạng đường\nThử giữ Z có nhiễu / tạo mới', bold=True, size=12)
    arrow([(36, 96), (36, 93), (25.5, 93), (25.5, 79)])
    arrow([(63.5, 95), (63.5, 92), (80, 92), (80, 79)], color=GRAY, dashed=True)
    arrow([(80, 92), (86, 92), (86, 56), (81, 56)], color=GRAY, dashed=True)
    arrow([(40, 73), (45, 73)])

    box(11, 48, 30, 11, 'Ước lượng vùng vị trí\nTừ Z và lịch sử đã bảo vệ', size=12)
    box(46, 48, 35, 11, 'Chọn K=5 điểm Q\nĐi được trên đường, phủ POI', size=12)
    arrow([(63, 67), (63, 63.5), (26, 63.5), (26, 59)])
    ax.text(48, 64.5, 'Z nội bộ', fontsize=10, color=TEAL)
    arrow([(41, 53.5), (46, 53.5)])
    arrow([(11, 69), (8, 69), (8, 56), (11, 56)], color=GRAY, dashed=True)
    ax.text(10.5, 64.3, 'Phiên đã cấp cap: không đọc mới → dự đoán', ha='left',
            fontsize=8.5, color=GRAY)

    box(11, 35, 70, 6, 'Xáo thứ tự Q · request chung · L cố định · mọi category', size=11.5)
    box(11, 27, 70, 6, 'Hợp POI còn hiệu lực, bỏ trùng · cache theo chính sách công khai', size=11)
    arrow([(63, 48), (63, 41)])
    box(90, 34, 20, 16, 'SERVER\nNhận Q\nTrả top-L/loại', fill='#fff4e8', color=ORANGE, size=12, bold=True)
    arrow([(81, 38), (90, 38)], color=ORANGE)
    ax.text(87, 40, 'Q', ha='center', fontsize=11, color=ORANGE)
    arrow([(100, 34), (100, 30), (81, 30)], color=ORANGE)
    ax.text(91, 27.5, 'POI', ha='center', fontsize=10.5, color=ORANGE)

    box(11, 14.5, 70, 5.3, 'GPS hiện tại + purpose riêng → lọc và xếp hạng POI', size=11.5)
    arrow([(46, 27), (46, 19.8)])
    box(90, 13, 20, 12, 'INPUT LOCAL\nPurpose, category,\nradius, destination', fill='#f5f8fa', color=GRAY, size=10.5)
    arrow([(90, 17), (81, 17)], color=GRAY, dashed=True)
    arrow([(11, 95), (2, 95), (2, 17), (11, 17)], color=GRAY, dashed=True)
    ax.text(2.7, 39, 'GPS chỉ xếp hạng local', rotation=90, fontsize=10, color=GRAY)
    box(25, 1, 43, 8, 'OUTPUT CHO NGƯỜI DÙNG\nP: top-k POI phù hợp, mặc định k=5', fill='#eaf4f2', color=TEAL, size=12)
    arrow([(46, 14.5), (46, 9)])

    for suffix in ('svg', 'pdf', 'png'):
        fig.savefig(OUT/f'architecture.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    draw()
