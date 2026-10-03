"""Standalone execution-flow explanation; does not rebuild either report."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16); ax.set_ylim(0, 11); ax.axis('off')
ink, teal, blue, amber = '#183A45', '#087F7A', '#326798', '#98612B'


def box(x, y, w, h, title, body, color=teal, dashed=False):
    tint = '#FFF5E9' if color == amber else '#EDF3FA' if color == blue else '#EAF4F2'
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.035',
                 facecolor=tint, edgecolor=color, linewidth=1.4,
                 linestyle='--' if dashed else '-', zorder=2))
    ax.text(x+w/2, y+h*.76, title, ha='center', va='center', fontsize=12,
            weight='bold', color=ink, zorder=3)
    ax.text(x+w/2, y+h*.33, body, ha='center', va='center', fontsize=10.7,
            color=ink, linespacing=1.4, zorder=3)


def route(points, color=ink, dashed=False):
    ax.plot([x for x, _ in points[:-1]], [y for _, y in points[:-1]],
            color=color, lw=1.5, ls='--' if dashed else '-', zorder=1)
    ax.annotate('', xy=points[-1], xytext=points[-2],
                arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 1.5,
                            'linestyle': '--' if dashed else '-'}, zorder=1)


def label(x, y, s, color=ink, ha='center'):
    ax.text(x, y, s, fontsize=10, color=color, ha=ha, va='center',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 2}, zorder=4)


ax.text(8, 10.75, 'LUỒNG XỬ LÝ BR-DUMMY TRÊN NỀN GEO-I',
        ha='center', fontsize=19, weight='bold', color=ink)
ax.text(8, 10.36, 'Đọc theo số: 1 → 2 → 3a / 3b → 4 → server → 5. A / B là boundary tùy chọn.',
        ha='center', fontsize=12, color=ink)
ax.add_patch(FancyBboxPatch((3.7, .65), 9.15, 9.35, boxstyle='round,pad=.04',
             facecolor='#F8FBFA', edgecolor=teal, linewidth=2, zorder=0))
ax.text(4.05, 9.65, 'KHUNG MÔ HÌNH CỦA TA', color=teal, fontsize=13, weight='bold')
ax.text(4.05, 8.95, 'A + B = BR-Boundary v1\nTên wrapper, không phải\nmột bước xử lý thứ ba.',
        color=amber, fontsize=11, va='top', linespacing=1.5)

box(.2, 8.68, 2.8, 1.0, 'INPUT RIÊNG', 'GPS hiện tại\nchỉ ở thiết bị', blue)
box(.2, 5.15, 2.8, 1.25, 'INPUT CÔNG KHAI', 'Mạng làn / luật rẽ, POI\nthời gian, B, K, h, Δ', blue)
box(.2, 2.12, 2.8, .95, 'TÍN HIỆU PHIÊN', 'Kết thúc khi xảy ra\nkhông biết trước đích', blue)

box(8, 8.68, 4, 1.0, 'A · S9: bỏ đầu trước lõi', 'Khi bật: h giây đầu không vào lõi\nKhi tắt: đi thẳng tới bước 1', amber, True)
box(8, 7.25, 4, 1.04, '1 · NGÂN SÁCH + PACING', 'Đến lịch đọc GPS và còn ngân sách?\nCận phiên 0,23 /m; đọc cách ≥60 s')
box(8, 5.82, 4, 1.04, '2 · NEO GEO-I / REM', 'Private test: giữ neo hoặc tạo neo mới\nKhông công bố GPS thật / neo nội bộ')
box(8, 4.39, 4, 1.04, '3a · BELIEF TỪ NEO', 'Có đọc mới: cập nhật từ neo\nKhông đọc mới: dự đoán từ lịch sử')
box(4.1, 4.39, 3.05, 1.04, '3b · ROAD-AWARE', 'Dummy trước + Δt + mạng làn\n→ miền đi tới được')
box(8, 2.96, 4, 1.04, '4 · POI-AWARE SELECTOR', 'Belief + miền khả thi + danh mục POI\n→ chọn K=5; progress / slack 0,03')
box(8, 1.65, 4, .94, 'B · S10: buffer sau lõi', 'Khi bật: trì hoãn Δ; close → hủy\nKhi tắt: gửi ngay output của bước 4', amber, True)
box(8, .80, 4, .58, '5 · CACHE + TOP-5 CỤC BỘ', 'Hợp POI cùng epoch 60 s; GPS xếp hạng')

box(13.25, 1.65, 2.55, 1.32, 'OUTPUT CÔNG KHAI', 'K tọa độ + thời điểm phát\n↓\nServer trả POI', blue)
box(13.25, .80, 2.55, .58, 'OUTPUT RIÊNG', 'Top-5 POI cho người dùng', blue)

# Execution spine and the explicit no-private-read branch.
route([(3.0, 9.18), (8, 9.18)], blue)
route([(10, 8.68), (10, 8.29)], amber, True)
route([(10, 7.25), (10, 6.86)])
label(10.45, 7.05, 'Có')
route([(10, 5.82), (10, 5.43)])
route([(12, 7.75), (12.50, 7.75), (12.50, 4.91), (12, 4.91)])
label(12.48, 6.31, 'Không\nđọc mới')
route([(10, 4.39), (10, 4.00)])
route([(7.15, 4.91), (7.48, 4.91), (7.48, 3.48), (8, 3.48)])
label(6.04, 3.76, 'Miền khả thi\ncho bộ chọn')
route([(10, 2.96), (10, 2.59)], amber, True)
route([(12, 2.12), (13.25, 2.12)], blue)
route([(14.52, 1.65), (14.52, 1.48), (10, 1.48), (10, 1.38)], blue)
label(12.75, 1.48, 'Phản hồi POI', blue)
route([(12, 1.08), (13.25, 1.08)], blue)

# Context and private local ranking are separate from the protected query path.
route([(3, 5.77), (3.43, 5.77), (3.43, 6.70), (5.63, 6.70), (5.63, 5.43)], blue, True)
label(5.62, 6.70, 'Mạng làn, Δt, dummy trước', blue)
route([(3, 5.58), (3.55, 5.58), (3.55, 3.48), (4.5, 3.48), (4.5, 2.78), (8.35, 2.78), (8.35, 2.96)], blue, True)
label(5.50, 2.78, 'Danh mục POI công khai', blue)
route([(3, 2.58), (3.30, 2.58), (3.30, 1.84), (8, 1.84)], amber, True)
label(5.70, 1.84, 'Close: hủy buffer chưa phát', amber)
route([(.2, 9.00), (.06, 9.00), (.06, 1.08), (8, 1.08)], blue, True)
label(4.78, 1.08, 'GPS riêng chỉ dùng xếp hạng ở bước 5', blue)

ax.text(8, .34, 'Đường liền: luồng xử lý / dữ liệu · Xanh nét đứt: dữ liệu phụ trợ · Nâu: boundary tùy chọn',
        ha='center', fontsize=10.5, color=ink)
ax.text(8, .04, 'Lõi đã có benchmark; Boundary đã kiểm tra tích hợp, chưa benchmark kết hợp toàn bộ dịch vụ.',
        ha='center', fontsize=10.5, color=amber)
for extension in ('pdf', 'svg', 'png'):
    fig.savefig(OUT / f'architecture_flow_explained.{extension}', bbox_inches='tight', dpi=180)
plt.close(fig)
