"""Portrait execution-flow figure sized for readable text in the report."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent / 'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(7.6, 8.4))
ax.set_xlim(0, 7.6); ax.set_ylim(0, 8.4); ax.axis('off')
ink, teal, blue, amber = '#183A45', '#087F7A', '#326798', '#98612B'


def box(x, y, w, h, title, body, color=teal, dashed=False):
    tint = '#FFF5E9' if color == amber else '#EDF3FA' if color == blue else '#EAF4F2'
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.025',
                 facecolor=tint, edgecolor=color, linewidth=1.1,
                 linestyle='--' if dashed else '-', zorder=2))
    ax.text(x+w/2, y+h*.76, title, ha='center', va='center', fontsize=9.0 if w < 2 else 9.5,
            weight='bold', color=ink, zorder=3)
    ax.text(x+w/2, y+h*.32, body, ha='center', va='center', fontsize=8.2 if w < 2 else 8.6,
            color=ink, linespacing=1.3, zorder=3)


def route(points, color=ink, dashed=False):
    ax.plot([x for x, _ in points[:-1]], [y for _, y in points[:-1]],
            color=color, lw=1.2, ls='--' if dashed else '-', zorder=1)
    ax.annotate('', xy=points[-1], xytext=points[-2],
                arrowprops={'arrowstyle': '-|>', 'color': color, 'lw': 1.2,
                            'linestyle': '--' if dashed else '-'}, zorder=1)


def label(x, y, s, color=ink):
    ax.text(x, y, s, fontsize=8.2, color=color, ha='center', va='center',
            bbox={'facecolor': 'white', 'edgecolor': 'none', 'pad': 1.3}, zorder=4)


ax.add_patch(FancyBboxPatch((.12, .47), 5.62, 6.9, boxstyle='round,pad=.035',
             facecolor='#F8FBFA', edgecolor=teal, linewidth=1.6, zorder=0))
ax.text(.35, 7.12, 'MÔ HÌNH CỦA TA', fontsize=10.2, color=teal, weight='bold')
ax.text(.40, 6.76, 'Che đầu/cuối\nS9/S10 tùy chọn.',
        fontsize=8.5, color=amber, va='top', linespacing=1.45)
for y, text in ((5.88, 'LỚP 1\nBẢO VỆ GPS'), (4.55, 'LỚP 2 · ƯỚC LƯỢNG'),
                (3.45, 'LỚP 3\nTÌM ĐỊA ĐIỂM')):
    ax.text(.40, y, text, fontsize=8.5, color=teal, weight='bold',
            va='top', linespacing=1.5)

box(.30, 7.65, 1.70, .61, 'DỮ LIỆU CHUNG', 'Bản đồ, POI\nthời gian, tham số', blue)
box(2.25, 7.65, 3.15, .61, 'GPS TRÊN THIẾT BỊ', 'Chỉ dùng tạo truy vấn khi qua bước 1', blue)

box(2.25, 6.60, 3.15, .61, 'S9: CHE ĐẦU CHUYẾN', 'Bật: bỏ h giây đầu\nTắt: đi thẳng tới bước 1', amber, True)
box(2.25, 5.60, 3.15, .61, '1 · GIỚI HẠN + LỊCH ĐỌC GPS', 'Đến lịch và còn ngân sách?\nCận 0,23 /m; đọc cách ≥60 s')
box(2.25, 4.60, 3.15, .61, '2 · GEO-I: LÀM NHIỄU GPS', 'Giữ hoặc tạo vị trí đã bảo vệ\nVị trí này chỉ dùng trong thiết bị')
box(2.25, 3.60, 3.15, .61, '3a · ƯỚC LƯỢNG VÙNG VỊ TRÍ', 'Dựa trên lịch sử vị trí đã bảo vệ\nKhông đọc mới: dự đoán từ lịch sử')
box(.35, 3.60, 1.65, .61, '3b · ĐƯỜNG ĐI', 'Đường + thời gian\n→ điểm giả có thể tới')
box(2.25, 2.60, 3.15, .61, '4 · CHỌN ĐIỂM TRUY VẤN', '5 điểm tìm POI bổ sung nhau\nLinh hoạt để tiếp tục di chuyển')
box(2.25, 1.60, 3.15, .61, 'S10: CHE CUỐI CHUYẾN', 'Bật: giữ tạm kết quả trước khi gửi\nKết thúc: hủy phần chưa gửi', amber, True)
box(2.25, .60, 3.15, .61, '5 · HỢP KẾT QUẢ + CHỌN TOP-5', 'Giữ phản hồi còn hiệu lực 60 s\nGPS thật xếp hạng tại thiết bị')
box(6.05, 1.60, 1.30, .95, 'CÔNG KHAI', '5 điểm\n+ thời điểm gửi\n↓ Trả POI', blue)
box(6.05, .60, 1.30, .61, 'RIÊNG', '5 POI mỗi loại\ncho người dùng', blue)
box(.35, .04, 1.70, .28, 'KẾT THÚC CHUYẾN', '', blue)
box(6.05, .04, 1.30, .28, 'GPS THẬT', '', blue)

route([(3.82, 7.65), (3.82, 7.21)], blue)
route([(3.82, 6.60), (3.82, 6.21)], amber, True)
route([(3.82, 5.60), (3.82, 5.21)])
label(4.08, 5.40, 'Có')
route([(3.82, 4.60), (3.82, 4.21)])
route([(5.40, 5.91), (5.60, 5.91), (5.60, 3.91), (5.40, 3.91)])
label(5.60, 4.37, 'Không\nđọc mới')
route([(3.82, 3.60), (3.82, 3.21)])
route([(2.00, 3.91), (2.12, 3.91), (2.12, 3.02), (2.25, 3.02)])
route([(3.82, 2.60), (3.82, 2.21)], amber, True)
route([(5.40, 1.91), (6.05, 1.91)], blue)
route([(6.70, 1.60), (6.70, 1.42), (3.82, 1.42), (3.82, 1.21)], blue)
label(5.18, 1.42, 'Phản hồi POI', blue)
route([(5.40, .97), (6.05, .97)], blue)

route([(1.15, 7.65), (.22, 7.65), (.22, 4.40), (1.18, 4.40), (1.18, 4.21)], blue, True)
label(1.15, 4.31, 'Đường + thời gian')
route([(.22, 4.40), (.22, 2.91), (2.25, 2.91)], blue, True)
label(1.20, 2.91, 'Danh mục POI')
route([(1.18, .32), (1.18, 1.91), (2.25, 1.91)], amber, True)
label(1.15, 1.52, 'Kết thúc:\nhủy phần chưa gửi')
route([(6.05, .18), (5.54, .18), (5.54, .78), (5.40, .78)], blue, True)

for extension in ('pdf', 'svg', 'png'):
    fig.savefig(OUT / f'architecture_report_flow.{extension}', bbox_inches='tight', dpi=180)
plt.close(fig)
