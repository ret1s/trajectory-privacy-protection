"""Presentation diagram: public plan, fixed schedule and local GPS ranking."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(12, 5.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ink, teal, blue = '#183A45', '#087F7A', '#326798'

def box(x, y, w, h, title, body, color=teal, tint='#EAF4F2'):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.006',
        facecolor=tint, edgecolor=color, linewidth=1.3))
    ax.text(x+w/2, y+h*.76, title, ha='center', va='center', weight='bold', fontsize=10, color=ink)
    ax.text(x+w/2, y+h*.31, body, ha='center', va='center', fontsize=9.3, color=ink, linespacing=1.4)

def arrow(a, b, color=ink, dashed=False):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'color': color,
        'lw': 1.5, 'linestyle': '--' if dashed else '-'})

ax.text(.02,.97,'CHUẨN BỊ CÔNG KHAI',fontsize=11,weight='bold',color=teal)
box(.02,.68,.24,.21,'Bản đồ + danh mục POI','419 POI, 6 loại\nỨng viên truy vấn trên đường')
box(.34,.68,.29,.21,'(1) Chọn tập phủ, khóa kế hoạch','30 / 67 query loại–tọa độ\n19 / 52 tọa độ cố định')
box(.73,.68,.25,.21,'(2) Lịch đăng ký cố định','Mỗi 60 s, trọn một giờ\nKể cả không đi / không dùng',blue,'#EDF3FA')
arrow((.265,.79),(.33,.79));arrow((.635,.79),(.72,.79))
ax.text(.48,.60,'Không phụ thuộc GPS: S1, S2, S3',ha='center',color=teal,fontsize=10,weight='bold')
ax.text(.73,.97,'LỊCH CÔNG KHAI: S9, S10',color=blue,fontsize=10.5,weight='bold')

box(.73,.30,.25,.21,'Máy chủ dịch vụ','Nhận loại + tọa độ công khai\nTrả top-10 POI khả dụng',ink,'#F6F6F2')
arrow((.855,.67),(.855,.52),blue)
ax.text(.705,.565,'Query',ha='right',color=blue,fontsize=9)
box(.38,.30,.26,.21,'(3) Cache trên thiết bị','Hợp phản hồi còn hiệu lực\nĐọc cache không gửi query')
arrow((.725,.41),(.65,.41))
ax.text(.687,.45,'POI',ha='center',fontsize=9,color=ink)
box(.02,.30,.25,.21,'(4) Xếp hạng cục bộ','Top-5 theo GPS thật\nKhông gửi lựa chọn ra ngoài')
arrow((.373,.41),(.28,.41))
box(.02,.035,.25,.15,'GPS thật','Chỉ đưa vào xếp hạng',blue,'#EDF3FA')
arrow((.145,.19),(.145,.29),blue,True)
ax.text(.4,.15,'S1: vị trí  •  S2: nơi dừng  •  S3: đường đi',color=teal,fontsize=10)
ax.text(.4,.09,'S9 / S10 cần cả kế hoạch độc lập GPS và lịch cố định.',color=blue,fontsize=10)
for extension in ('pdf','svg','png'):
    fig.savefig(OUT/f'architecture_current.{extension}',bbox_inches='tight',dpi=180)
plt.close(fig)
# Keep the generated SVG free of trailing whitespace for repository checks.
svg = OUT/'architecture_current.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
