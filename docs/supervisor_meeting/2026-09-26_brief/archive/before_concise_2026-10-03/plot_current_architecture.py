"""Final public category-query architecture; GPS stays in local ranking."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ink = '#293a40'

def box(x, y, w, h, title, body, tint=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.004',
        facecolor='#f0f4f3' if tint else 'white', edgecolor=ink, linewidth=1.0))
    ax.text(x+w/2, y+h*.73, title, ha='center', va='center', weight='bold', fontsize=9.7)
    ax.text(x+w/2, y+h*.30, body, ha='center', va='center', fontsize=8.6, linespacing=1.35)

def arrow(a, b, dashed=False):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': ink,
        'lw': 1.05, 'linestyle': '--' if dashed else '-'})

ax.text(.04,.95,'CHUẨN BỊ TỪ DỮ LIỆU CÔNG KHAI',fontsize=11,weight='bold',color=ink)
box(.035,.69,.23,.18,'Mạng đường và POI','Danh mục, loại POI\nquy tắc truy vấn top-10')
box(.345,.69,.27,.18,'Chọn tập phủ theo loại','Tham lam phần chưa phủ\nkhông nhận GPS / tuyến thật')
box(.695,.69,.27,.18,'Kế hoạch cố định','30 hoặc 67 query loại\n19 hoặc 52 tọa độ')
arrow((.265,.78),(.345,.78));arrow((.615,.78),(.695,.78))
box(.035,.365,.23,.18,'Lịch đăng ký trước','Mỗi 60 s trong trọn một giờ\nkể cả khi không có chuyến')
box(.695,.365,.27,.18,'Máy chủ POI khả dụng','Trả top-10 theo loại–tọa độ\nkèm phiên bản / epoch',True)
arrow((.83,.69),(.83,.545))
ax.plot([.265,.61,.61],[.455,.455,.63],color=ink,lw=1.05)
arrow((.61,.63),(.83,.63))
box(.695,.055,.27,.18,'Hợp phản hồi hợp lệ','Cache trong cùng epoch\nloại dữ liệu hết hiệu lực')
box(.345,.055,.27,.18,'Xếp hạng cục bộ','Top-5 theo vị trí thật\nkhông gửi lựa chọn',True)
box(.035,.055,.23,.18,'GPS thật','Chỉ vào khối xếp hạng\ntrên thiết bị',True)
arrow((.83,.365),(.83,.235));arrow((.695,.145),(.615,.145));arrow((.265,.145),(.345,.145),True)
ax.text(.47,.37,'YÊU CẦU / PHẢN HỒI',ha='center',fontsize=9,color=ink)
for extension in ('pdf','svg','png'):
    fig.savefig(OUT/f'architecture_current.{extension}',bbox_inches='tight',dpi=180)
plt.close(fig)
