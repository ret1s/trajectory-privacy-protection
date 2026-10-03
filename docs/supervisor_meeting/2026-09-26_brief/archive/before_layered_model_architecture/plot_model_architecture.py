"""BR-Dummy model components, with the separately implemented boundary wrapper."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(12, 6.0))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ink, teal, blue, amber = '#183A45', '#087F7A', '#326798', '#98612B'


def box(x, y, w, h, title, body, color=teal, tint='#EAF4F2', dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.005',
        facecolor=tint, edgecolor=color, linewidth=1.2,
        linestyle='--' if dashed else '-'))
    ax.text(x+w/2, y+h*.77, title, ha='center', va='center', weight='bold', fontsize=10, color=ink)
    ax.text(x+w/2, y+h*.32, body, ha='center', va='center', fontsize=9.2, color=ink, linespacing=1.4)


def arrow(a, b, color=ink, dashed=False):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'color': color,
        'lw': 1.4, 'linestyle': '--' if dashed else '-'})


ax.text(.02, .96, 'LÕI BR-DUMMY (S1–S3): GEO-I + ROAD-AWARE + POI-AWARE',
        fontsize=12, weight='bold', color=teal)
box(.02,.69,.12,.18,'GPS thật','Vị trí x(t)\nở thiết bị',blue,'#EDF3FA')
box(.20,.69,.23,.18,'(1) Neo Geo-I / REM','Cơ chế mũ trên support đường\nTest có nhiễu: giữ / tạo neo')
box(.50,.69,.21,.18,'(2) Belief từ neo','Ước lượng vị trí / chuyển động\nchỉ dùng lịch sử đã bảo vệ')
box(.77,.69,.21,.18,'(3) Road-aware','Làn có hướng, luật rẽ\nmiền tới được theo thời gian')
arrow((.145,.78),(.195,.78));arrow((.435,.78),(.495,.78));arrow((.715,.78),(.765,.78))
box(.20,.41,.23,.18,'Ngân sách + nhịp đọc','Cộng chi phí test / neo mới\nhết ngân sách: hậu xử lý',blue,'#EDF3FA')
arrow((.315,.595),(.315,.685),blue)
box(.02,.16,.41,.17,'Ngữ cảnh công khai','Mạng đường / làn + POI + thời gian / tham số\nKhông dùng GPS, tốc độ thật hoặc đích tương lai',ink,'#F6F6F2')
ax.plot([.435,.47,.47,.605],[.30,.30,.61,.61],color=ink,ls=':',lw=1.2)
arrow((.605,.61),(.605,.685),ink,True)
box(.77,.41,.21,.18,'(4) POI-aware selector','Chọn K điểm: phủ hợp POI\ntham lam + exchange / slack')
arrow((.875,.685),(.875,.595))
box(.50,.41,.21,.18,'Đầu ra lõi','K vị trí / track dummy\nneo không gửi ra ngoài')
arrow((.765,.50),(.715,.50))
ax.text(.02,.915,'Sau neo: hậu xử lý từ dữ liệu đã bảo vệ và ngữ cảnh công khai.',fontsize=9,color=teal)

box(.02,.015,.41,.105,'S9: bỏ đầu trước lõi BR','Trong h giây đầu: không đưa GPS đó vào cơ chế neo',amber,'#FFF5E9',True)
ax.plot([.014,.005,.005,.165,.165],[.067,.067,.635,.635,.78],color=amber,ls='--',lw=1.2)
arrow((.165,.78),(.195,.78),amber,True)
box(.50,.16,.48,.13,'S10: buffer sau đầu ra lõi','Giữ bản tin Δ giây; chỉ phát bản tin đủ tuổi\nkết thúc phiên: hủy phần chưa phát, không flush',amber,'#FFF5E9',True)
arrow((.605,.405),(.605,.295),amber,True)
ax.text(.50,.085,'BR-Boundary v1: lớp mở rộng đã có code / kiểm tra tích hợp;',fontsize=9,color=amber)
ax.text(.50,.035,'chưa có benchmark toàn bộ Geo-I + boundary + dịch vụ.',fontsize=9,color=amber)
for extension in ('pdf','svg','png'):
    fig.savefig(OUT/f'architecture_model.{extension}',bbox_inches='tight',dpi=180)
plt.close(fig)
svg=OUT/'architecture_model.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
