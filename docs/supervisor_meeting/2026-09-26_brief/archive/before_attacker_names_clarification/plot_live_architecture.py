"""Current point-service architecture; no boundary buffer or future data path."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(11, 5.3))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ink, accent = '#283b43', '#326766'
ax.add_patch(FancyBboxPatch((.025, .14), .755, .76, boxstyle='round,pad=.006',
    facecolor='#f7f8f8', edgecolor='#9aa7aa', linestyle='--'))
ax.text(.04, .94, 'THIẾT BỊ', weight='bold', color=ink)
ax.text(.805, .94, 'MÁY CHỦ', weight='bold', color=ink)

def box(x, y, w, h, title, body):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.005',
        facecolor='white', edgecolor=accent, linewidth=1.1))
    ax.text(x+w/2, y+h*.73, title, ha='center', va='center', weight='bold', fontsize=9.5)
    ax.text(x+w/2, y+h*.30, body, ha='center', va='center', fontsize=8.5, linespacing=1.4)

def arrow(a, b, dashed=False):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '->', 'color': ink,
        'lw': 1.1, 'linestyle': '--' if dashed else '-'})

box(.05,.70,.19,.16,'1. Vị trí thật','GPS hiện tại\nchỉ ở thiết bị')
box(.29,.70,.21,.16,'2. Neo riêng tư','Lịch đọc + ngân sách\nkiểm tra / tạo nhiễu')
box(.55,.70,.21,.16,'3. Chọn K = 5','Belief từ neo\nphủ POI, đường, slack')
box(.815,.70,.16,.16,'4. API theo điểm','Trạng thái khả dụng\ntrả top-L = 10')
box(.05,.43,.19,.15,'Ngữ cảnh công khai','Mạng đường, POI\nlịch và tham số')
box(.55,.22,.21,.16,'5. Cache phản hồi','Hợp trong cùng epoch\nxóa khi hết hiệu lực')
box(.29,.22,.21,.16,'6. Lọc cục bộ','Xếp hạng bằng GPS thật\nchọn top-5 khả dụng')
box(.05,.22,.19,.16,'7. Kết quả','POI cho người dùng\nkhông gửi lựa chọn')
arrow((.24,.78),(.29,.78)); arrow((.50,.78),(.55,.78)); arrow((.76,.78),(.815,.78))
ax.text(.79,.875,'K tọa độ',ha='center',fontsize=8,color=ink)
ax.plot([.895,.895,.985,.985],[.70,.65,.65,.30],color=ink,lw=1.1)
arrow((.985,.30),(.76,.30))
ax.text(.88,.48,'ID POI và\nepoch phản hồi',ha='center',fontsize=8.5,color=ink)
arrow((.55,.30),(.50,.30)); arrow((.29,.30),(.24,.30))
ax.plot([.145,.145,.395],[.70,.625,.625],color=ink,lw=1,ls='--')
arrow((.395,.625),(.395,.38),True)
ax.text(.41,.48,'GPS nội bộ',fontsize=8,color=ink)
arrow((.24,.53),(.55,.70))
ax.text(.62,.54,'Mô hình dịch vụ\ncông khai',ha='center',fontsize=8,color=ink)
ax.text(.05,.075,'Cache không đổi truy vấn đã gửi; bộ chọn không nhận toàn bộ trạng thái khả dụng.',
    color=ink,fontsize=9)
for extension in ('pdf','svg','png'):
    fig.savefig(OUT/f'architecture_live.{extension}',bbox_inches='tight',dpi=180)
plt.close(fig)
