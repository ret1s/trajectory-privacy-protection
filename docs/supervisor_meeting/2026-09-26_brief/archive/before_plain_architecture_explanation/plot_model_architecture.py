"""Layered BR-Dummy architecture with explicit inputs, outputs and model boundary."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(14, 5.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ink, teal, blue, amber = '#183A45', '#087F7A', '#326798', '#98612B'


def box(x, y, w, h, title, body, color=teal, tint='#EAF4F2', dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.004',
        facecolor=tint, edgecolor=color, linewidth=1.2,
        linestyle='--' if dashed else '-'))
    ax.text(x+w/2, y+h*.78, title, ha='center', va='center', weight='bold', fontsize=10, color=ink)
    ax.text(x+w/2, y+h*.34, body, ha='center', va='center', fontsize=9.2, color=ink, linespacing=1.4)


def arrow(a, b, color=ink, dashed=False):
    ax.annotate('', xy=b, xytext=a, arrowprops={'arrowstyle': '-|>', 'color': color,
        'lw': 1.35, 'linestyle': '--' if dashed else '-'})


def route(points, color=ink, dashed=False):
    ax.plot([p[0] for p in points[:-1]], [p[1] for p in points[:-1]],
            color=color, lw=1.2, ls='--' if dashed else '-')
    arrow(points[-2], points[-1], color, dashed)


# A solid enclosing frame identifies authored mechanisms, excluding inputs and consumers.
ax.add_patch(FancyBboxPatch((.205,.13),.60,.795,boxstyle='round,pad=.008',
    facecolor='#F7FAF9',edgecolor=teal,linewidth=2.0,zorder=0))
ax.text(.505,.955,'MÔ HÌNH CỦA TA: BR-DUMMY TRÊN NỀN GEO-I',
        ha='center',fontsize=12,weight='bold',color=teal)
headers=[(.095,'INPUT'),(.305,'LAYER 1 · BẢO VỆ'),(.51,'LAYER 2 · NGỮ CẢNH'),
         (.715,'LAYER 3 · TRUY VẤN'),(.91,'OUTPUT')]
for x,title in headers:
    ax.text(x,.87,title,ha='center',fontsize=10.2,weight='bold',color=blue if x in (.095,.91) else teal)
for x in (.4075,.6125):
    ax.plot([x,x],[.34,.84],color='#C2D5D0',ls=':',lw=.9)

box(.02,.62,.15,.18,'Dữ liệu riêng tư','GPS hiện tại x(t)\nchỉ ở thiết bị',blue,'#EDF3FA')
box(.02,.38,.15,.20,'Dữ liệu công khai','Mạng đường / làn, POI\nthời gian, tham số\nB, K, h, Δ',blue,'#EDF3FA')
box(.02,.19,.15,.12,'Tín hiệu phiên','Kết thúc khi xảy ra\nkhông biết trước đích',blue,'#EDF3FA')

box(.225,.62,.16,.18,'Neo Geo-I / REM','Test có nhiễu: giữ / tạo\nneo riêng tư z(t)')
box(.225,.41,.16,.15,'Ngân sách + pacing','Cận phiên 0,23 /m\nđọc GPS cách ≥60 s')
box(.43,.62,.16,.18,'Belief từ neo','Belief vị trí / chuyển động\nchỉ từ neo đã bảo vệ')
box(.43,.41,.16,.15,'Road-aware','Làn có hướng, luật rẽ\nMiền tới được theo Δt')
box(.635,.62,.16,.18,'POI-aware selector','K=5; L=10; phủ POI\nGeoI-Slack: slack 0,03')
box(.635,.41,.16,.15,'Cache + top-5 cục bộ','Phản hồi cùng epoch 60 s\nGPS chỉ xếp hạng ở đây')
box(.835,.62,.15,.18,'Transcript công khai','K vị trí / track dummy\nvà thời điểm công bố',blue,'#EDF3FA')
box(.835,.41,.15,.15,'Kết quả riêng','Top-5 POI tại thiết bị\nkhông gửi ra ngoài',blue,'#EDF3FA')

# Main data path: no raw-GPS edge enters the postprocessing layers.
arrow((.175,.71),(.22,.71));arrow((.39,.71),(.425,.71))
arrow((.595,.71),(.63,.71));arrow((.80,.71),(.83,.71))
arrow((.80,.485),(.83,.485))
arrow((.305,.565),(.305,.615),blue)
arrow((.51,.565),(.51,.615))
route([(.595,.485),(.614,.485),(.614,.60),(.715,.60),(.715,.615)])
route([(.985,.64),(.994,.64),(.994,.585),(.715,.585),(.715,.565)],blue,True)
route([(.175,.67),(.196,.67),(.196,.58),(.62,.58),(.62,.485),(.63,.485)],blue,True)
# Context enters budget/support and road-aware postprocessing through a shared public bus.
route([(.175,.48),(.187,.48),(.187,.365),(.51,.365),(.51,.405)],blue,True)
arrow((.305,.365),(.305,.405),blue,True)
route([(.402,.365),(.402,.59),(.305,.59),(.305,.615)],blue,True)
# Boundary mechanisms belong to our model, while retaining their separate evidence status.
ax.text(.505,.335,'LAYER 4 · BOUNDARY S9/S10 (trước / sau lõi)',ha='center',fontsize=9.5,weight='bold',color=amber)
box(.225,.195,.16,.12,'S9 · bỏ đầu trước lõi','h giây đầu: không đưa\nGPS vào cơ chế neo',amber,'#FFF5E9',True)
box(.43,.195,.16,.12,'BR-Boundary v1','Đã kiểm tra tích hợp\nChưa benchmark kết hợp',amber,'#FFF5E9',True)
box(.635,.195,.16,.12,'S10 · buffer sau lõi','Buffer Δ; phát khi đủ tuổi\nkết thúc: hủy, không flush',amber,'#FFF5E9',True)
route([(.22,.255),(.212,.255),(.212,.71),(.22,.71)],amber,True)
route([(.795,.65),(.807,.65),(.807,.33),(.715,.33),(.715,.32)],amber,True)
route([(.80,.255),(.819,.255),(.819,.71),(.83,.71)],amber,True)
route([(.175,.25),(.187,.25),(.187,.095),(.715,.095),(.715,.19)],blue,True)
ax.text(.91,.32,'Server trả phản hồi\ncho cache ở thiết bị',ha='center',fontsize=9.2,color=blue,linespacing=1.5)
ax.text(.91,.19,'Neo, belief, ngân sách\nlà trạng thái nội bộ.',ha='center',fontsize=9.2,color=ink,linespacing=1.5)
ax.text(.505,.045,'Luồng liền: đầu ra lõi · Luồng nâu nét đứt: đầu ra qua boundary khi bật · Xanh nét đứt: input / phản hồi',
        ha='center',fontsize=9,color=ink)
for extension in ('pdf','svg','png'):
    fig.savefig(OUT/f'architecture_model.{extension}',bbox_inches='tight',dpi=180)
plt.close(fig)
svg=OUT/'architecture_model.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
