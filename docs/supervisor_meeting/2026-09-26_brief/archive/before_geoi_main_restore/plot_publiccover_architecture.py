"""PublicCover model layers, with separate public and local outputs."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
fig, ax = plt.subplots(figsize=(14, 4.8))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
ink, teal, blue = '#183A45', '#087F7A', '#326798'


def box(x, y, w, h, title, body, external=False):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.004',
        facecolor='#EDF3FA' if external else '#EAF4F2', edgecolor=blue if external else teal, linewidth=1.2))
    ax.text(x+w/2,y+h*.82,title,ha='center',va='center',weight='bold',fontsize=10,color=ink)
    ax.text(x+w/2,y+h*.30,body,ha='center',va='center',fontsize=8.8,color=ink,linespacing=1.2)


def arrow(a,b,color=ink,dashed=False):
    ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'-|>','color':color,'lw':1.4,'linestyle':'--' if dashed else '-'} )


ax.add_patch(FancyBboxPatch((.205,.105),.60,.80,boxstyle='round,pad=.008',
    facecolor='#F7FAF9',edgecolor=teal,linewidth=2,zorder=0))
ax.text(.505,.96,'PUBLICCOVER: COVERLITE-30 / COVERPLUS-67',ha='center',fontsize=12,weight='bold',color=teal)
for x,title in [(.095,'INPUT'),(.305,'LAYER 1 · KẾ HOẠCH'),(.51,'LAYER 2 · LỊCH / CACHE'),(.715,'LAYER 3 · CỤC BỘ'),(.91,'OUTPUT')]:
    ax.text(x,.86,title,ha='center',fontsize=10,weight='bold',color=blue if x in (.095,.91) else teal)
box(.02,.62,.15,.18,'Dữ liệu công khai','Mạng đường, 6 loại POI\nL=10, vùng / lịch cố định',True)
box(.225,.62,.16,.18,'Greedy cover','30 cặp / 19 tọa độ\n67 cặp / 52 tọa độ')
box(.43,.62,.16,.18,'Gửi kế hoạch cố định','Tick mỗi 60 giây\ncả khi không có chuyến\n60 lần / giờ')
ax.text(.715,.575,'Không dùng neo Geo-I',ha='center',va='center',fontsize=10,color=teal,linespacing=1.5)
box(.835,.62,.15,.18,'Đầu ra công khai','Loại POI, tọa độ, thời gian\n30 / 67 truy vấn logic',True)
box(.02,.35,.15,.18,'Phản hồi máy chủ','POI hiện khả dụng\nTối đa 10 POI / cặp',True)
box(.43,.35,.16,.18,'Cache tại thiết bị','Hợp POI vừa nhận\nđọc không phát truy vấn')
box(.635,.35,.16,.18,'Xếp hạng cục bộ','GPS + khoảng cách đường\nchọn 5 POI theo loại cần')
box(.835,.35,.15,.18,'Kết quả cục bộ','Top-5 POI cho người dùng\nGPS / lượt đọc ở tại máy',True)
box(.02,.13,.15,.15,'Dữ liệu riêng tư','GPS thật, nhu cầu POI\nchỉ dùng tại thiết bị',True)
arrow((.175,.71),(.22,.71));arrow((.39,.71),(.425,.71));arrow((.595,.71),(.83,.71))
arrow((.175,.44),(.425,.44),blue);arrow((.595,.44),(.63,.44));arrow((.80,.44),(.83,.44))
ax.plot([.175,.715,.715],[.195,.195,.34],color=blue,lw=1.2,ls='--')
arrow((.715,.27),(.715,.345),blue,True)
ax.text(.505,.055,'Luồng xanh nét đứt: GPS / nhu cầu chỉ đi vào xếp hạng cục bộ · Không có đường quay lại kế hoạch hay lịch',
    ha='center',fontsize=9.2,color=ink)
for extension in ('pdf','svg','png'):
    fig.savefig(OUT/f'architecture_publiccover.{extension}',bbox_inches='tight',dpi=180)
plt.close(fig)
svg=OUT/'architecture_publiccover.svg'
svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
