"""Standalone teaching slides and offline step explorer from inspected run data."""
from pathlib import Path
import json
import textwrap
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path(__file__).resolve().parent
D=json.loads((OUT/'walkthrough.json').read_text())
plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,
                     'font.size':12,'axes.titlesize':14,'axes.labelsize':12})
INK,BLUE,GOLD,PINK,OLIVE='#213844','#28669b','#ad771b','#a94c79','#647539'
RUN=D['runs']['boundary']
FIRST=next(r for r in RUN['rows'] if not r['head_skipped'])
SENT=next(r for r in RUN['rows'] if r['released_source_times'])
SLACK=next(r for r in RUN['rows'] if r.get('objective',{}).get('objective_loss',0)>1e-8)
REUSE=next((r for r in RUN['rows'] if r.get('branch')=='reuse'),None)
CAT=D['categories'].index('restaurant')
origin=np.array([D['graph']['bbox'][1],D['graph']['bbox'][0]])
mlat=6371000*np.pi/180;mlon=mlat*np.cos(np.radians(D['projection_lat0']))


def xy(coords):
    a=np.asarray(coords);return (a-origin)*np.array([mlat,mlon])


def paragraph(fig,x,y,text,width=62,size=13,color=INK):
    lines='\n'.join(textwrap.fill(p,width=width) for p in text.split('\n'))
    return fig.text(x,y,lines,ha='left',va='top',fontsize=size,color=color,linespacing=1.5)


def slide(title,subtitle):
    fig=plt.figure(figsize=(16,9),facecolor='white')
    fig.text(.045,.935,title,color=INK,size=25,weight='bold')
    paragraph(fig,.045,.89,subtitle,width=140,size=12,color='#526670')
    fig.text(.045,.035,'Mẫu SUMO có sẵn · mạng demo tái dựng, chưa xác minh trùng benchmark · chưa đo attacker',
             color='#526670',size=10)
    fig.text(.955,.035,'© OpenStreetMap contributors',ha='right',color='#526670',size=10)
    return fig


def map_view(ax,row,queries=None,show_belief=False,pois=None):
    roads=[xy([[lat,lon] for lon,lat in line])[:,::-1] for line in D['roads_lonlat']]
    ax.add_collection(LineCollection(roads,colors='#d5dddf',linewidths=.8,zorder=0))
    a=xy([[p['lat'],p['lon']] for p in D['true_trace']])
    ax.plot(a[:,1],a[:,0],ls='--',color='#82949c',lw=1.2,label='GPS cả chuyến (phân tích)')
    if show_belief:
        w=np.array(row['region_weights']);order=np.argsort(-w);keep=order[np.r_[0,np.cumsum(w[order])[:-1]]<.8]
        latents=xy(np.array(D['latent_latlon'])[keep])
        ax.scatter(latents[:,1],latents[:,0],s=16+240*w[keep],color=GOLD,alpha=.22,
                   label='Ô chứa 80% trọng số ước lượng',zorder=2)
    if row.get('anchor') is not None and show_belief:
        z=xy(row['anchor']);ax.scatter(z[1],z[0],s=100,marker='D',color=GOLD,
                                    edgecolor='white',label='Z: tham chiếu trong máy',zorder=5)
        ax.annotate('Z',(z[1],z[0]),xytext=(8,9),textcoords='offset points',color=GOLD,weight='bold')
    if queries:
        q=xy(queries);ax.scatter(q[:,1],q[:,0],s=90,facecolor='white',edgecolor=PINK,
                                marker='o',linewidth=2,label='5 điểm truy vấn',zorder=4)
        for j,(y,x) in enumerate(q):ax.annotate(f'Q{j+1}',(x,y),xytext=(8,(-1)**j*9),
                                             textcoords='offset points',color=PINK,size=11)
    if pois:
        p=xy([[D['pois'][v]['lat'],D['pois'][v]['lon']] for v in pois])
        ax.scatter(p[:,1],p[:,0],marker='s',color=OLIVE,s=60,label='5 nhà hàng trả cho người dùng',zorder=6)
        for j,(y,x) in enumerate(p):ax.annotate(f'P{j+1}',(x,y),xytext=(-25,10+9*(j%2)),
                                             textcoords='offset points',color=OLIVE,size=10)
    p=xy(row['gps']);ax.scatter(p[1],p[0],s=125,marker='x',color=BLUE,
                              linewidth=3,label=f'GPS thật tại {row["t"]:g}s',zorder=7)
    ax.annotate('GPS',(p[1],p[0]),xytext=(9,9),textcoords='offset points',color=BLUE,weight='bold')
    ax.set_xlim(0,(D['graph']['bbox'][2]-D['graph']['bbox'][0])*mlon)
    ax.set_ylim(0,(D['graph']['bbox'][3]-D['graph']['bbox'][1])*mlat)
    ax.set_aspect('equal');ax.set_xlabel('Đông, từ biên vùng demo (m)');ax.set_ylabel('Bắc, từ biên vùng demo (m)')
    ax.spines[['top','right']].set_visible(False)
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.10),fontsize=10,frameon=False,ncol=2)


def table(ax,headers,rows,widths=None,size=11):
    ax.axis('off');t=ax.table(cellText=rows,colLabels=headers,loc='upper left',
                           cellLoc='left',colLoc='left',colWidths=widths)
    t.auto_set_font_size(False);t.set_fontsize(size);t.scale(1,1.9)
    for (r,c),cell in t.get_celld().items():
        cell.set_edgecolor('#d9e1e4');cell.set_linewidth(.6)
        if r==0:cell.set_facecolor('#f2f5f6');cell.set_text_props(weight='bold',color=INK)
        else:cell.set_text_props(color=INK)
    return t


def main():
    fig1=slide('1 · Chuyến đầu tiên trong dataset: u701_00',
        'GPS nguyên bản, 384 giây; K = 5, L = 10; bật che đầu 60s và gửi chậm 60s. Lần chạy minh họa mới, seed cố định trước kết quả.')
    ax=fig1.add_axes([.045,.19,.43,.61]);map_view(ax,FIRST)
    paragraph(fig1,.54,.79,'Đầu vào\n• GPS từ chuyến mô phỏng SUMO có sẵn.\n• Bản đồ công khai tái dựng + 61 POI OSM.\n• Trạng thái POI mô phỏng, p = 0,8.',width=60)
    paragraph(fig1,.54,.57,'Luồng để trình bày\nS9 bỏ đầu → lịch/ngân sách → Geo-I → ước lượng + đường đi → 5 điểm truy vấn → S10 giữ tạm → máy chủ → hợp POI và top-5.',width=60)
    paragraph(fig1,.54,.37,f'Điểm cần nhìn\n0/20/40s: GPS chưa vào mô hình.\n60s: tạo bộ đầu, giữ trong thiết bị.\n120s: gửi bộ đã tạo ở 60s.\n384s: đóng phiên, hủy {RUN["accounting"]["tail_cancelled"]} bộ đang chờ.',width=60)
    paragraph(fig1,.54,.16,'Mạng demo khác snapshot benchmark; nối đường và luật rẽ là giả định tái dựng. Mẫu này giải thích cách hoạt động, không chứng minh vượt baseline.',width=65,size=11,color='#526670')

    fig2=slide('2 · Bảo vệ GPS, ước lượng vùng và chọn điểm',
        'Tại 60s: lần đầu GPS đi vào mô hình sau S9. Vị trí tham chiếu và trọng số ước lượng chỉ dùng trong thiết bị.')
    ax=fig2.add_axes([.045,.21,.43,.59]);map_view(ax,FIRST,FIRST['queries'],True)
    paragraph(fig2,.54,.79,f'GPS thật: {FIRST["gps"][0]:.6f}, {FIRST["gps"][1]:.6f}\nZ đã bảo vệ: {FIRST["anchor"][0]:.6f}, {FIRST["anchor"][1]:.6f}\nKhoảng cách GPS–Z: {FIRST["anchor_error_m"]:.1f} m.\nTạo Z lần đầu: +0,01/m.',width=66)
    paragraph(fig2,.54,.60,f'Ước lượng trên {len(D["latent_latlon"])} ô công khai. Trọng số cập nhật từ Z; chấm vàng chứa 80% tổng trọng số, không phải vùng chính xác đã kiểm định.\nChọn 5 điểm Q để mang về các POI bổ sung nhau. Các mốc sau bị giới hạn bởi đường xe có thể tới trong thời gian đã trôi qua.',width=66)
    fresh=next(r for r in RUN['rows'] if r.get('distance_m') is not None and r['branch']=='fresh')
    paragraph(fig2,.54,.37,f'Phép kiểm tra có nhiễu, θ = 200 m\n{fresh["t"]:g}s: {fresh["distance_m"]:.1f} + ({fresh["test_noise_m"]:.1f}) = {fresh["noisy_distance_m"]:.1f} > 200 → tạo mới; +0,02/m.\n{REUSE["t"]:g}s: {REUSE["distance_m"]:.1f} + ({REUSE["test_noise_m"]:.1f}) = {REUSE["noisy_distance_m"]:.1f} ≤ 200 → giữ Z; +0,01/m.',width=66,size=12)
    o=SLACK['objective']
    paragraph(fig2,.54,.19,f'Slack thực tế ở {SLACK["t"]:g}s: điểm độ phủ {o["objective_before_slack"]:.6f} → {o["objective_after_slack"]:.6f}; giảm {o["objective_loss"]:.6f} ≤ 0,03.\nĐây là điểm lập kế hoạch; θ không giới hạn bán kính nhiễu, slack không đặt cận giảm Recall.',width=67,size=11)

    fig3=slide('3 · Máy chủ trả POI; thiết bị chọn top-5',
        'Tại 120s: máy chủ nhận 5 điểm đã tạo ở 60s. Phản hồi dùng trạng thái POI tại 120s; xếp hạng dùng GPS thật hiện tại trong thiết bị.')
    publication=next(p for p in RUN['publications'] if p['publication_t']==SENT['t'])
    ax=fig3.add_axes([.045,.21,.43,.59]);map_view(ax,SENT,publication['coordinates'],pois=SENT['service']['returned'][CAT])
    per_point=[[f'Q{i+1}',str(len(reply[CAT]))] for i,reply in enumerate(SENT['service']['replies'])]
    table(fig3.add_axes([.54,.54,.17,.25]),['Điểm','Nhà hàng'],per_point,[.55,.45],11)
    s=SENT['service'];union=len(set(v for reply in s['replies'] for v in reply[CAT]))
    paragraph(fig3,.74,.78,f'L = 10 / loại / điểm.\nBỏ trùng: {union} nhà hàng.\nTất cả loại: {s["fresh_union_count"]} POI.\nThiết bị chọn 5 mỗi loại từ phản hồi còn hiệu lực.',width=31,size=12)
    rows=[]
    for j,v in enumerate(s['returned'][CAT]):
        rows.append([f'P{j+1}',D['pois'][v]['id'].replace('osm/node/',''),f'{s["returned_distances_m"][CAT][j]:.1f} m','Có' if v in s['reference'][CAT] else 'Không'])
    table(fig3.add_axes([.54,.25,.415,.24]),['POI','ID OSM','Đường tới','Top-5 chuẩn?'],rows,[.11,.39,.22,.28],10)
    paragraph(fig3,.54,.18,f'Tại mốc này: tìm đúng {len(set(s["returned"][CAT]) & set(s["reference"][CAT]))}/{len(s["reference"][CAT])} nhà hàng trong top-5 chuẩn.\nRecall trung bình trên {len(D["categories"])-s["empty_reference_categories"]}/{len(D["categories"])} loại có chuẩn: {100*s["recall"]:.1f}%. Loại không có POI chuẩn bỏ khỏi mẫu số.',width=65,size=11)

    fig4=slide('4 · Ngân sách và đánh đổi khi che đầu/cuối',
        'Cùng chuyến, cùng tham số và seed. Một mẫu minh họa trên mạng nhỏ; số đo dịch vụ dưới đây không phải bảng benchmark.')
    ax=fig4.add_axes([.07,.52,.40,.28])
    for name,color,style in [('plain',BLUE,'--'),('boundary',PINK,'-')]:
        rs=D['runs'][name]['rows'];ax.step([r['t'] for r in rs],[r['spent'] for r in rs],where='post',
                                       color=color,ls=style,label='Tắt S9/S10' if name=='plain' else 'Bật S9/S10')
    ax.axhline(.23,color=INK,lw=1,ls=':');ax.text(375,.234,'Cận hiệu lực 0,23/m',size=10,ha='right')
    ax.set_ylim(0,.27);ax.set_xlim(0,384);ax.set_xticks([0,60,120,180,240,300,384]);ax.set_xlabel('Thời gian (s)');ax.set_ylabel('Ngân sách đã dùng /m')
    ax.legend(frameon=False,fontsize=10,loc='upper center',bbox_to_anchor=(.5,-.22),ncol=2);ax.spines[['top','right']].set_visible(False)
    compare=[]
    for key,label in [('plain','Tắt'),('boundary','Bật')]:
        r=D['runs'][key];a=r['accounting'];compare.append([label,str(r['private_reads']),str(a['released_events']),str(a['tail_cancelled']),f'{r["budget_spent"]:.2f}',f'{100*r["recall_all_input_times"]:.2f}%'])
    table(fig4.add_axes([.54,.58,.415,.21]),['S9/S10','Đọc GPS','Bộ gửi','Bộ hủy','Chi phí /m','Recall¹'],compare,[.15,.16,.15,.15,.17,.22],10)
    paragraph(fig4,.54,.49,'¹ Trung bình đều 21 mốc đầu vào, trên các loại có POI chuẩn. Không có phản hồi được chấm 0; tính cả giai đoạn chờ.',width=65,size=11)
    paragraph(fig4,.54,.38,f'Bật S9/S10: {RUN["accounting"]["head_suppressed"]} mốc đầu không tạo truy vấn; {RUN["accounting"]["protected_events"]} bộ được tạo, {RUN["accounting"]["released_events"]} gửi và {RUN["accounting"]["tail_cancelled"]} hủy.\nCác bộ tạo ở 340/360/380/384s chưa tới hạn gửi khi đóng phiên. Bộ cuối đã gửi: tạo ở 320s, gửi ở 380s.',width=65,size=12)
    paragraph(fig4,.07,.36,f'Ngân sách bản bật\n{RUN["private_reads"]-1} kiểm tra × 0,01 + {RUN["private_reads"]-RUN["reuse_count"]} lần tạo mới × 0,01 = {RUN["budget_spent"]:.2f}/m.\nCòn {(.23-RUN["budget_spent"]):.2f}/m; mọi lần đọc sau cần dành đủ 0,02/m.\nK = 5 và slack không nhân chi phí GPS.\nHủy truy vấn không hoàn lại ngân sách đã tiêu.',width=53,size=12)
    paragraph(fig4,.54,.18,f'Đánh đổi của mẫu này: lần gửi đầu ở 120s. Recall khi có gửi = {100*RUN["recall_delivery_times"]:.1f}%, nhưng tính cả 21 mốc là {100*RUN["recall_all_input_times"]:.2f}%.\nChe đoạn biên không bảo đảm attacker không suy được điểm đầu/cuối. Mẫu này chưa đo attacker.',width=66,size=11)
    with PdfPages(OUT/'walkthrough.pdf') as pdf:
        for i,fig in enumerate([fig1,fig2,fig3,fig4],1):
            fig.savefig(OUT/f'slide_{i}.png',dpi=120)
            pdf.savefig(fig);plt.close(fig)
    template=(OUT/'template.html').read_text()
    (OUT/'walkthrough.html').write_text(template.replace('__DATA__',json.dumps(D,ensure_ascii=False,separators=(',',':'))))
    print('Rendered 4 slides and offline step explorer.')


if __name__=='__main__':main()
