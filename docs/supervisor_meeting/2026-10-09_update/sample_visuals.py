"""Source-backed sample maps and walkthrough slide bodies.

These SVG maps preserve geometry and source points; they are evidence diagrams.
No sampler or benchmark is run by the presentation renderer.
"""
from collections import defaultdict
import math

GPS_COLOR = '#a24a55'
ROAD_COLOR = '#dcdde1'


def map_svg(uid, area, roads, important, marks, *, text, line, dot, belief=()):
    x,y,w,h=area
    # Uniform metric scale; viewport selection changes only the display.
    points=list(important)
    if not points:raise ValueError('Map needs source-backed coordinates')
    xmin=min(p[0] for p in points);xmax=max(p[0] for p in points)
    ymin=min(p[1] for p in points);ymax=max(p[1] for p in points)
    dx=max(xmax-xmin,200.);dy=max(ymax-ymin,200.)
    xmin-=.15*dx;xmax+=.15*dx;ymin-=.15*dy;ymax+=.15*dy
    scale=min(w/(xmax-xmin),h/(ymax-ymin))
    cx=(xmin+xmax)/2;cy=(ymin+ymax)/2
    def screen(p):return x+w/2+(p[0]-cx)*scale,y+h/2-(p[1]-cy)*scale
    b=f'<defs><clipPath id="map-{uid}"><rect x="{x}" y="{y}" width="{w}" height="{h}"/></clipPath></defs>'
    b+=f'<g clip-path="url(#map-{uid})">'
    b+=f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="#fbfcfd"/>'
    for road in roads:
        if not road:continue
        if max(p[0] for p in road)<xmin or min(p[0] for p in road)>xmax or max(p[1] for p in road)<ymin or min(p[1] for p in road)>ymax:continue
        pts=' '.join(f'{a:.2f},{c:.2f}' for a,c in map(screen,road))
        b+=f'<polyline points="{pts}" fill="none" stroke="{ROAD_COLOR}" stroke-width="1.3"/>'
    maxweight=max((r['weight'] for r in belief),default=1.)
    for cell in belief:
        a,c=screen(cell['xy']);radius=4+12*math.sqrt(cell['weight']/maxweight)
        b+=f'<circle cx="{a:.2f}" cy="{c:.2f}" r="{radius:.2f}" fill="#b86b28" opacity=".18"/>'
    label_rows=[]
    for mark in marks:
        a,c=screen(mark['xy']);color=mark['color'];shape=mark.get('shape','circle')
        if shape=='cross':b+=line(a-9,c-9,a+9,c+9,color,4)+line(a-9,c+9,a+9,c-9,color,4)
        elif shape=='diamond':b+=f'<polygon points="{a},{c-11} {a+11},{c} {a},{c+11} {a-11},{c}" fill="{color}" stroke="white" stroke-width="2"/>'
        elif shape=='square':b+=f'<rect x="{a-6}" y="{c-6}" width="12" height="12" fill="{color}" stroke="white" stroke-width="1"/>'
        else:
            radius=mark.get('radius',8)
            b+=dot(a,c,radius,color)+f'<circle cx="{a}" cy="{c}" r="{radius}" fill="none" stroke="white" stroke-width="1.5"/>'
        if mark.get('label'):
            side=1 if a<x+w*.6 else -1
            label_rows.append((a+mark.get('label_dx',side*16),c+mark.get('label_dy',-13),mark['label'],color,mark.get('anchor','start' if side==1 else 'end')))
    b+='</g>'
    for a,c,label,col,anchor in label_rows:b+=text(a,c,label,23,color=col,weight=700,anchor=anchor)
    # Fixed physical distance bar, never a privacy or error-radius claim.
    distance=1000 if max(dx,dy)>3500 else (500 if max(dx,dy)>1500 else 200)
    length=distance*scale
    a=x+20;c=y+h-20
    b+=f'<rect x="{a-7}" y="{c-29}" width="{length+30}" height="43" fill="white" opacity=".9"/>'
    b+=line(a,c,a+length,c,'#71838d',3)+line(a,c-5,a,c+5,'#71838d',2)+line(a+length,c-5,a+length,c+5,'#71838d',2)
    b+=text(a+length/2,c-8,f'{distance} m',18,color='#71838d',anchor='middle')
    return b


def endpoint_slide(data, *, text, line, dot, arrow, box, ink, teal, blue, gray, orange):
    frames=data['frames']
    lat,lon=frames[0]['x_latlon_evaluator_only'];x0,y0=frames[0]['x_xy_evaluator_only']
    xp=x0/lon;yp=y0/lat
    def xy(ll):return [ll[1]*xp,ll[0]*yp]
    roads=[[xy(p) for p in road] for road in data['public_road_geometry']['polylines_latlon']]
    b=text(56,142,'Cùng một hành trình: gửi truy vấn ngay tại điểm đầu và điểm cuối',28,weight=700)
    b+=dot(64,182,7,gray)+text(82,189,'Q · GeoI-Slack L20',23,color=gray)
    b+=dot(412,182,7,teal)+text(430,189,'Q · Endpoint20',23,color=teal)
    b+=line(822,175,836,189,GPS_COLOR,3)+line(822,189,836,175,GPS_COLOR,3)+text(850,189,'GPS phân tích (không gửi)',23,color=GPS_COLOR)
    for i,frame in enumerate(frames):
        px=56+i*592
        b+=text(px,242,f"{frame['scenario']} · {frame['label_vi']} · t = {frame['actual_endpoint_label_time_s']:g} s",29,weight=700)
        target=xy(frame['x_latlon_evaluator_only']);points=[target];marks=[]
        for key,col in [('scale100_L20',gray),('scale025_L20',teal)]:
            coords=[xy(ll) for ll in frame['methods'][key]['Q_latlon_public']]
            points+=coords;marks += [{'xy':p,'color':col,'shape':'circle','radius':8} for p in coords]
        marks.append({'xy':target,'color':GPS_COLOR,'shape':'cross','label':'GPS'})
        b+=map_svg('endpoint-'+str(i),(px,266,560,260),roads,points,marks,text=text,line=line,dot=dot)
        b+=text(px,558,f"Tạo Q lúc {frame['actual_endpoint_label_time_s']:g} s → gửi lúc {frame['actual_endpoint_label_time_s']:g} s",27,color=teal,weight=700)
    b+=text(56,609,'Tại 384 s: dùng lịch sử của lần đọc 360 s, không đọc GPS bảo vệ mới.',26)
    b+=text(56,644,'Endpoint20: ngân sách bằng 25% đối chứng; thời điểm bắt đầu/kết thúc vẫn quan sát được.',24,color=orange)
    return b


def protection_slide(data, *, text, line, dot, arrow, box, ink, teal, blue, gray, orange):
    events=data['main_sample']['events'];event=events[-1]
    b=text(56,140,'Hành trình minh họa: khởi tạo, giữ lịch sử và cập nhật điểm tham chiếu',29,weight=700)
    b+=text(56,183,'Bản đồ ở t = 60 s',26,weight=700)
    b+=text(680,183,'Ngân sách của cùng một chuyến',26,weight=700)
    marks=[{'xy':p,'color':teal,'label':f'Q{i+1}','shape':'circle','radius':9} for i,p in enumerate(event['Q_xy'])]
    marks += [{'xy':event['Z_xy'],'color':blue,'label':'Z','shape':'diamond'}, {'xy':event['gps_xy'],'color':GPS_COLOR,'label':'GPS','shape':'cross'}]
    belief=[{'xy':r['xy_m'],'weight':r['weight']} for r in event['belief']['top_weights']]
    points=[event['gps_xy'],event['Z_xy'],*event['Q_xy'],*event['belief_top_xy']]
    b+=map_svg('protection',(56,216,560,329),data['map_main']['road_segments_xy'],points,marks,text=text,line=line,dot=dot,belief=belief)
    b+=text(56,579,'12 vùng trọng số cao nhất: 9,88% khối lượng b.',23,color=orange)
    b+=text(56,616,'GPS và Z tại thiết bị; chỉ năm Q gửi máy chủ.',25,color=teal,weight=700)
    xs=[680,787,943,1132]
    for j,(x,label) in enumerate(zip(xs,['t (s)','Đọc GPS?','Z','Chi phí / tích lũy'])):b+=text(x,237,label,23,weight=700,anchor='start' if j==0 else 'middle')
    b+=line(680,254,1224,254,ink)
    for i,row in enumerate(events):
        y=294+i*60;prot=row['protection']
        label='Tạo mới' if prot['GPS_read'] else 'Giữ nguyên'
        vals=[str(int(row['t_s'])),'Có' if prot['GPS_read'] else 'Không',label,f"{prot['cost_units']} / {prot['spent_after_units']}"]
        for j,(x,v) in enumerate(zip(xs,vals)):b+=text(x,y,v,25,anchor='start' if j==0 else 'middle',color=blue if prot['GPS_read'] else gray)
        b+=line(680,y+18,1224,y+18)
    u=data['configuration']['epoch_budget']['unit_epsilon_per_m']
    b+=text(680,464,f'u = {str(u).replace(".",",")} m⁻¹; U = 23 đơn vị mỗi phiên.',23)
    reuse=data['test_reuse_inset']['events'][-1]['protection']
    b+=text(680,509,'Chuyến 6, t = 60 s: đọc GPS nhưng giữ Z',25,weight=700,color=teal)
    b+=text(680,547,f"Khoảng cách tới Z cũ: {reuse['distance_GPS_to_previous_Z_m']:.0f} m.",24,color=teal)
    b+=text(680,581,'Kiểm tra có nhiễu ≤ 200 m → giữ Z.',24,color=teal)
    b+=text(680,615,f"Chi +{reuse['cost_units']} đơn vị; đã dùng {reuse['spent_after_units']} đơn vị.",24,color=teal)
    return b


def utility_slide(data, protection, *, text, line, dot, arrow, box, ink, teal, blue, gray, orange):
    category=data['illustration_category'];category_results={key:row[category] for key,row in data['local_results'].items()}
    projection=protection['metadata']['projection']
    def xy(p):return [p['lon']*projection['m_per_deg_lon'],p['lat']*projection['m_per_deg_lat']]
    gps=xy(data['inputs']['GPS']);qs=[xy(q['payload']) for q in data['requests']]
    nearest=category_results['nearest_distance']['answer'];pois=[xy(p) for p in nearest]
    b=text(56,140,'Cùng năm Q tại t = 60 s; thay mục đích chỉ đổi xếp hạng tại thiết bị',29,weight=700)
    raw=data['merged']['raw_record_count'];unique=data['merged']['unique_count']
    b+=text(56,189,f'5 Q × 84 bản ghi → {raw} bản ghi → {unique} POI không trùng',29,color=teal,weight=700)
    # Zoom the actual returned-POI neighborhood. The five source Q are shown
    # on the preceding sample slide; adding distant Q would hide this cluster.
    marks=[{'xy':pos,'color':blue,'shape':'square','label':poi['display_alias']} for pos,poi in zip(pois,nearest)]
    for mark in marks:
        if mark['label']=='P13':mark.update(label_dx=-18,label_dy=0,anchor='end')
        if mark['label']=='P224':mark.update(label_dx=18,label_dy=-13,anchor='start')
    marks.append({'xy':gps,'color':GPS_COLOR,'shape':'cross','label':'GPS'})
    b+=map_svg('utility',(56,226,540,328),protection['map_main']['road_segments_xy'],[gps,*pois],marks,text=text,line=line,dot=dot)
    b+=text(56,587,'Vị trí GPS và năm POI café gần nhất',23,color=blue)
    b+=text(56,621,'P là POI; Q là tọa độ truy vấn.',25,weight=700)
    b+=text(652,249,'Gần nhất · khoảng cách đường',26,weight=700)
    for i,poi in enumerate(nearest):
        y=289+i*38
        b+=text(652,y,poi['display_alias'],25,color=blue)+text(880,y,f"{poi['score']:.0f} m".replace('.',','),25,color=blue)
    radius=category_results['within_radius']['answer']
    b+=text(652,507,'Trong bán kính 1.000 m',26,weight=700,color=teal)
    b+=text(652,545,f"Chỉ {radius[0]['display_alias']}: {radius[0]['score']:.0f} m; trả {len(radius)} POI." if radius else 'Không có POI phù hợp.',25,color=teal)
    detour=category_results['minimum_detour']['answer']
    b+=text(652,594,'Ít vòng tới đích · danh sách khác',26,weight=700)
    b+=text(652,632,' → '.join(p['display_alias'] for p in detour),25)
    return b
