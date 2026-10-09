"""Generate source-pinned tables and diagrams for the compact LaTeX report.

No sampler, attacker or evaluator is run. Compile report_explained.tex using
Tectonic, then render and review the resulting PDF before release.
"""
from pathlib import Path
import hashlib
import math
import json
import re

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def number(value, digits=2):
    return f'{value:.{digits}f}'.replace('.', ',')


def architecture(current):
    """Native TikZ flow with a framed device and an external POI server."""
    gate = r'\textbf{1. Kiểm tra lịch và ngân sách}'
    gate += (r'\\Cap $C_s=0{,}23\,\mathrm m^{-1}$ mỗi phiên\\Dự toán trước GPS; $u=0{,}01\,\mathrm m^{-1}$; khoảng đọc $\geq60$ s'
             if current else r'\\Cap $0{,}23/\mathrm m$ mỗi phiên; khoảng đọc $\geq60$ s\\Warmup 60 s trước GPS khi bật bảo vệ đầu chuyến')
    send = (r'\textbf{5. Yêu cầu chung, gửi ngay}\\$K=5$ Q; mọi loại POI; $L=30$ mỗi loại mỗi Q'
            if current else r'\textbf{5. Yêu cầu chung; delay tùy chọn}\\$K=5$ Q; mọi loại POI; $L=10$ mỗi loại mỗi Q\\Khi bật: giữ Q 60 s rồi mới công bố')
    local = (r'\textbf{6. Gộp, bỏ trùng, lọc và sắp xếp local}\\GPS + nhu cầu $\psi$: gần nhất, nhanh nhất,\\trong bán kính hoặc ít đi vòng tới đích'
             if current else r'\textbf{6. Hợp phản hồi và xếp hạng local}\\Giữ phản hồi còn hiệu lực theo epoch dịch vụ\\Xếp hạng gần nhất theo GPS; giới hạn $k=5$')
    col = 'teal' if current else 'ink'
    out = r'Danh sách POI phù hợp\\đã lọc và sắp xếp' if current else r'Tối đa 5 POI mỗi loại\\theo khoảng cách gần nhất'
    body = r'''\begin{tikzpicture}[x=1cm,y=1cm,>=Latex,
      stage/.style={draw=ink!55,fill=white,rounded corners=2pt,text width=8.45cm,
                    align=center,inner sep=7pt,font=\small,minimum height=1.28cm},
      input/.style={draw=blue!65,fill=blue!3,rounded corners=2pt,align=center,
                    inner sep=6pt,font=\small},
      external/.style={draw=blue!65,fill=blue!3,rounded corners=2pt,text width=2.55cm,
                    align=center,inner sep=6pt,font=\small},
      flow/.style={->,line width=.85pt,draw=ink},
      network/.style={->,line width=.85pt,draw=blue},
      public/.style={->,dashed,line width=.65pt,draw=gray},
      layer/.style={text width=2.35cm,align=left,font=\small\bfseries,text=teal}]
    \fill[teal!2,rounded corners=4pt] (.5,-1.1) rectangle (12.9,-16.25);
    \draw[teal,line width=1pt,rounded corners=4pt] (.5,-1.1) rectangle (12.9,-16.25);
    \node[anchor=west,font=\small\bfseries,text=teal] at (.85,-1.45) {MÔ HÌNH TẠI THIẾT BỊ};
    \node[input,text width=3.3cm] (pub) at (1.95,.2) {\textbf{Dữ liệu công khai}\\Bản đồ, POI, lịch gửi};
    \node[input,text width=7.6cm] (gps) at (8,.2) {\textbf{GPS cho cơ chế Geo-I}\\Chỉ đọc sau khi lịch và ngân sách cho phép};
    \node[layer] at (1.95,-2.7) {Tầng 1\\Bảo vệ GPS};
    \node[stage,draw=ink,line width=.9pt] (guard) at (8,-2.8) {GATE};
    \node[stage] (rem) at (8,-5.05) {\textbf{2. Geo-I / REM + kiểm tra tái sử dụng}\\Đọc GPS $x$; thử giữ $Z$ bằng phép thử có nhiễu.\\Giữ $Z$ hoặc lấy mẫu REM mới trên miền đường cố định.};
    \draw[flow] (gps.south) -- (guard.north);
    \draw[flow] (guard.south) -- node[right,font=\footnotesize] {được phép đọc} (rem.north);
    \node[layer] at (1.95,-7.2) {Tầng 2\\Ước lượng và\\chọn truy vấn};
    \node[stage] (belief) at (8,-7.4) {\textbf{3. Ước lượng phân bố $b$ trên mạng đường}\\Đọc được: cập nhật từ quan sát đã bảo vệ.\\Không đọc mới: dự đoán từ lịch sử và chuyển tiếp công khai.};
    \node[stage] (planner) at (8,-9.45) {\textbf{4. Chọn 5 tọa độ truy vấn $Q$}\\Phủ POI; giới hạn chuyển động theo mạng đường.\\Bảng planner $L_{\rm plan}=10$; slack $0{,}03$ trên điểm độ phủ.};
    \draw[flow] (rem.south) -- node[right,font=\footnotesize] {$Z$ nội bộ} (belief.north);
    \draw[flow] (belief.south) -- (planner.north);
    \draw[flow] (guard.east) -- (12.65,-2.8) -- (12.65,-7.4) -- (belief.east);
    \node[rotate=90,font=\footnotesize,text=gray,fill=white,inner sep=2pt] at (12.65,-5) {không đọc GPS mới};
    \draw[public] (pub.south) -- (1.95,-.7) -- (.15,-.7) -- (.15,-9.45) -- (planner.west);
    \draw[public] (.15,-7.4) -- (belief.west);
    \node[layer] at (1.95,-11.65) {Tầng 3\\Truy hồi POI};
    \node[stage,draw=COL,line width=.9pt] (send) at (8,-11.7) {SEND};
    \draw[flow] (planner.south) -- (send.north);
    \node[external,minimum height=1.9cm] (server) at (14.85,-11.7) {\textbf{MÁY CHỦ POI}\\Ngoài mô hình\\Trả top-$L$ mỗi loại\\tại từng $Q$};
    \draw[network] (send.east) -- node[above,font=\footnotesize] {$Q$} (server.west);
    \node[font=\small,text=gray,align=center,text width=8cm] at (8,-13.5) {$Z$ và GPS thật giữ tại thiết bị.\\Máy chủ chỉ nhận tọa độ $Q$ và yêu cầu chung.};
    \node[layer] at (1.95,-15.05) {Tầng 4\\Xử lý local};
    \node[stage,draw=COL,line width=.9pt] (local) at (8,-15.05) {LOCAL};
    \draw[network] (server.east) -- (16.75,-11.7) -- (16.75,-15.05) -- node[above,font=\footnotesize] {phản hồi POI} (local.east);
    CLOSE
    \node[input,text width=4.6cm] (private) at (3.3,-17.3) {\textbf{Đầu vào riêng cho local}\\PRIVATE};
    \node[input,text width=4.6cm,draw=teal,fill=teal!4] (answer) at (10.15,-17.3) {\textbf{Đầu ra riêng cho người dùng}\\OUTPUT};
    \draw[network] (private.north) -- (3.3,-16.45) -- (6,-16.45) -- (6,-15.82);
    \draw[flow,draw=teal] (10.15,-15.82) -- (answer.north);
    \end{tikzpicture}
    '''
    close = '' if current else r'''\node[external,draw=orange,fill=orange!4,font=\footnotesize] (close) at (14.85,-13.75) {\textbf{Khi đóng phiên}\\Hủy Q còn chờ\\nếu đang bật delay};
    \draw[->,dashed,draw=orange] (send.south east) -- (12.95,-12.55) -- (12.95,-13.75) -- (close.west);'''
    for key, value in [('COL',col),('GATE',gate),('SEND',send),('LOCAL',local),('CLOSE',close),
                       ('PRIVATE',r'GPS local + nhu cầu $\psi$' if current else 'GPS local'),('OUTPUT',out)]:
        body=body.replace(key,value)
    return '\n'.join(line.rstrip() for line in body.strip().splitlines())+'\n'


def main():
    figures=HERE/'report_figures';figures.mkdir(exist_ok=True)
    tables=json.loads((HERE/'benchmark_tables.json').read_text())
    sample=json.loads((HERE/'multistep_sample.json').read_text())
    pins=json.loads((HERE/'source_pins.json').read_text())
    for name,expected in pins.items():
        actual=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        if actual!=expected:raise ValueError('Source changed: '+name)
    for version in ['old','current']:
        (figures/f'architecture_{version}.tex').write_text(architecture(version=='current'))
    rows=tables['current_fresh_four_purpose_utility']['rows']
    utility='\n'.join(' & '.join([r['label_vi'],number(100*r['L20_recall']),number(100*r['L30_recall']),
                         '+'+number(r['gain_pp'])])+r'\\' for r in rows)
    historical_path=HERE.parent/'2026-09-26_brief/method_evidence.json'
    historical=json.loads(historical_path.read_text())
    historical_index={r['scenario']:r for r in historical['scenario_rows']}
    coordinates=[]
    for scenario in ['S1','S2','S3']:
        r=historical_index[scenario]
        coordinates.append([scenario,number(100*r['raw_hit100']),number(100*r['hit100']),
                            number(r['mae_m'],0),number(100*r['recall'])])
    endpoint={(r['scenario'],r['method']):r for r in tables['S9_S10_full_bank_endpoint']['rows']}
    endpoints=[]
    for scenario in ['S9','S10']:
        a=endpoint[scenario,'scale100_L20'];b=endpoint[scenario,'scale025_L20']
        endpoints.append([scenario,number(100*a['hit100']),number(a['mae_m'],0),
                          number(100*b['hit100']),number(b['mae_m'],0)])
    inference=[]
    for label,alpha in [(r'Chỉ khác một GPS, $r=10$ m',2*.01*10),
                        (r'Chỉ khác một GPS, $r=100$ m',2*.01*100),
                        (r'Cả trace, $D_\infty=100$ m, cap $0{,}23/\mathrm m$',.23*100)]:
        bound=100/(1+math.exp(-alpha))
        inference.append([label,number(alpha,2),r'$\approx100\%$' if alpha>20 else number(bound,2)+r'\%'])
    timeline=[]
    for time in [0,20,60,120,600]:
        event=next(e for e in sample['main_sample']['events'] if e['t_s']==time);pr=event['protection']
        timeline.append([str(time),'Có' if pr['GPS_read'] else 'Không','Tạo mới' if pr['branch']=='fresh' else 'Giữ',
                         'Cập nhật' if pr['GPS_read'] else 'Dự đoán',str(pr['cost_units'])+'u',str(pr['spent_after_units'])+'/23'])
    def texrows(values):return '\n'.join(' & '.join(v)+r'\\' for v in values)
    generated='\n'.join([r'\newcommand{\UtilityRows}{'+utility+'}',
                         r'\newcommand{\CoordinateRows}{'+texrows(coordinates)+'}',
                         r'\newcommand{\EndpointRows}{'+texrows(endpoints)+'}',
                         r'\newcommand{\InferenceRows}{'+texrows(inference)+'}',
                         r'\newcommand{\TimelineRows}{'+texrows(timeline)+'}'])+'\n'
    (HERE/'report_tables.tex').write_text(generated)
    import pymupdf
    source=HERE/'slides.pdf';original=pymupdf.open(source)
    output=pymupdf.open();clip=pymupdf.Rect(688*.75,192*.75,1234*.75,455*.75)
    page=output.new_page(width=clip.width,height=clip.height)
    page.show_pdf_page(page.rect,original,5,clip=clip)
    output.save(figures/'sample_60s.pdf');output.close()
    for name in ['benchmark_tables.json','multistep_sample.json','utility_sample.json','slides.pdf',
                 'slide_content.json','endpoint_focus.json']:
        pins[str((HERE/name).relative_to(ROOT))]=hashlib.sha256((HERE/name).read_bytes()).hexdigest()
    for name in ['protocol.json','results.json','validation.json']:
        f=ROOT/'artifacts/benchmarks/future_native_20261005_v1'/name
        pins[str(f.relative_to(ROOT))]=hashlib.sha256(f.read_bytes()).hexdigest()
    for name in ['docs/research/2026-10-10_session_cap_identity.md','docs/research/2026-10-10_session_cap_identity.json']:
        pins[name]=hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    (HERE/'report_sources.json').write_text(json.dumps(pins,ensure_ascii=False,indent=2)+'\n')
    script=(HERE/'presentation_script.md').read_text()
    parts=re.split(r'^## Trang (\d+): (.+)\n',script,flags=re.MULTILINE)
    if len(parts)!=19 or [int(parts[i]) for i in range(1,len(parts),3)]!=list(range(1,7)):
        raise ValueError('Presenter script must contain exactly six ordered report pages.')
    source_notes=script.split('## Nguồn đối chiếu\n',1)[1]
    notes=[]
    for i in range(1,len(parts),3):
        body=parts[i+2].split('## Nguồn đối chiếu\n',1)[0].strip()
        notes.append(f'TRANG {parts[i]}: {parts[i+1]}\n\n'+body.replace('**',''))
    (HERE/'speaker_notes.txt').write_text('\n\n'.join(notes)+'\n\nNGUỒN ĐỐI CHIẾU\n'+source_notes)
    print(f'Generated native TikZ architectures, five tables and vector sample map; {len(pins)} source pins. No scores regenerated.')


if __name__=='__main__':main()
