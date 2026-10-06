"""Render the same reviewed content as HTML and editable LaTeX.

Run prepare_report.py first. PDF compilation is a separate Tectonic/XeLaTeX
step; no benchmark, dataset or previous meeting document is modified.
"""
from html import escape
import json
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent

CSS = '''
:root{--ink:#193c49;--teal:#147d7b;--muted:#596c75;--line:#cedbdc}
*{box-sizing:border-box}body{margin:0;background:#edf1ef;color:#172d37;
font:17px/1.55 Georgia,"Times New Roman",serif}main{max-width:1080px;margin:24px auto;
background:white;padding:38px 54px}header{border-bottom:2px solid var(--teal);padding-bottom:22px}
h1{font-size:31px;line-height:1.25;margin:0 0 10px;color:var(--ink)}
h2{font-size:25px;line-height:1.3;color:var(--ink);margin:30px 0 14px}
h3{font-size:20px;color:var(--teal)}p{margin:11px 0}a{color:var(--teal)}
.muted,.caption{color:var(--muted);font-size:15px}.status{font-size:14px;color:var(--muted)}
.key{border-left:3px solid var(--teal);background:#eaf4f2;padding:12px 16px;font-weight:bold}
.table-wrap{overflow:auto;margin:16px 0}table{border-collapse:collapse;width:100%;font-size:15px;line-height:1.4}
th,td{padding:8px 7px;text-align:left;vertical-align:top;border-bottom:1px solid var(--line)}
th{background:#f4f8f7;border-top:2px solid var(--ink)}tr:last-child td{border-bottom:2px solid var(--ink)}
figure{margin:15px 0}svg{width:100%;height:auto;max-height:810px}svg text{font-family:Arial,sans-serif!important}
li{margin:5px 0}.page{border-bottom:1px solid var(--line);padding-bottom:18px;margin-bottom:25px}
.sources{font-size:14px;color:var(--muted)}.sources li{overflow-wrap:anywhere}
@media(max-width:700px){main{margin:0;padding:22px 16px}body{font-size:16px}h1{font-size:26px}
table{min-width:650px}svg{max-height:none}}
@media print{body{background:white}main{padding:0;margin:0;max-width:none}.page{break-before:page;border:0}
.page:first-of-type{break-before:auto}.toolbar,.status{display:none}tr{break-inside:avoid}}
'''

HEADER = r'''\documentclass[11pt,a4paper]{article}
\usepackage{fontspec}
\setmainfont{texgyretermes}[Extension=.otf,UprightFont=*-regular,BoldFont=*-bold,ItalicFont=*-italic,BoldItalicFont=*-bolditalic]
\usepackage[margin=1.8cm,headheight=14pt]{geometry}
\usepackage{booktabs,longtable,array,ragged2e,enumitem,microtype,xcolor,graphicx,fancyhdr,amsmath}
\usepackage[hidelinks,unicode]{hyperref}
\definecolor{ink}{HTML}{193C49}\definecolor{teal}{HTML}{147D7B}\definecolor{pale}{HTML}{EAF4F2}
\newcolumntype{P}[1]{>{\RaggedRight\arraybackslash}p{#1}}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\emergencystretch}{2em}
\setlength{\tabcolsep}{4pt}\renewcommand{\arraystretch}{1.14}
\setlist{nosep,leftmargin=1.3em,topsep=4pt}
\pagestyle{fancy}\fancyhf{}\fancyhead[L]{\small BẢO VỆ RIÊNG TƯ QUỸ ĐẠO}
\fancyhead[R]{\small Chuẩn bị 06/10/2026}\fancyfoot[C]{\small\thepage}
\renewcommand{\headrulewidth}{0.3pt}
\newcommand{\key}[1]{\par\smallskip\noindent\colorbox{pale}{\parbox{\dimexpr\linewidth-2\fboxsep}{\textbf{#1}}}\par\smallskip}
\begin{document}
'''


def tex_plain(value):
    special = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
               '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
               '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
               '↑': r'$\uparrow$', '↓': r'$\downarrow$', '→': r'$\rightarrow$',
               '≥': r'$\geq$', '≤': r'$\leq$', 'ε': r'$\varepsilon$',
               'α': r'$\alpha$', '×': r'$\times$'}
    return ''.join(special.get(c,c) for c in str(value))


def markup(value, kind):
    parts = re.split(r'(\*\*.*?\*\*|\[[^\]]+\]\([^)]+\))', str(value))
    output=[]
    for part in parts:
        if part.startswith('**') and part.endswith('**'):
            body=markup(part[2:-2],kind)
            output.append(f'<strong>{body}</strong>' if kind=='html' else r'\textbf{'+body+'}')
        elif (match := re.fullmatch(r'\[([^\]]+)\]\(([^)]+)\)',part)):
            title,url=match.groups()
            output.append(f'<a href="{escape(url,quote=True)}">{escape(title)}</a>' if kind=='html'
                          else r'\href{'+tex_plain(url)+'}{'+tex_plain(title)+'}')
        else:
            output.append(escape(part) if kind=='html' else tex_plain(part))
    return ''.join(output)


def render(block,kind):
    typ=block['type']
    if typ in ('p','key','h3'):
        text=markup(block['text'],kind)
        if kind=='html':
            return f'<h3>{text}</h3>' if typ=='h3' else f'<p class="{typ}">{text}</p>'
        return (r'\subsection*{'+text+'}\n' if typ=='h3' else
                r'\key{'+text+'}\n' if typ=='key' else text+'\n\n')
    if typ=='list':
        items=[markup(v,kind) for v in block['items']]
        return '<ul>'+''.join(f'<li>{v}</li>' for v in items)+'</ul>' if kind=='html' else (
            '\\begin{itemize}\n'+''.join('\\item '+v+'\n' for v in items)+'\\end{itemize}\n')
    if typ=='table':
        headers=[markup(v,kind) for v in block['headers']]
        rows=[[markup(v,kind) for v in row] for row in block['rows']]
        assert all(len(r)==len(headers) for r in rows)
        if kind=='html':
            return '<div class="table-wrap"><table><thead><tr>'+''.join(f'<th>{v}</th>' for v in headers)+(
                '</tr></thead><tbody>'+''.join('<tr>'+''.join(f'<td>{v}</td>' for v in r)+'</tr>' for r in rows)+
                '</tbody></table></div>')
        relative=block.get('widths',[1]*len(headers))
        # Account for the spaces between columns within the 17.4cm text width.
        total=17.4-(len(headers)-1)*.29
        widths=[total*w/sum(relative) for w in relative]
        columns='@{}'+''.join(f'P{{{v:.3f}cm}}' for v in widths)+'@{}'
        result='\\begingroup\\small\n\\begin{longtable}{'+columns+'}\n\\toprule\n'
        result+=' & '.join(r'\textbf{'+h+'}' for h in headers)+r'\\\midrule'+'\n'
        result+=r'\endfirsthead'+'\n'+r'\toprule'+'\n'
        result+=' & '.join(r'\textbf{'+h+'}' for h in headers)+r'\\\midrule'+'\n'
        result+=r'\endhead\bottomrule\endfoot'+'\n'
        result+=''.join(' & '.join(row)+r'\\'+'\n'+r'\addlinespace[2pt]'+'\n' for row in rows)
        return result+'\\end{longtable}\n\\endgroup\n'
    if typ=='figure':
        caption=markup(block.get('caption',''),kind)
        if kind=='html':
            svg=(OUT/block['svg']).read_text()
            svg=svg[svg.index('<svg'):]
            return '<figure role="img" aria-label="'+escape(block['alt'],quote=True)+'">'+svg+f'<figcaption class="caption">{caption}</figcaption></figure>'
        return r'\begin{center}\includegraphics[width=\linewidth,height='+str(block.get('height_cm',18))+(
            r'cm,keepaspectratio]{'+tex_plain(block['pdf'])+r'}\end{center}'+'\n'+r'{\small '+caption+'}\n')
    raise ValueError(f'Unknown authored block: {typ}')


def build(document,filename,status):
    title=markup(document['title'],'html')
    html=['<!doctype html><html lang="vi"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">',
          '<title>'+escape(document['title'])+'</title><style>'+CSS+'</style><body><main>',
          '<header><h1>'+title+'</h1><p>'+markup(document['subtitle'],'html')+'</p>',
          '<p class="muted">Chuẩn bị 06/10/2026 · Buổi GVHD tiếp theo · Geo-I là nền tảng</p>',
          '<p class="toolbar"><a href="'+filename+'.pdf">PDF</a> · <a href="'+filename+'.tex">LaTeX</a> · <a href="evidence.json">Evidence</a> · <a href="'+(
              'preparation_guide.html' if filename=='report_explained' else 'report_explained.html')+'">'+(
              'Lời nói gợi ý' if filename=='report_explained' else 'Report')+'</a></p>',
          '<p class="status">'+('Đang hoàn thiện nội dung và kiểm chứng.' if status!='complete' else 'Đã dựng và kiểm chứng nội dung.')+'</p></header>']
    tex=[HEADER,r'\begin{center}{\LARGE\bfseries\color{ink}'+tex_plain(document['title'])+r'}\\[5pt]{\large '+tex_plain(document['subtitle'])+r'}\\[4pt]{\small Chuẩn bị 06/10/2026 · Buổi GVHD tiếp theo · Geo-I là nền tảng}\end{center}'+'\n']
    for i,page in enumerate(document['pages']):
        html.append('<section class="page" id="section-'+str(i+1)+'"><h2>'+markup(page['heading'],'html')+'</h2>')
        if i:tex.append('\\clearpage\n')
        tex.append(r'\section*{'+markup(page['heading'],'tex')+'}\n')
        for block in page['blocks']:
            html.append(render(block,'html'));tex.append(render(block,'tex'))
        html.append('</section>')
    html.append('</main></body></html>');tex.append('\\end{document}\n')
    (OUT/(filename+'.html')).write_text('\n'.join(html))
    (OUT/(filename+'.tex')).write_text(''.join(tex))


def main():
    content=json.loads((OUT/'content.json').read_text())
    for key,name in [('report','report_explained'),('preparation','preparation_guide')]:
        build(content[key],name,content['build_status'])
    print('Generated shared HTML/LaTeX:',content['build_status'])


if __name__=='__main__':
    main()
