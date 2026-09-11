import React from 'react';
import { RichNarrative, DataComponent, DataTable, EvidenceChart, useDataApp } from '../../data-app-public.jsx';
import content from './artifact.json';
import pageStarts from './presentation_pages.json';

const groups = [];
for (const block of content.manifest.blocks) {
  if (!groups.length || pageStarts.includes(block.id)) groups.push([]);
  groups.at(-1).push(block);
}
const specs = {
  'recall-chart': {type:'bar', x:'method', y:'recall', series:'depth', stackable:false,
    yLabel:'Recall@5 (%)', startAtZero:true, valueDecimals:2, showValues:true,
    colors:{'L = 5':'#2563A6','L = 10':'#C47C21'}},
  'family-chart': {type:'bar', x:'family', y:'delta', stackable:false,
    yLabel:'Chênh lệch (điểm %)', startAtZero:true, valueDecimals:2, showValues:true,
    colorBySign:false, colors:{delta:'#2563A6'}},
};
const sourcePreviews = {
  'https://arxiv.org/html/2409.09495v1': {title:'TransProtect',summary:'Cơ chế dùng ngữ cảnh đường; bài gốc đánh giá sai số suy luận kỳ vọng và sai lệch chi phí hành trình.',source:'Bản tác giả',approvedForReport:true},
  'https://link.springer.com/article/10.1007/s44443-026-00899-w': {title:'Semantic correlation',summary:'Bài định nghĩa ASR và DER cho tập thật–giả; các chỉ số này không phải Recall POI.',source:'Toàn văn nhà xuất bản',approvedForReport:true},
  'https://link.springer.com/article/10.1007/s44443-025-00438-z': {title:'Fake-query insertion',summary:'Chèn truy vấn giả để làm khó nối chuỗi; cần tính cả chi phí bản tin chèn.',source:'Toàn văn nhà xuất bản',approvedForReport:true},
  'https://arxiv.org/abs/1409.1716': {title:'Prolonging the Hide-and-Seek Game',summary:'Tối ưu cơ chế che vị trí có xét tương quan và mục tiêu hiện tại/tương lai, dưới mô hình di chuyển và đối thủ đã định.',source:'Bản tác giả',approvedForReport:true},
};

export function ReportContent() {
  const {snapshot, visible} = useDataApp();
  function blockView(block) {
    if (!visible(block.id)) return null;
    if (block.type === 'markdown') return <RichNarrative key={block.id} id={block.id}
      value={block.body} sourcePreviews={sourcePreviews} className={'brief-copy '+block.id} label="Sửa nội dung" />;
    if (block.type === 'html') return <div key={block.id} className="brief-flow"
      data-reviewed-rows dangerouslySetInnerHTML={{__html:block.body}} />;
    if (block.type === 'chart') {
      const chart = content.manifest.charts.find(c=>c.id===block.chartId);
      const rows = snapshot.queries[chart.dataset].rows;
      return <EvidenceChart key={block.id} id={chart.id} queryId={chart.dataset}
        title={chart.title} description={chart.subtitle} spec={specs[chart.id]}
        rows={rows} sourceRows={rows} height={chart.id==='family-chart'?210:190} className="brief-chart">
        {chart.id==='family-chart' && <RichNarrative id="family-chart:interpretation" className="brief-copy"
          value="Hai chế độ − phủ + thay điểm: âm là giảm rủi ro, dương là tăng. Nhóm 307 và 312 bằng 0." />}
      </EvidenceChart>;
    }
    const table = content.manifest.tables.find(t=>t.id===block.tableId);
    const rows = snapshot.queries[table.dataset].rows;
    const columns = table.columns.map(column => column.field !== 'related' ? column : {
      ...column,
      renderCell: value => <div className="brief-method-lines">{value.split('\n').map((line,i) => {
        const colon = line.indexOf(':');
        return <div key={i}>{colon < 0 ? line : <><strong>{line.slice(0,colon)}:</strong>{line.slice(colon+1)}</>}</div>;
      })}</div>,
    });
    const render = part => <DataTable rows={part} columns={columns}
      searchable={false} compactNumbers={false} label={table.title} />;
    return <DataComponent key={block.id} id={table.id} title={table.title}
      kind="table" queryId={table.dataset} sourceRows={rows} displayRows={rows}
      className={'brief-table '+table.id}>
      {rows.length>8 ? <div className="brief-table-pair">{render(rows.slice(0,5))}{render(rows.slice(5))}</div> : render(rows)}
    </DataComponent>;
  }
  return <article className="report-content supervisor-brief" aria-label="Báo cáo trao đổi với GVHD">
    {groups.map((blocks,i)=><section key={blocks[0].id} className={'brief-page brief-page-'+(i+1)}>
      <div className="brief-kicker">LUẬN VĂN TỐT NGHIỆP · TRAO ĐỔI VỚI GVHD · 11/09/2026</div>
      {blocks.map(blockView)}
      <div className="brief-footer">Bằng chứng đến 11/09/2026 <span>{i+1} / {groups.length}</span></div>
    </section>)}
  </article>;
}
