"""Scenario definitions and source-backed maps in one reading sequence."""
import json
REPRESENTATIVES={'S1':'A','S2':'C','S3':'C','S9':'B','S10':'A','S5':'C','S6':'C','S8':'C','S4':'B','S7':'C'}
NOTES={
'S1':'u301_00, FCD[333]: một vị trí tại cạnh có 4 hướng đi tiếp. Sao đỏ là chính vị trí cần suy ra; bị trùng với chấm đầu vào.',
'S2':'18 lần lấy mẫu = 9 + 9, cùng một tọa độ. u301_07 dừng tại [214,258] và [1153,1197], mỗi lần 44 giây. Hai lần dừng trùng vị trí nên các chấm chồng nhau; giữa chúng xe có di chuyển.',
'S3':'u301_00: chỉ 4 điểm [361,421,481,541]. Chấm là quan sát rời rạc; nét đứt cho thấy đường thật bộ đánh giá giữ để chấm tái dựng.',
'S9':'Hai phiên u301_00/u301_04 xuất phát cách 281,76 m rồi nhập tuyến. Hai sao đỏ là hai điểm đầu bị che. Cửa sổ bắt đầu sau nhập tuyến.',
'S10':'A che 60 giây cuối một chuyến; C che cùng đoạn cuối trong các phiên lặp. Sao đỏ là endpoint thật chỉ dành cho bộ đánh giá.',
'S5':'Cặp u304_00/u304_02 có tiền tố [0,20,…,100,119]. Hai sao là vị trí ở FCD[122] trên hai cạnh kế tiếp khác nhau; phóng to trong bản tương tác để xem nhánh.',
'S6':'Sáu chuyến u302_14…19 tạo lịch sử (5 đích thường, 1 đích hiếm). Chuyến u302_20 chỉ cấp tiền tố 7 điểm; sao đỏ là đích chưa được cấp. Màu phân biệt bảy phiên.',
'S8':'Cặp u301_00/u301_03 không được gán đồng hành nhưng vẫn gần nhau ở 309/460 thời điểm đồng bộ. Gần trên bản đồ không đủ xác nhận quan hệ; chấm định vị với/không có dữ liệu xe kia.',
'S4':'u301_00/u301_10: cùng người, đổi thiết bị. Hình học lặp giúp liên kết nhưng không tự chứng minh tên thật. Nhãn xe phải suy từ physical_vehicle_id; bản ghi hiện mới có same_person/same_device.',
'S7':'u301_00, FCD [0,20,40]: pharmacy → clinic → hospital; nhãn tổng hợp medical_visit. Khác biệt A/B/C nằm ở nội dung query, không chỉ ở đường đi.'}
def add_dataset_content(ns):
    p,sub,page=ns['p'],ns['sub'],ns['page']
    guide=json.loads((ns['OUT']/'scenario_guide.json').read_text())
    p('**Cách đọc 29 mẫu:** A/B/C là các điều kiện của cùng nhiệm vụ, không phải mức khó tăng dần; riêng S10 giữ A/C. Số sau “/” là số bản ghi. Mỗi ca có map và trục thời gian; phóng to ở sample_maps.html, tra tọa độ trong data_samples.json.')
    p('**Đọc bản đồ:** chấm màu = mẫu được phép trước bảo vệ (nhiều mẫu có thể chồng nhau); trục thời gian = các lần lấy mẫu riêng; nét đứt = toàn chuyến chỉ để đánh giá; sao đỏ = nhãn vị trí. Màu phân biệt phiên, không phải danh tính. Nền SUMO benchmark cùng checksum OSM nhưng chưa xác minh trùng hình học mạng dataset vì thiếu file mạng gốc. © OpenStreetMap contributors.')
    p('**Giới hạn cần giữ khi đánh giá:** S9/S10 là điểm đầu/cuối, chưa phải nhãn nhà/nơi làm việc; truy vấn bị mất vẫn thuộc mẫu số utility. S5/S6 cần đối thủ nền chỉ dùng bản đồ/tần suất. S8 dùng thời gian chung cho cặp. S4 cần bổ sung nhãn xe; S7 dùng ý định tổng hợp, chưa đại diện hành vi người thật.')
    page();ns['sec']('Dataset theo kịch bản: đặc tả đi cùng mẫu trên bản đồ')
    explain=json.loads((ns['OUT']/'scenario_explanations.json').read_text())
    readings=json.loads((ns['OUT']/'case_readings.json').read_text())
    order=list(REPRESENTATIVES)
    for i,name in enumerate(order):
        if i:page()
        g=guide[name];sub(g['title'])
        p('**Rủi ro chung:** '+explain[name]['brief'])
        if name=='S10':
            p('**Hai ca cần giải quyết:** A suy endpoint từ phần cuối còn nhìn thấy và mạng đường; C còn kết hợp các chuyến lặp cùng đích. Giữ mã A/C để truy nguyên dữ liệu. Bài toán chung tiền tố nhưng khác đích thuộc S6.A.')
        for suffix,desc in zip(ns['suffixes'](name),g['cases']):
            case=name+'.'+suffix;r=ns['FIRST'][case]
            counts='; '.join(sid+': '+str(len(ix))+' mẫu' for sid,ix in zip(r['session_ids'],r['observed_indices']))
            if len(r['session_ids'])>3:counts='6 phiên lịch sử + phiên hiện tại: '+str(len(r['observed_indices'][-1]))+' mẫu tiền tố.'
            ns['blocks'].append(('case_panel',(case,desc,readings[case],counts)))
