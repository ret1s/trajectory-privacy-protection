"""Scenario definitions and source-backed maps in one reading sequence."""
import json
REPRESENTATIVES={'S1':'A','S2':'C','S3':'C','S9':'B','S10':'B','S5':'C','S6':'C','S8':'C','S4':'B','S7':'C'}
NOTES={
'S1':'u301_00, FCD[333]: một vị trí tại cạnh có 4 hướng đi tiếp. Sao đỏ là chính vị trí cần suy ra; bị trùng với chấm đầu vào.',
'S2':'u301_07 dừng tại [214,258] và [1153,1197], mỗi lần 44 giây. Hai lần dừng trùng vị trí nên các chấm chồng nhau; giữa chúng xe có di chuyển.',
'S3':'u301_00: chỉ 4 điểm [361,421,481,541]. Chấm là quan sát rời rạc; nét đứt cho thấy đường thật bộ đánh giá giữ để chấm tái dựng.',
'S9':'Hai phiên u301_00/u301_04 xuất phát cách 281,76 m rồi nhập tuyến. Hai sao đỏ là hai điểm đầu bị che. Cửa sổ bắt đầu sau nhập tuyến.',
'S10':'u304_00/u304_02 chung tiền tố nhưng kết thúc cách 4,47 km. Chấm màu chỉ tiền tố; hai sao đỏ là FCD cuối [693,483], giữ kín để chấm ngoại tuyến.',
'S5':'Cặp u304_00/u304_02 có tiền tố [0,20,…,100,119]. Hai sao là vị trí ở FCD[122] trên hai cạnh kế tiếp khác nhau; phóng to trong bản tương tác để xem nhánh.',
'S6':'Sáu chuyến u302_14…19 tạo lịch sử (5 đích thường, 1 đích hiếm). Chuyến u302_20 chỉ cấp tiền tố 7 điểm; sao đỏ là đích chưa được cấp. Màu phân biệt bảy phiên.',
'S8':'Cặp u301_00/u301_03 không được gán đồng hành nhưng vẫn gần nhau ở 309/460 thời điểm đồng bộ. Gần trên bản đồ không đủ xác nhận quan hệ; chấm định vị với/không có dữ liệu xe kia.',
'S4':'u301_00/u301_10: cùng người, đổi thiết bị. Hình học lặp giúp liên kết nhưng không tự chứng minh tên thật. Nhãn xe phải suy từ physical_vehicle_id; bản ghi hiện mới có same_person/same_device.',
'S7':'u301_00, FCD [0,20,40]: pharmacy → clinic → hospital; nhãn tổng hợp medical_visit. Khác biệt A/B/C nằm ở nội dung query, không chỉ ở đường đi.'}
def add_dataset_content(ns):
    p,sub,page=ns['p'],ns['sub'],ns['page']
    guide=json.loads((ns['OUT']/'scenario_guide.json').read_text())
    sub('Cách đọc các mẫu trên bản đồ')
    p('A/B/C là ba điều kiện của cùng nhiệm vụ, không phải mức khó tăng dần. Số sau dấu “/” là số bản ghi. Mỗi scenario có một map đại diện lớn; đủ 30 map và tọa độ chi tiết nằm trong sample_maps.html / sample_maps.pdf và data_samples.json.')
    p('**Đọc bản đồ:** chấm màu = mẫu được phép trước bảo vệ; nét đứt = toàn chuyến chỉ để đánh giá; sao đỏ = nhãn vị trí. Màu phân biệt phiên, không phải danh tính. Nền SUMO benchmark cùng checksum OSM nhưng chưa xác minh trùng hình học mạng dataset vì thiếu file mạng gốc. © OpenStreetMap contributors.')
    page();ns['sec']('Dataset theo kịch bản: đặc tả đi cùng mẫu trên bản đồ')
    order=list(REPRESENTATIVES)
    for i,name in enumerate(order):
        if i and i%2==0:page()
        g=guide[name];sub(g['title']);case=name+'.'+REPRESENTATIVES[name]
        ns['blocks'].append(('scenario_panel',(name,case,g['cases'],NOTES[name])))
    p('**Giới hạn cần giữ khi đánh giá:** S9/S10 là điểm đầu/cuối, chưa phải nhãn nhà/nơi làm việc; truy vấn bị mất vẫn thuộc mẫu số utility. S5/S6 cần đối thủ nền chỉ dùng bản đồ/tần suất. S8 dùng thời gian chung cho cặp. S4 cần bổ sung nhãn xe; S7 dùng ý định tổng hợp, chưa đại diện hành vi người thật.')
