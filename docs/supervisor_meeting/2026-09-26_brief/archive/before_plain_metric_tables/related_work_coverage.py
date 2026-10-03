"""Task-level literature mapping, not a claim of passing our dataset cases."""

LEGEND = ('**Fully (F):** paper trực tiếp xử lý mục tiêu suy luận của scenario, trong giả định của paper. '
          '**Partially (P):** chỉ xử lý một phần hoặc bối cảnh hẹp hơn. **CX:** chưa đủ nguồn để phân loại. '
          '**—:** chưa tìm thấy bằng chứng xử lý trực tiếp; không kết luận cơ chế chắc chắn thất bại. '
          'F/P là đối chiếu của ta, không phải nhãn do tác giả công bố, không có nghĩa bảo vệ 100% hoặc đã vượt các ca A/B/C của ta.')
SCENARIOS = ('S1: vị trí hiện tại; S2: nơi dừng; S3: đoạn đường đã đi; S4: liên kết danh tính giữa phiên; '
             'S5: cạnh đường kế tiếp; S6: đích chưa tới; S7: nội dung/ý định truy vấn; '
             'S8: định vị nhờ dữ liệu người đi cùng; S9: điểm đầu bị che; S10: điểm cuối bị che của chuyến đã hoàn tất.')
ROWS = [
    ['TransProtect 2024 [R1]', 'S1, S3', 'S2', '§3–5: suy vị trí từ chuỗi đã làm nhiễu, biết đường/traffic; chưa đánh giá riêng nơi dừng.'],
    ['Semantic correlation 2026 [R2]', 'S1, S3', 'S2', '§3–6: giữ dummy hợp lý theo thời gian/ngữ nghĩa; chưa chấm riêng nơi dừng hay ý định S7.'],
    ['Fake queries 2026 [R3]', 'S1, S3', 'S2', '§3–6: chèn truy vấn giả để khó nối tuyến; chưa đánh giá riêng tọa độ nơi dừng.'],
    ['Road-aware PTPPM 2025 [R4]', 'S1, S3', 'S2', '§III, VI: chống suy vị trí/quỹ đạo có tương quan; bước dự báo chưa là phép thử S5/S6.'],
    ['CPCROK 2025 [R5]', '—', 'S4', '§3–5: chống nối bí danh xe trong VANET; chưa tách danh tính người/thiết bị như S4.A/B/C.'],
    ['DP-FETC 2025 [R6]', '—', 'S3, S8, S9, S10', 'SynFE/PreOD: sinh dữ liệu offline, đổi đầu/cuối của tuyến tương quan; không chấm định vị từ người đi cùng.'],
    ['Improved PIR 2025 [R7]', 'CX', 'CX: S1, S7', 'Tóm tắt nêu bảo vệ vị trí/nội dung bằng mã hóa; thiếu toàn văn để đối chiếu quyền quan sát và suy ý định.'],
    ['SoK 2024 [R8]', 'Không áp dụng', 'Không áp dụng', 'Khảo sát cách đánh giá/tấn công; không phải cơ chế bảo vệ để gán F/P.'],
    ['PRISM 2025 [R9]', 'CX', 'CX: S1, S3', 'Tóm tắt: bảo vệ chia sẻ quỹ đạo; chưa đủ mô hình attacker. Dùng LSTM không xác nhận xử lý S5/S6.'],
    ['Options to Action 2026 [R10]', 'Không áp dụng', 'Không áp dụng', 'Khảo sát sử dụng tính năng, có che đầu/cuối (S9/S10); không đo khả năng chống suy luận.'],
    ['Forsch et al. 2023 [R11]', 'S9, S10', '—', '§3.2: S-TT cắt cả hai đầu, xét nhiều tuyến cùng đích; bảo đảm theo tập tấn công/địa điểm cho trước, offline.'],
    ['EV querying 2024 [R12]', 'S1, S3', 'S2', 'Proposed Query Model: AGeoI + dummy chống định vị/nối tuyến online; chưa chấm riêng nơi dừng hoặc S4–S10.'],
]
NOTE = ('Scenario không được liệt kê ở một hàng được hiểu là —; với R7/R9 là chưa xác minh, '
        'với R8/R10 là không áp dụng. Bảng cho thấy các cơ chế chuỗi chủ yếu xét S1/S3; '
        'S2 cần phép thử nơi dừng riêng, còn S9/S10 cần kiểm tra suy ngược từ phần tuyến vẫn công bố.')

def coverage_evidence(refs):
    return {'checked_on': '2026-10-03', 'mapping_is_our_inference': True,
            'granularity': 'scenario task within paper assumptions; not all A/B/C cases',
            'legend': LEGEND, 'scenario_key': SCENARIOS, 'rows': ROWS, 'note': NOTE,
            'sources': [{'id': rid, 'url': url, 'access_note': verify}
                        for rid, _, url, _, verify, _ in refs if rid.startswith('R')]}
