# Sửa cách giải thích H trong prospective privacy filter

Ngày 06/10/2026. Đây là correction về ký hiệu, không đổi primitive, sampler,
Q, dữ liệu hoặc kết quả đã chốt. Historical fixed-H reports giữ nguyên; report
mới cần diễn giải đúng prospective branch-charged filter hiện tại.

Trong controller fixed-H trước đây, H là hard horizon số events được phép
đọc GPS. Trong `FilteredCoverLaneDummy` kết hợp matched bound/pacing, H được
giữ để tương thích API và hiệu chỉnh ngân sách: **H không còn là hard maximum
số GPS reads**. Filter dùng U=2H−1 units và u=B/(2H); first fresh anchor tốn
1 unit, later reuse tốn 1, later refresh tốn 2. Trước mỗi later read phải còn
ít nhất 2 units, bất kể branch cuối cùng tiêu 1 hay 2.

Với H=12, U=23:

| Branch path | GPS reads | Units đã chi | Quyết định kế tiếp |
|---|---:|---:|---|
| First fresh + 11 refresh | 12 | 1+11×2=23 | Hết cap, không đọc mới |
| First fresh + 21 reuse | 22 | 1+21×1=22 | Còn 1 nhưng phải reserve 2, không đọc lần 23 |

Vì vậy **maximum reads là 22** cho H12 trong reuse-heavy path; all-refresh
path đọc 12 lần. Số đã đọc trong một experiment còn phụ thuộc public ticks,
pacing và branch path. Timestamps là lịch cho phép kiểm tra, không bảo đảm
mỗi tick đọc GPS. Uu=.0575 m⁻¹ cho B=.06 vẫn là cap đúng, không phải B hay H
đếm số samples thực tế. Với H≥2, maximum reads là 2H−2; H1 đọc một lần.

Kiểm tra accounting độc lập sau không gọi sampler hoặc GPS. Nó duyệt toàn
bộ reachable `(spent,reads)` states theo đúng prospective rule:

```python
H = 12
U = 2 * H - 1
seen = {(1, 1)}  # First fresh is compulsory: one read, one unit.
todo = [(1, 1)]
terminal = []
while todo:
    spent, reads = todo.pop()
    if spent + 2 > U:
        terminal.append((spent, reads))
        continue
    for cost in (1, 2):  # Later reuse or refresh.
        state = (spent + cost, reads + 1)
        if state not in seen:
            seen.add(state)
            todo.append(state)
assert min(reads for _, reads in terminal) == 12
assert max(reads for _, reads in terminal) == 22
assert all(spent <= U for spent, _ in seen)
```

Đã chạy bằng Python 3.11: 143 reachable states; terminal read range 12–22.
All-reuse terminal `(spent=22, reads=22)` và all-refresh terminal
`(spent=23, reads=12)`. Đây là kiểm tra deterministic logic, không phải ước lượng
branch frequencies hoặc một thí nghiệm privacy mới.

Source: [prospective filter](../../benchmark/engines/filtered_cover.py),
[matched U=2H−1](../../benchmark/engines/matched_filter.py),
[public pacing](../../benchmark/engines/paced_guard.py) và
[GPS supplier boundary](../../core/session_budget.py).
Nếu sản phẩm cần “tối đa 12 GPS reads”, phải thêm một public read-count ceiling
riêng. Không cần sửa sampler Geo-I để sửa ký hiệu hoặc thêm ceiling đó.
