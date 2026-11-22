# Oracle pipeline cho đồ án T02

## Cài đặt

    pip install -r requirements.txt
    playwright install chromium

## Chạy

Bước 1 — chạy app test chèn payload (giữ terminal này mở):

    python app.py

Bước 2 — lọc dữ liệu Tainted/Obfuscated ban đầu (terminal khác):

    python filter_dataset.py data/raw_payloads.csv --context html_body --seed 0

Bỏ `--context` nếu muốn xử lý toàn bộ trong một lần (chạy trên một máy).
File CSV đầu vào bắt buộc có cột id, payload, group, context, event, family.
Payload được đọc nguyên văn từ payload; event quyết định thao tác Playwright,
và id là khóa checkpoint/merge.

Kết quả ghi ra `oracle_results_{context}_{seed}.csv`, log từng lần gọi ra
`oracle_log_{context}_{seed}.jsonl`. Nếu pipeline bị dừng giữa chừng (tắt máy,
crash), chạy lại đúng lệnh cũ — log JSONL được đọc lại để bỏ qua các payload
đã test xong, không chạy lại từ đầu.

Bước 3 — chạy GA (pilot hoặc full run, cùng một script):

    python -m ga.ga_runner --context html_body --seed 0 --population 20 --generations 10

Checkpoint lưu sau mỗi thế hệ vào `ga_checkpoint_{context}_{seed}.json`. Chạy
lại đúng lệnh trên sẽ tự resume từ thế hệ đã lưu nếu bị dừng giữa chừng.
Kết quả từng thế hệ (fitness tốt nhất, thời gian) ghi vào
`ga_results_{context}_{seed}.jsonl`.

## Chạy song song nhiều máy theo ngữ cảnh

Mỗi máy chạy với `--context` khác nhau (html_body/attribute/js_string/url) và
`--seed` riêng. Tên file output (`oracle_results_*`, `oracle_log_*`,
`ga_checkpoint_*`, `ga_results_*`) đều có context + seed trong tên nên gộp lại
cuối cùng không bị đè lên nhau.

## Oracle: timeout, retry, dọn bộ nhớ

- Timeout load trang: 4 giây (`NAV_TIMEOUT_MS = 4_000` trong `oracle.py`), timeout
  thao tác Playwright 2 giây (`ACTION_TIMEOUT_MS = 2_000`), chờ sự kiện trễ 300ms
  (`WAIT_MS = 300`, hoặc `LONG_WAIT_MS = 1_500` cho nhóm sự kiện media/animation),
  tổng một lần check bị chặn tối đa 8 giây (`OVERALL_TIMEOUT_S = 8.0`) dù
  `page.goto` hay `page.evaluate` có treo vì lý do khác cũng không làm đứng cả
  pipeline.
- Gặp exception (trang crash, lỗi mạng, timeout) → thử lại 1 lần
  (MAX_ATTEMPTS = 2). Hết số lần thử mà vẫn lỗi thì kết luận valid = 0, không
  để treo hàng loạt. Payload chạy xong bình thường nhưng không nổ (fired =
  False) thì không retry, vì đó là kết quả thật chứ không phải lỗi hạ tầng.
- Mỗi lần thử dùng một browser context riêng (new_context/close), không chỉ
  đóng page — tốn thêm chút thời gian mở context nhưng tránh rò rỉ bộ nhớ khi
  chạy hàng chục nghìn lần. Nếu sau khi đo pilot thấy quá chậm, có thể đổi
  lại thành dùng chung context và chỉ đóng page, đánh đổi lấy tốc độ.

## Oracle event policy

event_actions.py là nguồn duy nhất cho bảng event -> thao tác; filter_dataset
và ga_runner gọi chung oracle.py.

- onerror, onload, media load và lifecycle: chờ sự kiện thật.
- onmouseover/onpointerover/onpointerenter: hover().
- onfocus/onfocusin: focus().
- onclick, ondblclick, onmousedown: click/dblclick/mouse down thật.
- onkeyup, oninput, onchange: focus + press, fill, hoặc fill + blur.
- URL chỉ click với family url_js, url_data, anchor_js, svg_xlink; meta_refresh,
  iframe_srcdoc và object_data được chờ tự tải.
- `onstart`/`marquee` được gắn `unsupported_browser` (không tính là
  false-negative Chromium). URL `java%0ascript:` và biến thể encoded whitespace
  được gắn `known_browser_patched`; kết quả CSV giữ `valid=0` kèm
  `oracle_status`, thay vì trộn chúng với payload đã chạy Oracle.

Không route nào auto-click link. app.py chèn payload raw, không dùng Jinja, và
có sink attribute phù hợp: img lỗi cho onerror, input cho keyboard/input.

## Xác minh bắt buộc

Sau khi app đang chạy, chạy:

    python verify_pipeline.py ../malicious_payloads_mixed_3060_v8.csv --concurrency 8 --concurrent-size 60

Lệnh chỉ thành công nếu 3 positive/3 negative known cases đúng, payload không
bị escape trong HTML, concurrent bằng serial, và resume giữa chừng không ghi
lặp hoặc bỏ sót. Nó in bảng mẫu thật (tối đa 5 dòng/event; event hiếm chỉ có
1 dòng), tỷ lệ valid, và toàn bộ ánh xạ event. Nếu một event được đánh dấu
`supported` có toàn bộ mẫu thật `valid=0`, lệnh kết thúc lỗi để buộc điều tra;
không được diễn giải kết quả đó là pipeline OK.

## Những chỗ cần làm thêm

- classifier_stub.py đang trả cố định 0.5, cần thay bằng mô hình thật
  (giữ nguyên tên hàm predict_proba(payload: str) -> float).
- ga/genes.py mới có pool gen cơ bản, cần nhóm bổ sung thêm giá trị.
- Event browser deprecated/không được Chromium hỗ trợ (đặc biệt onstart của
  marquee) có thể vẫn valid=0; báo cáo verify tách riêng chúng, không xem đó
  là payload đã xác minh.
