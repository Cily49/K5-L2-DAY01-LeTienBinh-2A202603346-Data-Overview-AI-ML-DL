# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:**

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468; `class_name`=cab ; `rank`=1 ; `score`=0.510915 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    
    Cả bức ảnh. Record này không có tọa độ box hay polygon gì cả, nó chỉ có tên lớp và score, nên model không chỉ ra "cab" nằm ở chỗ nào trong ảnh. Nó chỉ đoán cả ảnh giống "cab" nhất.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?

    Đi kèm checkpoint. Model `yolo11n-cls.pt` được train trên ImageNet-1K nên chỉ chọn được trong 1000 lớp có sẵn của ImageNet, không tự nghĩ ra tên mới. Vì ImageNet không có lớp kiểu "đường phố" hay "giao thông" nên nó chọn lớp gần nhất là cab.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?

    Vì chỉ ghi tên thì dễ nhầm. Mỗi taxonomy có danh sách lớp và số ID khác nhau, ví dụ cab là lớp 468 bên ImageNet, còn COCO lại không có lớp cab mà chỉ có car, bus. Ghi đủ cả ba thì sẽ biết chính xác nhãn đó thuộc bộ nào, lớp nào.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?

    Guideline phải nói rõ là chọn theo vật chính trong ảnh hay theo cả khung cảnh, và vật chính được hiểu là gì. Ngoài ra cần ghi rõ dự án chỉ cho chọn 1 lớp hay được chọn nhiều lớp, và nếu không có lớp nào hợp thì báo Lab Coach. Như ảnh traffic có cả xe buýt, ô tô, người đi bộ, nếu không có quy tắc thì mỗi người sẽ chọn một kiểu.
6. Vì sao không được lấy `score` của model làm ground truth?

    Vì score chỉ là mức độ model "tự tin", không có nghĩa là đúng. Ở ảnh này model đoán cab với 0.51, nhưng nhìn vào ảnh thì thứ nổi bật nhất là mấy chiếc xe buýt xanh, mình không thấy rõ chiếc taxi nào. Nhãn đúng phải do người gán theo guideline rồi có người kiểm tra lại.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?

    Đây là box của người đàn ông mặc tạp dề đứng quay lưng ở giữa bên phải ảnh. Ảnh rộng 640, cao 427 pixel, gốc (0, 0) ở góc trên-trái. Góc trên-trái của box ở (385.33, 69.24), tức là cách mép trái ảnh khoảng 385 pixel và cách mép trên khoảng 69 pixel, ngang chỗ đầu người. Góc dưới-phải ở (498.92, 348.92), khoảng chỗ chân. Box rộng 113.58 pixel (498.92 − 385.33) và cao 279.68 pixel (348.92 − 69.24). Box cao gấp khoảng 2.5 lần chiều rộng, hợp với dáng một người đang đứng.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?

    Ở ngưỡng 0.20 có 17 box, ở 0.60 chỉ còn 6 box (person, bowl, bowl, oven, oven, person). 
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?

    Phải xem nhiều box hơn và kiểm tra từng cái xem có đúng vật đó không, có bị trùng box không, box có lệch không. 
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).

    Vẽ box ôm sát phần nhìn thấy được của vật, không chừa nhiều nền xung quanh, mỗi vật một box riêng. Vật bị che thì chỉ vẽ phần thấy được, không đoán phần bị che. Vật bị cắt ở mép ảnh thì vẫn vẽ, cho box chạm mép ảnh.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?

    Box "`person 0.61`" ở mép trái ảnh: thật ra chỉ thấy một cánh tay thò vào từ ngoài khung hình, người đó bị cắt gần hết. Trường hợp này tôi sẽ viết thành quy tắc trong guideline, vì người hay vật bị cắt ở mép ảnh gặp rất thường xuyên, không phải trường hợp hiếm. Chỉ khi phần thấy được quá nhỏ đến mức không chắc là vật gì thì mới hỏi Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899318 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`= [[446.0,70.0],[445.0,71.0],[444.0,71.0],[443.0,72.0],[442.0,72.0]]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?

    Box của kitchen-001 là [385.45, 66.44, 498.02, 348.58], gần như trùng với box person 0.91 ở mục 2. Nhưng box chỉ là hình chữ nhật nên bị lẫn cả nền vào, ví dụ khoảng trống giữa hai chân, hai bên đầu và chỗ khuỷu tay chống hông. Polygon thì đi theo đúng đường viền người: phần đầu, vai, khuỷu tay và hai chân, nên biết được pixel nào là người, pixel nào là nền.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?

    `instance_id` là số thứ tự để phân biệt từng vật riêng trong ảnh này, ví dụ kitchen-001 là người đứng giữa, còn người ở mép trái sẽ có id khác dù cùng là person. Nó không phải class ID vì class ID (person = 0) chỉ cho biết loại vật, hai người thì class ID giống nhau. Nó cũng không phải tracking ID vì không dùng để theo dõi cùng một người qua nhiều frame video, chạy ảnh khác thì số này đánh lại từ đầu.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.

    Vẽ biên bám sát mép ngoài phần nhìn thấy của vật, không tô sang nền hay vật khác. Chỗ hở bên trong (như khoảng giữa hai chân) thì không tô. Hai vật chạm nhau thì mỗi pixel chỉ thuộc về một vật, vật nào ở phía trước thì được phần chồng lên.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?

    Cái bàn gỗ (dining table 0.58) và đồ đặt trên bàn chạm nhau. Nhìn ảnh segmentation thì mask màu xanh của bàn tô lan lên cả hũ mứt, bột và con lăn bột đặt trên mặt bàn, còn mấy cái bowl ở góc trái thì mask bàn và mask bowl chồng lên nhau. Trường hợp này tôi dùng guideline, vì đồ vật đặt trên bàn là chuyện gặp ở rất nhiều ảnh. Còn những thứ như đống bột không có lớp nào trong COCO thì có nên tính vào bàn hay không, tôi sẽ hỏi Lab Coach.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn lớp cho cả ảnh (class_id + class_name + taxonomy) | Ảnh `traffic` top 1 là `cab` (0.51) nhưng trong ảnh nổi bật nhất là xe buýt xanh, `minibus` chỉ 0.16 | Chọn 1 lớp theo quy tắc vật chính, không có lớp hợp thì báo Lab Coach | Lớp có đúng taxonomy và đúng quy tắc chọn vật chính không |
| Phát hiện vật thể | Mỗi vật 1 box xyxy (pixel) + 1 lớp | Ảnh `kitchen`: cái bàn gỗ to chỉ hiện ở ngưỡng 0.20, ở 0.35 bị bỏ sót. Box `person 0.61` ở mép trái chỉ là một cánh tay | Vẽ box sát phần nhìn thấy cho mọi vật, kể cả vật model bỏ sót | Có box nào thiếu, thừa, trùng, lệch hay sai lớp không |
| Instance segmentation | Mỗi vật 1 polygon/mask + lớp, mỗi vật một instance riêng | Mask `dining table 0.58` tô lan lên hũ mứt, bột trên bàn và chồng lên mask bowl. Người ở mép trái (`person 0.43`) chỉ thấy cánh tay | Vẽ polygon theo mép vật, tách riêng từng instance | Biên có lệch không, mask có tô lấn sang vật khác không, có sót vật không |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)

   Chỉ dùng 3 ảnh COCO công khai có sẵn trong notebook, không đưa ảnh cá nhân hay dữ liệu nội bộ lên Google Colab/GitHub, và không ghi họ tên, MSSV vào báo cáo hay output.

2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?

    Tôi sẽ dừng lại, không dùng ảnh đó và báo cho Lab Coach hoặc giảng viên.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
