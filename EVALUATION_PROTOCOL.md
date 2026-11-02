# Phase 5B.3A — Dataset audit và thiết kế đánh giá

**Audit/thiết kế: COMPLETE. Split: NOT FINALIZED.**
Không chạy recognition, tính accuracy/FAR/FRR, chọn threshold, tạo registry hoặc
thay đổi inference trong phase này. Các con số bên dưới là inventory và kích
thước thiết kế, không phải kết quả hiệu năng.

## 1. Phạm vi và nguồn

Quét ba project D:/Facenet-Recognition, D:/MTCNN, D:/Face-Recognition-System,
loại trừ .venv/.git/cache. Không tuyên bố đã tìm toàn bộ ổ D hoặc máy.
Đọc source enrollment/recognition, adapter, MTCNN alignment và notebooks
03/04/05/07/08/09/10; không Run All hoặc thực thi cell inference.

[Báo cáo IMM gốc](https://www2.imm.dtu.dk/pubdb/edoc/imm3160.pdf) mô tả
240 JPEG, 40 người, sáu kiểu ảnh/người, ảnh 640x480, 58 landmark ASF.
Đây là thông tin công bố, không phải bằng chứng full dataset đang nằm local.

Hồ sơ local ghi nguồn OpenIMAJ mirror, archive hash tự tính đã được người dùng
duyệt trước đây. Hash local khớp manifest; không có checksum DTU để chứng minh
archive mirror giống byte gốc nhà phát hành. Chỉ dùng nội bộ đồ án theo hồ sơ
giáo dục/nghiên cứu đã có, kèm trích dẫn; không xuất bản lại ảnh.

## 2. Inventory thực tế

| Nhóm | Ảnh | Danh tính có nhãn | Kích thước / trạng thái |
|---|---:|---:|---|
| IMM raw Known, data/sample_faces | 25 JPEG | 5 | 640x480 RGB uint8; 5/người |
| IMM raw Unknown, data/unknown_faces_imm | 48 JPEG | 8 | 640x480 RGB uint8; 6/người |
| Known PNG alignment ASF | 25 PNG | cùng 5 | 160x160, không phải ảnh độc lập mới |
| Unknown PNG alignment ASF | 46 PNG | cùng 8 | 160x160; hai nguồn không có PNG |
| MTCNN aligned_faces outputs | 5 PNG | Chưa xác minh | 160x160; không dùng cho accuracy |
| MTCNN demo/ảnh output khác | 2 file | Chưa xác minh | Một ảnh nhóm và một output chưa rõ nguồn |
| Integration fixtures | 2 file | Không thêm danh tính | Hai bản sao từ MTCNN |

Tổng 153 file ảnh, 153 đọc được. Có 73 JPEG IMM nguyên gốc, 13 mã người, 73 ASF;
71 ASF có bảy trường/dòng, hai ASF có mười trường/dòng.
Full IMM không có trong ba project: thiếu 27/40 mã người; năm mã local chỉ có
loại ảnh 1-5. Không có archive dataset local hoặc dataset ảnh khác ngoài các
nhóm đã liệt kê trong phạm vi quét.

Mã IMM 01/05/06/07/08 có 5 ảnh/mã; 09-16 có 6 ảnh/mã.
Alias cũ person_01..person_05 tương ứng IMM 01/05/06/07/08; person_02 KHÔNG
phải IMM02. Báo cáo thêm reference IMM:NN từ provenance, không đổi nhãn/file.
Các unknown_imm_NN giữ nguyên nhãn gốc. Known/unknown là vai trò theo registry,
không phải thuộc tính cố định của một người.

Header kiểm tra thấy MTCNN/face.jpg và fixture mtcnn_multi.jpg thực chất là
PNG RGBA 666x409; không phải JPEG ba kênh. detected.jpg là JPEG 1200x675,
không đủ provenance để khẳng định nguồn giống ảnh demo. Không sửa các file này.
Unique identity count của ảnh nhóm/crops không thể suy ra từ tên file hoặc
số detection. Cần manual review nếu muốn đưa chúng vào đánh giá có ground truth;
hiện loại khỏi protocol.

## 3. Integrity, duplicate và lịch sử exposure

- 73 JPEG raw khác cả SHA-256 byte và hash decoded pixels.
- Có hai nhóm duplicate byte/pixel: ảnh demo -> multi fixture, face_1.png ->
  single fixture. Không coi các bản sao là mẫu độc lập.
- Manifest ghi 73 quan hệ nguồn -> PNG ASF: 71 output có thật, hai output failed.
  Những phiên bản dẫn xuất không được chia sang vai trò khác nguồn.
- Source JPEG/ASF, checksum PNG Unknown đã lưu và fixture hashes khớp metadata.
  Known PNG không có expected byte hash trong manifest; chỉ xác nhận nguồn
  được khai báo và inventory hiện tại, không chạy lại transform để attest nó.
- 14-5f.jpg và 15-5f.jpg đọc được. Lỗi cũ thuộc parser ASF, không chứng minh
  JPEG lỗi hoặc MTCNN thất bại. Cả hai vẫn nằm trong candidate raw evaluation.
- Không chạy visual/perceptual near-duplicate hoặc face identity comparison:
  **visual near-duplicates NOT VERIFIED**. Decoded-pixel equality không thay
  thế kiểm tra gần trùng, ảnh chụp gần nhau hoặc quan hệ cùng nguồn chưa có manifest.

Notebook 04 đã khảo sát cả 25 Known PNG trước split 3/2. Notebook 09 dùng
Unknown 09-12 calibration, 13-16 evaluation và chọn 0.48 theo protocol ASF,
Unknown-only, không có Known calibration độc lập. Notebook 10 và regression
tiếp tục dùng lại các ảnh. Phase 5B.2A/2B đã dùng cả năm JPEG IMM01 cho
correctness MTCNN. Không chạy lại hay diễn giải các kết quả đó thành hiệu năng mới.
Do exposure này, phần final của split mới cũng không phải test mù hoàn toàn.

## 4. Candidate split: common-five, identity-disjoint

Đề xuất thăm dò một fold primary_0, chỉ ảnh loại 1-5 cho cả 13 mã:
**65 JPEG**, tám ảnh loại 6 để reserve, không dùng để chọn threshold hoặc
báo primary result. Loại 6 thiếu ở năm người; bỏ theo quy tắc đồng đều trước
inference, không lọc theo score hoặc detector success.

| Stage | Known IDs | Unknown IDs | Enrollment | Genuine query | Unknown query |
|---|---|---|---:|---:|---:|
| Calibration | 01,05,06 | 07,08,09 | 9 | 6 | 15 |
| Final evaluation | 10,11,12 | 13,14,15,16 | 9 | 6 | 20 |

Mọi ID trong bảng là mã IMM. Gallery mỗi stage có ba người, ba ảnh/người
(loại 1,2,3). Known queries là loại 4,5; Unknown dùng cả loại 1-5.
Sorting theo mã IMM, numeric image type, source path; không cần random seed.
Hai stage có ID hoàn toàn khác nhau; trong mỗi stage Known và Unknown tách ID.
Ảnh/hash/pixel và ảnh nguồn chỉ có một role trong fold.

Vai trò mới không kế thừa máy móc Known/Unknown cũ: ví dụ IMM10 có thể là Known
trong gallery mới dù trước đây được dùng làm Unknown. Đây không đổi identity.
Không tái sử dụng registry/representatives alignment ASF.

**Split vẫn NOT FINALIZED:** assignment đã qua checks disjointness trong RAM,
nhưng cần người dùng duyệt reuse thăm dò, mục tiêu threshold và gate thực thi.
Không phát hành manifest đầy đủ đóng băng; JSON hiện chỉ có schema/example,
assignment rules và counts. Đánh giá mù cần một cohort mới, được duyệt riêng.

Có thể làm sensitivity analysis với 13 cyclic rotations của danh sách 13 ID
vào các slot 3 calibration-K / 3 calibration-U / 3 final-K / 4 final-U.
Trong từng fold vẫn identity-disjoint. Qua các fold có dùng lại người/ảnh:
không gộp như các quan sát độc lập, không chọn fold đẹp nhất hoặc chỉnh threshold
từ kết quả final. Optional folds chưa được tạo hoặc chạy.

## 5. Pipeline và failures cho phase sau

Tất cả JPEG phải đi qua MTCNN thật, size=160, align=True, adapter đã duyệt,
FaceNet và SQLite mới theo model_version hiện tại. Không dùng ASF để alignment,
không thay bằng PNG cũ, không tự chọn mặt lớn nhất.

- Enrollment cố định ba ảnh/người. Một ảnh lỗi -> enrollment atomic thất bại;
  primary fold không hợp lệ, dừng và báo. Không thay ảnh hoặc âm thầm giảm gallery.
- Query 0 face, >1 face, alignment/crop/embedding lỗi -> outcome failure.
  Không chọn một face trong ảnh single-subject và không tính failure thành
  unknown rejected đúng.
- Schema/model/version/IO error là lỗi thí nghiệm: dừng fold, không tính vào
  nhận nhầm hoặc unknown.
- Ghi stage failure và mẫu số planned/valid. Không xóa mẫu khỏi denominator E2E.
- Coverage đếm một face hợp lệ không phải detector precision/recall hay kiểm
  chứng localization đúng; muốn các chỉ số đó cần ground truth và protocol riêng.

Profile mặc định chỉ được fingerprint bằng AST constructor/config và
build_model_version; không load model hoặc inference ở phase này. Trước chạy
phải khớp SHA profile thực tế, freeze source manifest/hash và protocol version.
Gallery calibration và final tách riêng, SQLite tạm; không dùng baseline cũ.

Public recognition không trả embeddings/ranking. Evaluator phase sau có thể
dùng lower-level APIs đã duyệt để tính scores trong RAM và đối chiếu public
decisions, không sửa runtime hoặc ghi vector/BLOB vào báo cáo.

## 6. Định nghĩa và số cặp dự kiến

Genuine: query thuộc gallery identity, so với representative đúng ID.
Impostor: query so representative khác ID. Unknown: ID query không thuộc gallery.

Với gallery hoàn chỉnh G=3, nếu mọi query hợp lệ:
- Calibration: 6 genuine pairs; 12 known-query impostor pairs; 45 unknown-query
  impostor pairs, tổng 63 comparisons trên 21 query.
- Final: 6 genuine pairs; 12 known-query impostor pairs; 60 unknown-query
  impostor pairs, tổng 78 comparisons trên 26 query.

Các comparisons dùng chung query/representative không độc lập.
Không coi 78 comparisons là 78 người hoặc 78 query độc lập.

## 7. Metrics và mẫu số

K0,U0: số Known/Unknown query đã định trước. nK,nU: số query hợp lệ.
G: gallery identities. s(q,j): cosine theo matching API.
j*(q): top-1, hòa điểm theo person_id tăng dần; m(q)=max_j s(q,j).
A: Known đúng top-1 và m>=t. W: Known sai top-1 nhưng m>=t.
R: Known bị từ chối, m<t. UA/UR: Unknown chấp nhận/từ chối hợp lệ.
Mẫu số bằng 0 -> N/A, không tự chuyển thành 0.

### Pairwise verification — chỉ các cặp có embedding hợp lệ

| Metric | Tử số / mẫu số |
|---|---|
| Genuine distribution | nK scores s(q,true); báo count và phân vị/histogram sau này |
| Known impostor distribution | nK*(G-1) scores khác ID |
| Unknown impostor distribution | nU*G scores, tách khỏi Known impostor |
| TAR_pair(t) | count genuine score >= t / nK |
| FRR_pair(t) | count genuine score < t / nK = 1-TAR_pair |
| FAR_pair_K(t) | count accepted known-impostor pairs / [nK*(G-1)] |
| FAR_pair_U(t) | count accepted unknown-impostor pairs / [nU*G] |

Có thể thêm FAR_pair pooled với denominator nK*(G-1)+nU*G, nhưng phải ghi strata
và trọng số. Pair FAR không phải tần suất hệ thống nhận nhầm một Unknown query.

### Identification/open-set — conditional trên query hợp lệ

| Metric | Tử số / mẫu số |
|---|---|
| Known closed-set top-1 accuracy | count j*=true / nK, bỏ qua threshold |
| Known correct acceptance / TPIR / DIR rank1 | A/nK |
| FNIR rank1 | (W+R)/nK = 1-TPIR |
| Known wrong accepted identification | W/nK |
| Known rejection rate | R/nK; không gọi đây là FRR_pair |
| Unknown FPIR | UA/nU |
| Unknown rejection rate | UR/nU = 1-FPIR |
| Balanced open-set correct rate | 0.5*(A/nK + UR/nU) |

[NIST phân biệt FPIR/FNIR theo search](https://pages.nist.gov/frvt/html/frvt1N.html).
Protocol này dùng rank1 và quy tắc >= đã duyệt. UA đếm một query sai, không
đếm số cặp/candidates vượt ngưỡng; nhiều impostor pairs vượt ngưỡng vẫn chỉ
là một false-positive search.

### End-to-end và failure coverage

| Metric | Tử số / mẫu số |
|---|---|
| Known/Unknown usable extraction coverage | nK/K0 và nU/U0, tách hai nhóm |
| Zero-face / multiple-face rate | số ảnh của từng loại / planned images của nhóm |
| Alignment success conditional | số align hợp lệ / số ảnh có đúng một detection |
| Embedding success conditional | số embedding hợp lệ / số align hợp lệ |
| Enrollment subject success | số người đủ cả ba ảnh và lưu thành công / planned gallery IDs |
| TPIR_E2E / known correct acceptance | A/K0 |
| Known failure-or-miss | 1-A/K0, gồm wrong/reject/preprocessing failures |
| Unknown correct-rejection yield | UR/U0; failure không được cộng vào UR |
| FPIR_E2E | UA/U0, luôn kèm extraction coverage/failure counts |

Với Unknown, báo đủ UA/U0 + UR/U0 + failure/U0 = 1; không trình bày FPIR_E2E
thấp do detector không thấy mặt như bằng chứng rejection tốt.
Có thể thêm tổng đúng (A+UR)/(K0+U0), nhưng primary dùng balanced form để tránh
Unknown nhiều hơn Known chi phối kết quả.
Không có metric hiệu năng nào được tính trong phase này.

## 8. Thiết kế calibration — không thực thi

Primary đề xuất cho đồ án: tối đa
B_E2E(t)=0.5*(A_cal(t)/K0_cal + UR_cal(t)/U0_cal).
Hai nhóm có trọng số bằng nhau; wrong ID, rejection Known và query failure
không được tính đúng. Nếu không có gallery hoàn chỉnh hoặc không có ít nhất
một Known và Unknown query hợp lệ, ghi NOT_ESTIMABLE, không ép chọn threshold.

Candidate thresholds chỉ từ calibration top1 scores: {-1,1}, mỗi m, và
nextafter(m,+infinity) còn trong [-1,1]. Xử lý scores bằng nhau cùng lúc.
Không tạo threshold >1 để ép reject score=1. Giữ similarity>=threshold.
Các candidate có cùng B chọn threshold cao hơn, thiên về reject hơn.

Trước khi mở final scores: khóa exact float.hex threshold, objective, nguồn
calibration, counts/failures, profile SHA và split-manifest SHA. Final chỉ dùng
threshold đã khóa. Không sửa preprocessing, chọn lại ảnh hoặc threshold khi
thấy final errors. Không tự mang 0.48 từ ASF sang MTCNN.

Alternative: đặt trước empirical FPIR cap rồi tối đa TPIR, nếu người dùng chọn
chi phí false accept cao hơn. 15 calibration Unknown quá ít để chứng minh
một mức FAR nhỏ. Pairwise EER có thể là phân tích phụ, không thay mục tiêu
open-set hoặc chứng minh threshold tối ưu.

## 9. Thống kê và giới hạn

Chỉ có 13 identity clusters. Primary stage có 3 Known IDs; final có 4 Unknown IDs.
Genuine final chỉ 6 query; một lỗi làm tỷ lệ thay đổi khoảng 16.67 điểm phần trăm.
Unknown final có 20 query; một lỗi là 5 điểm phần trăm, không chứng minh low FAR.

CI phase sau phải xét cluster theo identity; pair/query dùng chung người và
gallery có tương quan. Có thể báo raw counts, per-identity rates và cluster
bootstrap thăm dò với seed cố định được khai báo trước; 3-4 clusters vẫn quá
ít cho CI vững. Không dùng binomial CI trên toàn bộ cặp như các lần thử độc lập.

Điều kiện ảnh theo nhãn IMM: frontal/biểu cảm, hướng đầu và ánh sáng có kiểm soát.
Primary không phủ kiểu 6, phụ kiện, tuổi, camera/video, motion blur hoặc bối
cảnh webcam thực tế. Cần cohort độc lập có consent và protocol riêng cho tuyên
bố generalization. Không thu thập/tải ảnh mới trong phase này.
Overlap với tập pretraining FaceNet chưa xác minh vì không có training manifest.

## 10. Manifest và readiness Phase 5B.3B

evaluation_phase5b3a_protocol.json chứa schema/example, explicit ID slots,
source rules, nominal counts, profile reference và 23 checks trong RAM.
Fields gồm protocol/dataset ID, root alias/path, source/pixel SHA, identity
reference + label gốc, fold/stage/role/Known status, model-version ID và rule ID.
Không chứa ảnh hoặc embeddings. Chưa tạo full evaluation manifest đóng băng.

Chỉ bắt đầu 5B.3B khi người dùng:
1. Duyệt mục đích exploratory reuse hoặc yêu cầu cohort mù mới ở phase riêng.
2. Duyệt split/common-five/reserve-type6 và có dùng sensitivity folds hay không.
3. Duyệt objective/cost tradeoff calibration.
4. Cho phép thực thi, sau khi freeze manifest/profile và kiểm tra nguồn lần nữa.

Không chạy lại 242 inference tests: chỉ thêm docs/reports; source/runtime/tests
và assets được đối chiếu hash, không có code inference thay đổi.
Không thêm audit utility lâu dài hoặc infrastructure tests không cần thiết.
23 assertions kiểm tra inventory, determinism, disjointness và negative controls
đều PASS; chỉ dùng dữ liệu trong RAM, không sửa dataset.

Các báo cáo:
- reports/dataset_phase5b3a_inventory.json
- reports/dataset_phase5b3a_leakage_audit.json
- reports/evaluation_phase5b3a_protocol.json
- reports/evaluation_phase5b3a_summary.json

Dừng tại thiết kế. Không chọn threshold, benchmark accuracy, tạo SQLite hoặc UI.
