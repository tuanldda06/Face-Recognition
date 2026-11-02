# Face Recognition — MTCNN + FaceNet

## 1. Mục đích

Backend nhận diện khuôn mặt trên ảnh tĩnh bằng MTCNN, FaceNet ONNX và SQLite.
Ứng dụng/UI gọi API Python; project chưa có CLI nhận diện, webcam, server hoặc GUI.
Entry point chính là `integration.recognition_runtime.recognize_image()`.

## 2. Kiến trúc và công nghệ

`Ảnh RGB gốc → MTCNN → alignment 160×160 → FaceNet → embedding512/L2 → cosine/top-1 → Known/Unknown`.

Python3.13, NumPy, OpenCV DNN, Torch CPU, Pillow và sqlite3.
Enrollment lưu representative của một danh tính; recognition chỉ đọc registry.
Models được caller nạp một lần và tái sử dụng. Không tải model/dataset khi gọi API.

## 3. Cấu trúc

~~~text
Face-Recognition-System/
  README.md
  UI_HANDOFF.md
  EVALUATION_PROTOCOL.md          # scientific evidence,giữ nguyên
  requirements-integration.txt
  requirements-test.txt
  requirements-lock.txt
  COPY_MANIFEST.json
  .venv/
  integration/
  src/facenet_recognition/
  MTCNN/                         # code + weights P/R/O-Net
  models/facenet.onnx
  scripts/run_verification.py
  tests/                         # năm regression modules + fixtures
  reports/                       # scientific JSON + evidence_index.json
  FINAL_HANDOFF_REPORT.md
  final_handoff_manifest.json
~~~

Main vẫn giữ approval controls trong reports/phase5c1 và phase5c2.
Bản handoff riêng không kèm controls,archive,backup,FEI hoặc registry thật.
Registry mới do caller lựa chọn; không có database vận hành được tạo sẵn.

## 4. Vai trò code và notebook

| File | Vai trò |
|---|---|
| integration/recognition_runtime.py | V2 default,V1 rollback/override;chuyển tiếp pipeline |
| integration/recognition_pipeline.py | Điều phối RGB→MTCNN/alignment→FaceNet→SQLite→recognition |
| integration/enrollment_pipeline.py | Canonical model_version,tạo registry tường minh,enrollment |
| integration/mtcnn_adapter.py | Validate/chuyển output MTCNN sang contract FaceNet,giữ thứ tự |
| src/facenet_recognition/contracts.py | Validate aligned RGB và detection metadata |
| src/facenet_recognition/facenet_model.py | Load ONNX CPU,preprocess,embedding/L2 |
| src/facenet_recognition/enrollment.py | Mean float64→L2→representative float32 |
| src/facenet_recognition/database.py | Schema,codec/checksum,metadata và SQLite CRUD |
| src/facenet_recognition/matching.py | Cosine,ranking,top-1,tie-break và threshold |
| src/facenet_recognition/recognition.py | Core recognition cho aligned face/batch |
| MTCNN/ | Detector,pyramid,bounding boxes,P/R/O-Net,weights loader,alignment/image utilities |
| scripts/run_verification.py | Bốn suite hiện có với source-write/network guards |

Giữ nguyên package __init__.py. Main không có notebook runtime.
Notebook gốc ở source FaceNet:01_facenet_model,02_embedding_extraction,
05_face_enrollment,06_face_database,07_face_matching,08_unknown_detection,
09_open_set_evaluation và10_mtcnn_facenet_interface.ipynb.
Các notebook nghiên cứu/IMM preparation vẫn ở source/archive; không cần Jupyter để gọi runtime.

## 5. Môi trường

Đã dùng Windows x64,CPython3.13.14,NumPy2.1.3,opencv-python4.10.0.84
(cv2 4.10.0),Pillow12.2.0,Torch2.12.1+cpu vàpytest9.2.0.
requirements-lock.txt khóa19 wheels bằng URL+SHA cho CPython3.13 Windows x64.
sqlite3 thuộc standard library; không dùng ONNX Runtime hoặc TensorFlow.

.venv đi kèm là bản sao môi trường hiện có,không phải portable environment.
pyvenv.cfg còn cần Python hệ thống C:/Program Files/Python313.
Không đổi phiên bản runtime tùy tiện:versions là thành phần của model_version.

## 6. Thiết lập Windows CPU

Mở terminal tại project root (main hoặc handoff). Không cần activation:
~~~powershell
$projectRoot = (Get-Location).Path
$env:PYTHONPATH = "$projectRoot;$projectRoot\src"
$env:PYTHONNOUSERSITE = '1'
$env:PYTHONDONTWRITEBYTECODE = '1'
New-Item -ItemType Directory -Force .work\tmp | Out-Null
$env:TEMP = Join-Path $projectRoot '.work\tmp'
$env:TMP = $env:TEMP
.\.venv\Scripts\python.exe -B -m pip check
~~~

Chỉ khi .venv chưa tồn tại,mới tạo bằng CPython3.13 đã xác minh và cài lock:
~~~powershell
& 'C:\Program Files\Python313\python.exe' -B -m venv .venv
.\.venv\Scripts\python.exe -B -m pip install --require-hashes --no-deps --no-cache-dir -r requirements-lock.txt
~~~
Không ghi đè environment đang có. Fresh-machine install chưa được kiểm chứng đầy đủ.
VS Code:Python:Select Interpreter→Enter interpreter path→.venv/Scripts/python.exe.

## 7. Nạp models một lần

~~~python
# example:bootstrap
from pathlib import Path
from MTCNN.mtcnn import MTCNN
from facenet_recognition.facenet_model import load_facenet_model
from integration.enrollment_pipeline import build_model_version

root = Path.cwd().resolve()  # terminal đang ở project root
detector = MTCNN(device="cpu")
facenet_net = load_facenet_model(root / "models/facenet.onnx")
model_version = build_model_version(detector)
~~~

Loader FaceNet dùng OpenCV DNN backend/target CPU.
MTCNN đọc weights bundled trong MTCNN/weights;không dùng models tùy biến
rồi gắn model_version của bộ weights bundled. cv2.Net không tự attest weights trong RAM.

## 8. Ảnh đầu vào và alignment

~~~python
# example:read_rgb
import cv2

def read_rgb(path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(str(path))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
~~~

recognize_image nhận ảnh RGB uint8(H,W,3),H/W>0,trước detection/alignment.
Không tự đoán BGR/RGB;caller chịu trách nhiệm chuyển màu.
Pipeline gọi detector.detect rồi extract_faces_rgb(size=160,align=True)
và adapter hiện có. Alignment dùng MTCNN5-point similarity,template ArcFace112 scale160.
Không tự thêm resize192/center-crop,normalize hoặc alignment lần thứ hai ở UI.

## 9. Face embedding

Core FaceNet nhận aligned RGB uint8(160,160,3).
Preprocess float32:`(pixel-127.5)/128.0`,layout NCHW,batch1.
Output512D được L2 normalize và trả float32(512,),finite,norm gần1.
Không dùng per-image mean/std;không chuẩn hóa pixel hai lần.

Khi cần gọi core trực tiếp,sau khi đã có aligned face hợp lệ:
~~~python
# example:embedding
from facenet_recognition.facenet_model import extract_embedding
embedding = extract_embedding(aligned_face_rgb, facenet_net)
~~~
UI thông thường chỉ gọi orchestration;không cần tự quản lý embeddings.

## 10. Enrollment và SQLite

Caller cung cấp person_id,images_rgb cùng người vàregistry_path tuyệt đối.
N>0;mỗi ảnh phải có đúng một face hợp lệ. Không tự xác nhận danh tính của ảnh
hoặc chọn mặt lớn nhất từ ảnh nhóm. Chỉ dùng ảnh có quyền sử dụng.

~~~python
# example:new_registry
from integration.enrollment_pipeline import initialize_enrollment_registry
# Chỉ gọi khi người dùng chủ động tạo registry mới;parent directory phải tồn tại.
initialize_enrollment_registry(registry_path)
~~~
File đã có→FileExistsError;không overwrite/migrate/initialize ngầm.

~~~python
# example:enroll
from integration.enrollment_pipeline import enroll_person_from_images
enrolled = enroll_person_from_images(
    person_id, images_rgb, detector, facenet_net, registry_path, model_version
)
~~~

Registry phải ở project tích hợp hoặc vùng tạm được policy cho phép;
không ghi vào hai source projects/baseline hoặc qua symlink/hardlink.
Schema/codec2048-byte float32/checksum/num_images/model_version giữ nguyên.
Mọi record phải cùng canonical model_version;duplicate ID bị từ chối,không upsert.
Một ảnh lỗi→không enrollment từng phần;transaction rollback của database API giữ nguyên.
Kết quả có status,person_id,num_images,model_version vàrepresentative embedding;
không log ảnh,vectors hoặc BLOB. Chưa bảo đảm concurrent writers.

Reader orchestration hỗ trợ mọi N>0.Reader legacy load_registered_embeddings()
còn yêu cầu3 ảnh/người;UI không dùng reader đó thay pipeline.

## 11. Recognition

~~~python
# example:recognize
from integration.recognition_runtime import recognize_image
results = recognize_image(
    image_rgb, detector, facenet_net, registry_path, model_version
)
~~~

Registry phải tồn tại và tương thích.Preflight ảnh/threshold/version/DB xảy ra
trước detection;nên DB thiếu/hỏng vẫn gây lỗi dù ảnh không có mặt.
Pipeline giữ face order,không bỏ detections hoặc fallback khi alignment/model lỗi.
Recognition chỉ đọc SQLite;không tự initialize/enroll/update registry.
Giữ registry ổn định trong lúc đọc;list/get chưa là snapshot đa-process thống nhất.

## 12. V2 mặc định và V1 rollback

| Profile | Decimal | Exact binary64 |
|---|---:|---|
| v2 (default) |0.665513450766159|0x1.54be2dd42d71dp-1|
| v1 (rollback)|0.925533688089963|0x1.d9df8d3362a3cp-1|

~~~python
# example:rollback
rollback_results = recognize_image(
    image_rgb, detector, facenet_net, registry_path, model_version,
    threshold_profile="v1",
)
~~~

Rollback chỉ áp dụng call đó;call tiếp theo bỏ keyword vẫn dùng V2.
Override threshold tường minh được API hỗ trợ:số hữu hạn[-1,1],không bool/None/chuỗi.
Profile vẫn phải là"v1"/"v2" dù có override;None không có nghĩa default.
Cosine/top-1 giữ nguyên;tie-break theo person_id;chấp nhận khi similarity>=threshold.
Lower APIs như recognize_faces_from_image vẫn yêu cầu threshold tường minh.
Không tự điều chỉnh ngưỡng theo ảnh hoặc tối ưu lại trên FEI.

## 13. Contract dành cho UI

Xem [UI_HANDOFF.md](UI_HANDOFF.md) cho integration ngắn.
Caller sở hữu detector/network đã nạp,ảnh RGB,registry tuyệt đối vàcanonical version.
Tái sử dụng models;serialize model access/registry writes vì thread-safety chưa được bảo đảm.

Output:list[dict] với face_index,bbox(xyxy half-open),confidence,status,
person_id,best_match_id,similarity,threshold,model_version vàlandmarks khi có.
status là"recognized"/"unknown";Unknown có person_id=None.
best_match_id là candidate,kể cả bị từ chối;không hiển thị candidate như identity đã xác nhận.
similarity không phải xác suất;confidence là detection confidence.
Registry rỗng hợp lệ→Unknown,score/candidate=None;vẫn validate embedding.
Không face→[] sau preflight hợp lệ.Không trả ảnh,vectors hoặc toàn bộ ranking.

TypeError/ValueError cho input/threshold/metadata;FileNotFoundError cho assets/registry thiếu.
Lỗi SQLite/model/detection/alignment được truyền lên,có stage/face index khi API cung cấp.
Không chuyển lỗi infrastructure thành Unknown;không trả batch một phần.
Không dùng reserved person_id="Unknown".

## 14. Tests và kiểm tra hệ thống

Sau setup PYTHONPATH ở mục6:
~~~powershell
.\.venv\Scripts\python.exe -B -X utf8 -m unittest discover -s tests -p "test_*.py" -v
.\.venv\Scripts\python.exe -B -X utf8 scripts\run_verification.py facenet
.\.venv\Scripts\python.exe -B -X utf8 scripts\run_verification.py mtcnn
.\.venv\Scripts\python.exe -B -X utf8 scripts\run_verification.py adapter
.\.venv\Scripts\python.exe -B -X utf8 scripts\run_verification.py embedding
~~~

Năm retained unittest modules kiểm adapter,embedding,enrollment,recognition vàruntime.
FaceNet source suite kiểm model/preprocess/matching/database;MTCNN source suite kiểm detector.
Một số tests đọc notebooks,fixtures vàIMM từ D:/Facenet-Recognition vàD:/MTCNN;
runner chặn ghi source vànetwork.SQLite ghi chỉ dùng temp-owned paths vàđược dispose.
Không gọi source-only tests là bằng chứng duy nhất cho main runtime.
Kết quả cuối,CPU smoke vàhandoff simulation:[FINAL_HANDOFF_REPORT.md](FINAL_HANDOFF_REPORT.md).

## 15. Provenance,scientific evidence vàgiới hạn

FaceNet ONNX từ Axelera,VoyagerSDKv1.9.0;artifact URL vàpublisher/local checksums
được giữ tại [reports/evidence_index.json](reports/evidence_index.json).
SHA-256 ONNX:f6688ba6c95465fac18df53fdf8e094048fa3f1dd2c09f1cc6dd734a1c8dedf8.
VGGFace2/architecture là khai báo đã ghi của nhà phát hành,không suy ra từ512D.
Giữ notices/nguồn giấy phép gốc;không suy diễn quyền thương mại hoặc tái phân phối dataset.

Frozen FEI evidence:45 ảnh (15 enrollment,15 Known query,15 Unknown query),5 Known identities.
V1 Known10/15,V2 Known15/15;Unknown rejection15/15 vàwrong-ID0/15 cho cả hai.
Đây là LIMITED_INDEPENDENT_EVIDENCE;không production-grade accuracy.
Smoke dùng outcomes đã biết là correctness regression,không evaluation độc lập mới.

Giữ nguyên [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md) và23 scientific/environment JSON.
Research scripts/tests cùng dependency closure được bảo tồn ngoài project;
evidence index ghi original paths/SHA/archive locator/restore instructions.
Historical snapshots không bị viết lại theo source hashes sau cleanup.
Archive/backup không kèm FEI payload;ảnh FEI ngoài project chỉ được tham chiếu.
Scientific reproduction có absolute source paths;handoff runtime đã thử trên Windows hiện tại.
Fresh-machine/portable execution,multi-writer,webcam,UI performance vàpopulation accuracy chưa được kiểm chứng.
