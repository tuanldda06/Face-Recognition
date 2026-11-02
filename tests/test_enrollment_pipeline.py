
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT/"src"):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from integration import enrollment_pipeline as pipeline
from facenet_recognition import database as db
from facenet_recognition.enrollment import representative_from_embeddings


def unit(index=0):
    vector = np.zeros(512,np.float32)
    vector[index] = 1
    return vector


def fake_detector():
    points = np.array([[[30,40],[70,40],[50,60],[35,85],[65,85]]],np.float32)
    result = SimpleNamespace(boxes=np.array([[10,10,130,150]],np.float32),
        scores=np.array([.95],np.float32),landmarks=points)
    return SimpleNamespace(min_face_size=20,factor=.709,thresholds=(.6,.7,.7),
        batch_size=256,device="cpu",detect=Mock(return_value=result),
        extract_faces_rgb=Mock(return_value=np.full((1,160,160,3),100,np.uint8)))


class RegistryCase(unittest.TestCase):
    def setUp(self):
        parent=ROOT/".work/tmp"
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix="enrollment_phase_",dir=parent)
        self.folder=Path(self.temporary.name).resolve()
        self.addCleanup(self.safe_cleanup)
        self.path=self.folder/"registry.sqlite"
        pipeline.initialize_enrollment_registry(self.path)

    def safe_cleanup(self):
        if not self.folder.is_relative_to((ROOT/".work/tmp").resolve()) or not self.folder.name.startswith("enrollment_phase_"):
            raise AssertionError("Unsafe temporary cleanup target")
        self.temporary.cleanup()
        if self.folder.exists():
            raise AssertionError("Temporary registry not cleaned")

    def digest(self):
        return hashlib.sha256(self.path.read_bytes()).hexdigest()

    def count(self):
        return len(db.list_people(self.path))


class TestEnrollmentPipeline(RegistryCase):
    @classmethod
    def setUpClass(cls):
        cls.version=pipeline.build_model_version(fake_detector())

    def setUp(self):
        super().setUp()
        self.detector=fake_detector()
        self.images=[np.full((200,220,3),120,np.uint8)]
        self.net=object()

    def enroll(self,images=None,person_id="unit_only",version=None,path=None):
        return pipeline.enroll_person_from_images(person_id,
            self.images if images is None else images,self.detector,self.net,
            self.path if path is None else path,self.version if version is None else version)

    def test_n1_n2_n3_n5_and_existing_reader_policy(self):
        for count in (1,2,3,5):
            path=self.folder/("count_"+str(count)+".sqlite")
            pipeline.initialize_enrollment_registry(path)
            images=[np.full((200,220,3),i+30,np.uint8) for i in range(count)]
            embeddings=[unit(i) for i in range(count)]
            expected=representative_from_embeddings(np.stack(embeddings))
            with patch.object(pipeline,"extract_embedding",side_effect=embeddings):
                result=self.enroll(images,path=path)
            self.assertEqual(result["num_images"],count)
            self.assertEqual(result["status"],"enrolled")
            self.assertEqual(result["model_version"],self.version)
            self.assertTrue(np.array_equal(result["embedding"],expected))
            person=db.get_person(path,"unit_only",self.version)
            self.assertEqual(person["num_images"],count)
            self.assertTrue(np.array_equal(person["embedding"],expected))
            if count==3:
                self.assertEqual(len(db.load_registered_embeddings(path,self.version)),1)
            else:
                with self.assertRaisesRegex(ValueError,"3 ảnh/người"):
                    db.load_registered_embeddings(path,self.version)

    def test_tuple_supported_and_input_pixels_unchanged(self):
        image=self.images[0][:,::-1,:]
        before=image.copy()
        with patch.object(pipeline,"extract_embedding",return_value=unit()):
            self.enroll((image,))
        self.assertTrue(np.array_equal(image,before))
        self.detector.extract_faces_rgb.assert_called_once()
        self.assertEqual(self.detector.extract_faces_rgb.call_args.kwargs,{"size":160,"align":True})

    def test_empty_and_invalid_container(self):
        for images in ([],(),None,np.zeros((1,160,160,3),np.uint8)):
            with self.assertRaises((TypeError,ValueError)):
                pipeline.enroll_person_from_images("unit_only",images,self.detector,
                    self.net,self.path,self.version)
        self.detector.detect.assert_not_called()
        self.assertEqual(self.count(),0)

    def test_all_images_validated_before_detection(self):
        for bad in (None,[],np.zeros((0,10,3),np.uint8),np.zeros((20,20),np.uint8),
                    np.zeros((20,20,4),np.uint8),np.zeros((20,20,3),np.float32)):
            before=self.digest()
            with self.assertRaisesRegex((TypeError,ValueError),"image_index 1"):
                self.enroll([self.images[0],bad])
            self.detector.detect.assert_not_called()
            self.assertEqual(self.digest(),before)

    def test_invalid_person_id(self):
        for person_id in (None,""," "," bad","bad "):
            with self.assertRaises((TypeError,ValueError)):
                self.enroll(person_id=person_id)
        self.detector.detect.assert_not_called()

    def test_invalid_and_imm_model_versions_rejected(self):
        for version in (None,""," v1","v1",json.dumps({"alignment":"IMM_ASF_not_MTCNN"}),
                        json.dumps({"schema":"facenet-mtcnn-enrollment/1"})):
            with self.assertRaises((TypeError,ValueError)):
                pipeline.enroll_person_from_images("unit_only",self.images,
                    self.detector,self.net,self.path,version)
        self.detector.detect.assert_not_called()
        self.assertEqual(self.count(),0)

    def test_profile_changes_with_detector_and_binds_assets(self):
        profile=json.loads(self.version)
        self.assertEqual(profile["schema"],"facenet-mtcnn-enrollment/1")
        self.assertIn("MTCNN_5pt",profile["alignment"])
        self.assertEqual(profile["aligned_face"],[160,160,3])
        self.assertEqual(profile["assets_and_source_sha256"]["models/facenet.onnx"],
            hashlib.sha256((ROOT/"models/facenet.onnx").read_bytes()).hexdigest())
        self.detector.factor=.8
        changed=pipeline.build_model_version(self.detector)
        self.assertNotEqual(changed,self.version)
        with self.assertRaisesRegex(ValueError,"model_version"):
            self.enroll()
        self.detector.detect.assert_not_called()

    def test_profile_generation_does_not_load_models(self):
        import cv2
        with patch.object(cv2.dnn,"readNetFromONNX",side_effect=AssertionError("No model loading")):
            self.assertEqual(pipeline.build_model_version(self.detector),self.version)

    def test_missing_registry_not_implicitly_initialized(self):
        path=self.folder/"missing.sqlite"
        with patch.object(pipeline.db,"initialize_database") as initialize:
            with self.assertRaises(FileNotFoundError):
                self.enroll(path=path)
            initialize.assert_not_called()
        self.assertFalse(path.exists())
        self.detector.detect.assert_not_called()

    def test_explicit_initialization_never_overwrites(self):
        before=self.digest()
        with self.assertRaises(FileExistsError):
            pipeline.initialize_enrollment_registry(self.path)
        self.assertEqual(self.digest(),before)

    def test_zero_faces_and_multiple_faces_are_errors(self):
        for count in (0,2):
            result=SimpleNamespace(boxes=np.tile(np.array([[1,1,50,50]],np.float32),(count,1)))
            self.detector.detect.return_value=result
            before=self.digest()
            with self.assertRaisesRegex(ValueError,"image_index 0:.*đúng 1 face"):
                self.enroll()
            self.assertEqual(self.digest(),before)
        self.detector.extract_faces_rgb.assert_not_called()

    def test_malformed_detection_rejected(self):
        for result in (SimpleNamespace(),SimpleNamespace(boxes=np.zeros((1,5)))):
            self.detector.detect.return_value=result
            with self.assertRaises((AttributeError,ValueError)):
                self.enroll()
        self.assertEqual(self.count(),0)

    def test_alignment_failure_no_partial_write(self):
        before=self.digest()
        self.detector.extract_faces_rgb.side_effect=[np.zeros((1,160,160,3),np.uint8),ValueError("alignment failed")]
        with patch.object(pipeline,"extract_embedding",return_value=unit()):
            with self.assertRaisesRegex(ValueError,"image_index 1: alignment failed"):
                self.enroll(self.images*2)
        self.assertEqual(self.digest(),before)

    def test_adapter_failure_and_bad_output_count(self):
        before=self.digest()
        for value in ([],[{},{}]):
            with patch.object(pipeline,"mtcnn_to_facenet_detections",return_value=value):
                with self.assertRaisesRegex(ValueError,"Adapter"):
                    self.enroll()
        with patch.object(pipeline,"mtcnn_to_facenet_detections",side_effect=ValueError("adapter failed")):
            with self.assertRaisesRegex(ValueError,"image_index 0: adapter failed"):
                self.enroll()
        self.assertEqual(self.digest(),before)

    def test_last_inference_failure_no_write(self):
        before=self.digest()
        with patch.object(pipeline,"extract_embedding",side_effect=[unit(),RuntimeError("inference failed")]):
            with self.assertRaisesRegex(RuntimeError,"image_index 1: inference failed"):
                self.enroll(self.images*2)
        self.assertEqual(self.digest(),before)
        self.assertEqual(self.count(),0)

    def test_invalid_embedding_no_write(self):
        nan=unit();nan[7]=np.nan
        for vector in (np.zeros(511,np.float32),unit().astype(np.float64),unit()*2,nan):
            before=self.digest()
            with patch.object(pipeline,"extract_embedding",return_value=vector):
                with self.assertRaises((TypeError,ValueError)):
                    self.enroll()
            self.assertEqual(self.digest(),before)

    def test_float64_mean_and_l2_preserved(self):
        middle=unit(1);middle[0]=np.float32(2**-25)
        vectors=np.stack([unit(),middle,-unit()])
        expected=representative_from_embeddings(vectors)
        with patch.object(pipeline,"extract_embedding",side_effect=list(vectors)):
            result=self.enroll(self.images*3)
        self.assertTrue(np.array_equal(result["embedding"],expected))
        self.assertGreater(float(result["embedding"][0]),0)

    def test_zero_mean_or_bad_representative_does_not_write(self):
        before=self.digest()
        with patch.object(pipeline,"extract_embedding",side_effect=[unit(),-unit()]):
            with self.assertRaises(ValueError):
                self.enroll(self.images*2)
        with patch.object(pipeline,"extract_embedding",return_value=unit()), \
                patch.object(pipeline,"representative_from_embeddings",return_value=np.zeros(512,np.float32)):
            with self.assertRaises(ValueError):
                self.enroll()
        self.assertEqual(self.digest(),before)

    def test_duplicate_person_id_does_not_update(self):
        with patch.object(pipeline,"extract_embedding",return_value=unit()):
            self.enroll()
        before=self.digest()
        self.detector.detect.reset_mock()
        with patch.object(pipeline.db,"update_person",side_effect=AssertionError("No update")):
            with self.assertRaisesRegex(ValueError,"đã tồn tại"):
                self.enroll()
        self.detector.detect.assert_not_called()
        self.assertEqual(self.digest(),before)

    def test_registry_version_mismatch_any_record(self):
        db.add_person(self.path,"same",unit(),1,self.version)
        db.add_person(self.path,"different",unit(1),3,"IMM_ASF_not_MTCNN")
        before=self.digest()
        with self.assertRaisesRegex(ValueError,"model_version"):
            self.enroll()
        self.detector.detect.assert_not_called()
        self.assertEqual(self.digest(),before)

    def test_corrupt_checksum_rejected(self):
        db.add_person(self.path,"old",unit(),1,self.version)
        with closing(db.connect_database(self.path)) as connection:
            with connection:
                connection.execute("UPDATE FaceDatabase SET embedding_sha256=?",("0"*64,))
        before=self.digest()
        with self.assertRaisesRegex(ValueError,"Checksum"):
            self.enroll()
        self.assertEqual(self.digest(),before)

    def test_invalid_schema_not_migrated(self):
        path=self.folder/"wrong.sqlite"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE wrong (value TEXT)")
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            self.enroll(path=path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_sqlite_write_error_propagates(self):
        before=self.digest()
        with patch.object(pipeline,"extract_embedding",return_value=unit()), \
                patch.object(pipeline.db,"add_person",side_effect=sqlite3.OperationalError("write failed")):
            with self.assertRaisesRegex(sqlite3.OperationalError,"write failed"):
                self.enroll()
        self.assertEqual(self.digest(),before)

    def test_real_transaction_rolls_back_after_insert_failure(self):
        class FailAfterInsert(sqlite3.Connection):
            def execute(self,sql,parameters=()):
                cursor=super().execute(sql,parameters)
                if sql.lstrip().upper().startswith("INSERT"):
                    raise sqlite3.OperationalError("forced failure after INSERT")
                return cursor
        original=db.connect_database
        def connect(path,*,readonly=False):
            if readonly:
                return original(path,readonly=True)
            self.assertEqual(Path(path),self.path)
            connection=sqlite3.connect(self.path.as_uri()+"?mode=rw",uri=True,factory=FailAfterInsert)
            db.check_schema(connection)
            return connection
        before=self.digest()
        with patch.object(pipeline,"extract_embedding",return_value=unit()), \
                patch.object(db,"connect_database",side_effect=connect):
            with self.assertRaisesRegex(sqlite3.OperationalError,"forced failure"):
                self.enroll()
        self.assertEqual(self.count(),0)
        self.assertEqual(self.digest(),before)

    def test_no_original_images_saved_or_returned(self):
        with patch.object(pipeline,"extract_embedding",return_value=unit()):
            result=self.enroll()
        self.assertEqual(set(result),{"person_id","num_images","model_version","embedding","status"})
        with closing(db.connect_database(self.path,readonly=True)) as connection:
            columns=[row[1] for row in connection.execute("PRAGMA table_info(FaceDatabase)")]
            length=connection.execute("SELECT length(embedding) FROM FaceDatabase").fetchone()[0]
        self.assertEqual(columns,["person_id","embedding","num_images","model_version","embedding_sha256"])
        self.assertEqual(length,2048)

    def test_source_baseline_and_outside_paths_rejected(self):
        baseline=Path("D:/Facenet-Recognition/data/face_database_test.db")
        before=hashlib.sha256(baseline.read_bytes()).hexdigest()
        for path in (baseline,Path("D:/MTCNN/new_registry.sqlite"),Path("D:/outside_phase_registry.sqlite")):
            with self.assertRaises(PermissionError):
                self.enroll(path=path)
        self.assertEqual(hashlib.sha256(baseline.read_bytes()).hexdigest(),before)

    def test_hardlink_registry_rejected(self):
        alias=self.folder/"alias.sqlite"
        os.link(self.path,alias)
        with self.assertRaisesRegex(PermissionError,"hardlink"):
            self.enroll(path=alias)

    def test_import_does_not_access_assets_or_connect_sqlite(self):
        code=textwrap.dedent("""
            import os,sys
            sys.dont_write_bytecode=True
            def audit(event,args):
                if event=="open":
                    mode,flags=args[1:3]
                    if str(args[0]).lower().endswith((".onnx",".pt",".db",".sqlite",".png",".jpg",".ipynb")):
                        raise AssertionError("Import accessed asset")
                    if ((isinstance(mode,str) and any(c in mode for c in "wax+")) or
                        (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))):
                        raise AssertionError("Import wrote file")
                if event in ("sqlite3.connect","os.mkdir","socket.connect"):
                    raise AssertionError("Import I/O")
            sys.addaudithook(audit)
        """)+f"\nsys.path[:0]={[str(ROOT),str(ROOT/'src')]!r}\n"+textwrap.dedent("""
            import integration.enrollment_pipeline
            assert "torch" not in sys.modules
            assert "MTCNN.mtcnn" not in sys.modules
            print("ENROLLMENT_IMPORT_OK")
        """)
        result=subprocess.run([sys.executable,"-B","-c",code],cwd=ROOT/".work/tmp",
            capture_output=True,text=True,encoding="utf-8",
            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"},timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("ENROLLMENT_IMPORT_OK",result.stdout)


class TestRealEnrollment(RegistryCase):
    @classmethod
    def setUpClass(cls):
        from MTCNN.mtcnn import MTCNN
        from facenet_recognition.facenet_model import load_facenet_model
        cls.detector=MTCNN(device="cpu")
        cls.net=load_facenet_model(ROOT/"models/facenet.onnx")
        cls.version=pipeline.build_model_version(cls.detector)

    def test_real_single_fixture_enrollment_and_readback(self):
        from MTCNN.preprocessing.image_processing import read_image
        image=read_image(ROOT/"tests/fixtures/mtcnn_single.png")
        before=image.copy()
        result=pipeline.enroll_person_from_images("fixture_single",[image],
            self.detector,self.net,self.path,self.version)
        person=db.get_person(self.path,"fixture_single",self.version)
        vector=result["embedding"]
        self.assertEqual(result["num_images"],1)
        self.assertEqual(person["num_images"],1)
        self.assertEqual(person["model_version"],self.version)
        self.assertEqual(vector.shape,(512,))
        self.assertEqual(vector.dtype,np.float32)
        self.assertTrue(np.isfinite(vector).all())
        self.assertTrue(np.array_equal(person["embedding"],vector))
        self.assertTrue(np.array_equal(image,before))
        with closing(db.connect_database(self.path,readonly=True)) as connection:
            blob,checksum=connection.execute("SELECT embedding,embedding_sha256 FROM FaceDatabase").fetchone()
        self.assertEqual(len(blob),2048)
        self.assertEqual(hashlib.sha256(blob).hexdigest(),checksum)
        self.assertTrue(blob==db.embedding_to_blob(vector))
        # Đối chiếu chuỗi API có sẵn, không mock model hoặc ghi vào baseline.
        detected=self.detector.detect(image)
        self.assertEqual(len(detected.boxes),1)
        aligned=self.detector.extract_faces_rgb(image,detected,size=160,align=True)
        direct=pipeline.extract_embedding(aligned[0],self.net)
        expected=representative_from_embeddings(direct[None])
        self.assertTrue(np.allclose(vector,expected,atol=1e-6,rtol=1e-5))
        print("REAL_ENROLLMENT "+json.dumps({"images":1,"detections_per_image":1,
            "representative_shape":[512],"dtype":str(vector.dtype),
            "max_l2_norm_error":abs(float(np.linalg.norm(vector.astype(np.float64)))-1),
            "readback_max_abs_difference":float(np.max(np.abs(person["embedding"]-vector))),
            "reference_max_abs_difference":float(np.max(np.abs(vector-expected))),
            "blob_bytes":len(blob),"checksum_valid":True,"original_image_unchanged":True,
            "registry_is_temporary":True,"scope":"single_image_case"}))

    def test_real_three_independent_imm_images_same_dataset_identity(self):
        from MTCNN.preprocessing.image_processing import read_image
        source_root=Path("D:/Facenet-Recognition")
        manifest=json.loads((source_root/"data/processed_faces_imm/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset"],"IMM Face Database")
        records=sorted((r for r in manifest["records"] if r["person_id"]=="person_01" and r["status"]=="saved"),
                       key=lambda r:r["source_image"])[:3]
        self.assertEqual(len(records),3)
        self.assertEqual(len({r["source_sha256"] for r in records}),3)
        images=[]
        for record in records:
            path=source_root/record["source_image"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record["source_sha256"])
            images.append(read_image(path))  # JPEG gốc; không dùng PNG ASF-aligned.
        originals=[image.copy() for image in images]
        result=pipeline.enroll_person_from_images("imm_person_01_fixture",images,
            self.detector,self.net,self.path,self.version)
        person=db.get_person(self.path,"imm_person_01_fixture",self.version)
        self.assertEqual(person["num_images"],3)
        self.assertTrue(np.array_equal(person["embedding"],result["embedding"]))
        registered=db.load_registered_embeddings(self.path,self.version)
        self.assertEqual(len(registered),1)
        self.assertTrue(np.array_equal(registered["imm_person_01_fixture"],result["embedding"]))
        direct=[]
        for image,original in zip(images,originals):
            self.assertTrue(np.array_equal(image,original))
            detected=self.detector.detect(image)
            self.assertEqual(len(detected.boxes),1)
            aligned=self.detector.extract_faces_rgb(image,detected,size=160,align=True)
            direct.append(pipeline.extract_embedding(aligned[0],self.net))
        expected=representative_from_embeddings(np.stack(direct))
        self.assertTrue(np.allclose(result["embedding"],expected,atol=1e-6,rtol=1e-5))
        with closing(db.connect_database(self.path,readonly=True)) as connection:
            blob,checksum=connection.execute("SELECT embedding,embedding_sha256 FROM FaceDatabase").fetchone()
        self.assertEqual(len(blob),2048)
        self.assertEqual(hashlib.sha256(blob).hexdigest(),checksum)
        print("REAL_MULTI_ENROLLMENT "+json.dumps({"images":3,"independent_source_images":3,
            "identity_evidence":"IMM manifest person_01; distinct verified JPEG hashes",
            "alignment":"MTCNN on original JPEG; not ASF",
            "detections_per_image":[1,1,1],"num_images_readback":3,
            "readback_max_abs_difference":float(np.max(np.abs(person["embedding"]-result["embedding"]))),
            "reference_max_abs_difference":float(np.max(np.abs(result["embedding"]-expected))),
            "max_l2_norm_error":abs(float(np.linalg.norm(result["embedding"].astype(np.float64)))-1),
            "checksum_valid":True,"registry_is_temporary":True}))


    def test_real_multi_face_image_rejected_without_registration(self):
        from MTCNN.preprocessing.image_processing import read_image
        image=read_image(ROOT/"tests/fixtures/mtcnn_multi.jpg")
        before=self.digest()
        with self.assertRaisesRegex(ValueError,"đúng 1 face"):
            pipeline.enroll_person_from_images("not_registered",[image],
                self.detector,self.net,self.path,self.version)
        self.assertEqual(self.count(),0)
        self.assertEqual(self.digest(),before)

    def test_real_no_face_image_rejected_without_registration(self):
        before=self.digest()
        with self.assertRaisesRegex(ValueError,"đúng 1 face"):
            pipeline.enroll_person_from_images("not_registered",[np.zeros((240,320,3),np.uint8)],
                self.detector,self.net,self.path,self.version)
        self.assertEqual(self.count(),0)
        self.assertEqual(self.digest(),before)


if __name__=="__main__":
    unittest.main()
