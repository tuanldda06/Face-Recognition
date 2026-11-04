
from contextlib import closing, contextmanager, ExitStack
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

ROOT=Path(__file__).resolve().parents[2]
for directory in (ROOT,):
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))

from integration import recognition_pipeline as pipeline
from integration.enrollment_pipeline import (
    build_model_version, initialize_enrollment_registry, enroll_person_from_images,
)
from FaceNet import database as db
from FaceNet.matching import detect_unknown
from FaceNet.facenet_model import extract_embedding


def unit(index=0):
    vector=np.zeros(512,np.float32)
    vector[index]=1
    return vector


def fake_detector(count=1):
    boxes=np.array([[10+i*30,20,100+i*30,140] for i in range(count)],np.float32).reshape(count,4)
    points=np.array([[30,40],[60,40],[45,55],[35,80],[55,80]],np.float32)
    landmarks=np.stack([points+[i*30,0] for i in range(count)]).astype(np.float32) if count else np.empty((0,5,2),np.float32)
    detected=SimpleNamespace(boxes=boxes,scores=np.linspace(.7,.95,count,dtype=np.float32),landmarks=landmarks)
    faces=np.stack([np.full((160,160,3),40+i*30,np.uint8) for i in range(count)]) if count else np.empty((0,160,160,3),np.uint8)
    return SimpleNamespace(min_face_size=20,factor=.709,thresholds=(.6,.7,.7),
        batch_size=256,device="cpu",detect=Mock(return_value=detected),
        extract_faces_rgb=Mock(return_value=faces))


class RegistryCase(unittest.TestCase):
    def setUp(self):
        parent=Path(tempfile.gettempdir())
        parent.mkdir(parents=True,exist_ok=True)
        self.temporary=tempfile.TemporaryDirectory(prefix="recognition_phase_",dir=parent)
        self.folder=Path(self.temporary.name).resolve()
        self.addCleanup(self.safe_cleanup)
        self.path=self.folder/"registry.sqlite"
        initialize_enrollment_registry(self.path)

    def safe_cleanup(self):
        if not self.folder.is_relative_to((Path(tempfile.gettempdir())).resolve()) or not self.folder.name.startswith("recognition_phase_"):
            raise AssertionError("Unsafe cleanup path")
        self.temporary.cleanup()
        if self.folder.exists():
            raise AssertionError("Temporary registry not cleaned")

    @contextmanager
    def readonly_check(self):
        before=hashlib.sha256(self.path.read_bytes()).hexdigest()
        before_sidecars=set(self.folder.glob("registry.sqlite-*"))
        original=db.connect_database
        calls=[];statements=[]
        def connect(path,*,readonly=False):
            self.assertTrue(readonly,"Recognition attempted a writable connection")
            connection=original(path,readonly=readonly)
            self.assertEqual(connection.execute("PRAGMA query_only").fetchone(),(1,))
            connection.set_trace_callback(statements.append)
            calls.append(True)
            return connection
        with patch.object(db,"connect_database",side_effect=connect):
            yield
        self.assertTrue(calls)
        self.assertTrue(all(sql.lstrip().upper().startswith(("SELECT","PRAGMA")) for sql in statements))
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),before)
        self.assertEqual(set(self.folder.glob("registry.sqlite-*")),before_sidecars)


class TestRecognitionPipeline(RegistryCase):
    @classmethod
    def setUpClass(cls):
        cls.version=build_model_version(fake_detector())

    def setUp(self):
        super().setUp()
        self.detector=fake_detector()
        self.image=np.full((240,320,3),100,np.uint8)
        self.net=object()

    def seed(self,person_id="unit_a",vector=None,count=1,version=None):
        db.add_person(self.path,person_id,unit() if vector is None else vector,count,
                      self.version if version is None else version)

    def query(self,vectors=None,threshold=.5,path=None,version=None,image=None):
        values=[unit()] if vectors is None else vectors
        extract=Mock(side_effect=values)
        with patch.object(pipeline.core,"extract_embedding",extract), \
                patch.object(pipeline,"extract_embedding",extract):
            return pipeline.recognize_faces_from_image(
                self.image if image is None else image,self.detector,self.net,
                self.path if path is None else path,
                self.version if version is None else version,threshold)

    def test_single_known_and_output_contract(self):
        self.seed()
        with self.readonly_check():
            output=self.query()
        self.assertEqual(len(output),1)
        row=output[0]
        self.assertEqual(row["status"],"recognized")
        self.assertEqual(row["person_id"],"unit_a")
        self.assertEqual(row["best_match_id"],"unit_a")
        self.assertEqual(row["similarity"],1.0)
        self.assertEqual(row["face_index"],0)
        self.assertEqual(row["model_version"],self.version)
        self.assertEqual(set(row),{"face_index","bbox","confidence","status","person_id",
            "best_match_id","similarity","threshold","model_version","landmarks"})
        self.assertNotIn("embedding",row)
        self.assertNotIn("face_rgb",row)
        self.assertNotIn("ranking",row)

    def test_unknown_retains_candidate_but_person_id_is_none(self):
        self.seed()
        row=self.query([unit(1)])[0]
        self.assertEqual(row["status"],"unknown")
        self.assertIsNone(row["person_id"])
        self.assertEqual(row["best_match_id"],"unit_a")
        self.assertEqual(row["similarity"],0.0)

    def test_multiple_faces_and_deterministic_input_order(self):
        self.detector=fake_detector(3)
        self.seed("unit_a",unit(),1);self.seed("unit_b",unit(1),5)
        output=self.query([unit(1),unit(),unit(2)])
        self.assertEqual([r["face_index"] for r in output],[0,1,2])
        self.assertEqual([r["person_id"] for r in output],["unit_b","unit_a",None])
        self.assertEqual([r["status"] for r in output],["recognized","recognized","unknown"])
        for i,row in enumerate(output):
            self.assertEqual(row["bbox"],self.detector.detect.return_value.boxes[i].tolist())

    def test_empty_registry_unknown_with_no_fabricated_similarity(self):
        with self.readonly_check():
            row=self.query(threshold=-1)[0]
        self.assertEqual(row["status"],"unknown")
        self.assertIsNone(row["person_id"])
        self.assertIsNone(row["best_match_id"])
        self.assertIsNone(row["similarity"])
        self.assertEqual(len(db.list_people(self.path)),0)

    def test_empty_registry_still_extracts_and_validates_embedding(self):
        with patch.object(pipeline,"extract_embedding",side_effect=RuntimeError("failed")):
            with self.assertRaisesRegex(RuntimeError,"face_index 0"):
                pipeline.recognize_faces_from_image(self.image,self.detector,self.net,
                    self.path,self.version,.5)
        with self.assertRaises(ValueError):
            self.query([np.zeros(512,np.float32)])

    def test_no_face_returns_empty_without_inference(self):
        self.detector=fake_detector(0)
        with patch.object(pipeline,"extract_embedding") as extract, \
                patch.object(pipeline.core,"recognize_detected_faces") as core:
            result=pipeline.recognize_faces_from_image(self.image,self.detector,self.net,
                self.path,self.version,.5)
        self.assertEqual(result,[])
        extract.assert_not_called();core.assert_not_called()

    def test_no_face_does_not_hide_missing_or_corrupt_registry(self):
        self.detector=fake_detector(0)
        with self.assertRaises(FileNotFoundError):
            self.query(path=self.folder/"missing.sqlite")
        self.detector.detect.assert_not_called()
        self.seed()
        with closing(db.connect_database(self.path)) as connection:
            with connection:
                connection.execute("UPDATE FaceDatabase SET embedding_sha256=?",("0"*64,))
        with self.assertRaises(ValueError):
            self.query()
        self.detector.detect.assert_not_called()

    def test_n1_n2_n3_n5_registry_records_supported(self):
        for i,count in enumerate((1,2,3,5)):
            self.seed("unit_"+str(count),unit(i),count)
        for i,count in enumerate((1,2,3,5)):
            row=self.query([unit(i)])[0]
            self.assertEqual(row["person_id"],"unit_"+str(count))
            self.assertEqual(row["status"],"recognized")

    def test_threshold_equality_is_accepted(self):
        self.seed()
        self.assertEqual(self.query([unit(1)],threshold=0)[0]["status"],"recognized")
        above=float(np.nextafter(0.0,1.0))
        self.assertEqual(self.query([unit(1)],threshold=above)[0]["status"],"unknown")
        self.assertEqual(self.query(threshold=1)[0]["status"],"recognized")

    def test_similarity_ties_do_not_depend_on_registry_order(self):
        self.seed("z",unit(),5);self.seed("a",unit(),2)
        original=db.list_people
        with patch.object(db,"list_people",side_effect=lambda path:list(reversed(original(path)))):
            row=self.query()[0]
        self.assertEqual(row["person_id"],"a")
        self.assertEqual(row["best_match_id"],"a")

    def test_invalid_images_rejected_before_detection(self):
        for image in (None,[],np.zeros((0,2,3),np.uint8),np.zeros((20,20),np.uint8),
                      np.zeros((20,20,4),np.uint8),np.zeros((20,20,3),np.float32)):
            with self.assertRaises((TypeError,ValueError)):
                pipeline.recognize_faces_from_image(image,self.detector,self.net,self.path,self.version,.5)
        self.detector.detect.assert_not_called()

    def test_invalid_thresholds_rejected_before_detection(self):
        for threshold in (None,True,np.bool_(False),"0.5",np.nan,np.inf,-1.01,1.01):
            with self.assertRaises((TypeError,ValueError)):
                self.query(threshold=threshold)
        self.detector.detect.assert_not_called()

    def test_model_version_type_legacy_and_mismatch_rejected(self):
        for version in (None,""," v1","v1",json.dumps({"alignment":"IMM_ASF_not_MTCNN"}),
                        self.version+" "):
            with self.assertRaises((TypeError,ValueError)):
                pipeline.recognize_faces_from_image(self.image,self.detector,self.net,
                    self.path,version,.5)
        self.detector.detect.assert_not_called()

    def test_changed_detector_configuration_rejected(self):
        self.detector.factor=.8
        with self.assertRaisesRegex(ValueError,"model_version"):
            self.query()
        self.detector.detect.assert_not_called()

    def test_registry_version_mismatch_in_any_record(self):
        self.seed("good",unit(),1);self.seed("bad",unit(1),3,"IMM_ASF_not_MTCNN")
        with self.assertRaisesRegex(ValueError,"model_version"):
            self.query()
        self.detector.detect.assert_not_called()

    def test_missing_relative_and_invalid_sqlite_are_errors(self):
        with self.assertRaises(FileNotFoundError):
            self.query(path=self.folder/"missing.sqlite")
        with self.assertRaises(ValueError):
            self.query(path=Path("relative.sqlite"))
        path=self.folder/"not_sqlite"
        path.write_bytes(b"not a SQLite database")
        with self.assertRaises(sqlite3.DatabaseError):
            self.query(path=path)
        self.assertFalse((self.folder/"missing.sqlite").exists())

    def test_wrong_schema_not_migrated(self):
        path=self.folder/"wrong.sqlite"
        with closing(sqlite3.connect(path)) as connection:
            connection.execute("CREATE TABLE wrong (x TEXT)")
        before=hashlib.sha256(path.read_bytes()).hexdigest()
        with self.assertRaises(ValueError):
            self.query(path=path)
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),before)

    def test_checksum_corruption_not_skipped(self):
        self.seed()
        with closing(db.connect_database(self.path)) as connection:
            with connection:
                connection.execute("UPDATE FaceDatabase SET embedding_sha256=?",("0"*64,))
        before=hashlib.sha256(self.path.read_bytes()).hexdigest()
        with self.assertRaisesRegex(ValueError,"Checksum"):
            self.query()
        self.assertEqual(hashlib.sha256(self.path.read_bytes()).hexdigest(),before)
        self.detector.detect.assert_not_called()

    def test_invalid_stored_embedding_norm_or_nan_not_skipped(self):
        self.seed()
        for value in (0,np.nan,np.inf):
            vector=np.zeros(512,np.float32);vector[0]=value
            blob=vector.tobytes()
            with closing(db.connect_database(self.path)) as connection:
                with connection:
                    connection.execute("UPDATE FaceDatabase SET embedding=?,embedding_sha256=?",
                        (blob,hashlib.sha256(blob).hexdigest()))
            with self.assertRaises(ValueError):
                self.query()
        self.detector.detect.assert_not_called()

    def test_reserved_unknown_id_rejected(self):
        self.seed("Unknown")
        with self.assertRaisesRegex(ValueError,"dành riêng"):
            self.query()
        self.detector.detect.assert_not_called()

    def test_disappearing_record_is_error_not_empty_registry(self):
        self.seed()
        with patch.object(db,"get_person",return_value=None):
            with self.assertRaisesRegex(RuntimeError,"thay đổi"):
                self.query()

    def test_detection_alignment_and_adapter_failures_keep_cause(self):
        for stage in ("detection","alignment","adapter"):
            self.detector=fake_detector()
            with ExitStack() as stack:
                if stage=="detection":
                    self.detector.detect.side_effect=RuntimeError("detector failed")
                elif stage=="alignment":
                    self.detector.extract_faces_rgb.side_effect=ValueError("alignment failed")
                else:
                    stack.enter_context(patch.object(pipeline,"mtcnn_to_facenet_detections",
                                                     side_effect=ValueError("face_index 0: adapter failed")))
                with self.assertRaises((RuntimeError,ValueError)) as error:
                    self.query()
                self.assertIsNotNone(error.exception.__cause__)

    def test_adapter_count_and_metadata_errors(self):
        face={"face_rgb":np.zeros((160,160,3),np.uint8)}
        with patch.object(pipeline,"mtcnn_to_facenet_detections",return_value=[]):
            with self.assertRaises(ValueError):
                self.query()
        with patch.object(pipeline,"mtcnn_to_facenet_detections",return_value=[face]):
            with self.assertRaisesRegex(ValueError,"face_index 0"):
                self.query()

    def test_later_face_inference_failure_has_index_no_partial_return(self):
        self.seed();self.detector=fake_detector(2)
        returned=object();before=returned
        with self.assertRaisesRegex(RuntimeError,"face_index 1") as error:
            returned=self.query([unit(),RuntimeError("inference failed")])
        self.assertIs(returned,before)
        self.assertIsInstance(error.exception.__cause__,RuntimeError)

    def test_invalid_query_embedding_nonempty_registry(self):
        self.seed()
        nan=unit();nan[2]=np.nan
        for vector in (np.zeros(511,np.float32),unit().astype(np.float64),unit()*2,nan):
            with self.assertRaises((TypeError,ValueError)):
                self.query([vector])

    def test_reuses_approved_recognition_for_nonempty_registry(self):
        self.seed()
        original=pipeline.core.recognize_detected_faces
        with patch.object(pipeline.core,"recognize_detected_faces",wraps=original) as recognize:
            self.query()
        self.assertEqual(recognize.call_count,1)

    def test_no_registry_writing_apis_called(self):
        self.seed()
        with ExitStack() as stack:
            for name in ("initialize_database","add_person","update_person","delete_person"):
                stack.enter_context(patch.object(db,name,side_effect=AssertionError("Unexpected write API")))
            with self.readonly_check():
                self.query()

    def test_input_and_metadata_unchanged_output_has_independent_metadata(self):
        self.seed()
        image=self.image[:,::-1,:]
        before=image.copy()
        detected=self.detector.detect.return_value
        snapshot=deepcopy(detected)
        row=self.query(image=image)[0]
        row["bbox"][0]=999
        row["landmarks"]["nose"][0]=999
        self.assertTrue(np.array_equal(image,before))
        for name in ("boxes","scores","landmarks"):
            self.assertTrue(np.array_equal(getattr(detected,name),getattr(snapshot,name)))

    def test_bbox_clipping_and_negative_landmark_policy_preserved(self):
        detected=self.detector.detect.return_value
        detected.boxes[0]=[-5,-5,400,300]
        detected.landmarks[0,0,0]=-1
        row=self.query()[0]
        self.assertEqual(row["bbox"],[0,0,320,240])
        self.assertNotIn("landmarks",row)
        self.assertEqual(float(detected.boxes[0,0]),-5)

    def test_import_no_asset_access_model_loading_or_sqlite_connection(self):
        code=textwrap.dedent("""
            import os,sys
            sys.dont_write_bytecode=True
            def guard(event,args):
                if event=="open":
                    mode,flags=args[1:3]
                    if str(args[0]).lower().endswith((".onnx",".pt",".db",".sqlite",".png",".jpg",".ipynb")):
                        raise AssertionError("Asset accessed")
                    if ((isinstance(mode,str) and any(c in mode for c in "wax+")) or
                        (isinstance(flags,int) and flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))):
                        raise AssertionError("File write")
                if event in ("sqlite3.connect","os.mkdir","socket.connect"):
                    raise AssertionError("Unexpected I/O")
            sys.addaudithook(guard)
        """)+f"\nsys.path[:0]={[str(ROOT)]!r}\n"+textwrap.dedent("""
            import integration.recognition_pipeline
            assert "torch" not in sys.modules
            assert "MTCNN.mtcnn" not in sys.modules
            print("RECOGNITION_IMPORT_OK")
        """)
        result=subprocess.run([sys.executable,"-B","-c",code],cwd=Path(tempfile.gettempdir()),
            capture_output=True,text=True,encoding="utf-8",
            env={**os.environ,"PYTHONDONTWRITEBYTECODE":"1"},timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn("RECOGNITION_IMPORT_OK",result.stdout)


class TestRealRecognition(RegistryCase):
    @classmethod
    def setUpClass(cls):
        import torch
        import MTCNN.mtcnn as mtcnn_module
        import FaceNet.facenet_model as model_module
        cls.detector=mtcnn_module.MTCNN(device="cpu")
        cls.net=model_module.load_facenet_model(ROOT/"FaceNet/models/facenet.onnx")
        cls.version=build_model_version(cls.detector)
        for module in (mtcnn_module,model_module,pipeline):
            if not Path(module.__file__).resolve().is_relative_to(ROOT):
                raise AssertionError("Use integrated runtime modules only")
        if torch.version.cuda is not None or torch.cuda.is_available():
            raise AssertionError("CPU-only runtime required")

    def imm(self,indices):
        from MTCNN.preprocessing.image_processing import read_image
        source=Path("D:/Facenet-Recognition")
        manifest=json.loads((source/"data/processed_faces_imm/manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["dataset"],"IMM Face Database")
        records=sorted((r for r in manifest["records"] if r["person_id"]=="person_01" and r["status"]=="saved"),
                       key=lambda r:r["source_image"])
        selected=[records[index] for index in indices]
        self.assertEqual(len({r["source_sha256"] for r in selected}),len(indices))
        images=[]
        for record in selected:
            path=source/record["source_image"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record["source_sha256"])
            images.append(read_image(path))
        return images,{r["source_sha256"] for r in selected}

    def enroll(self,indices):
        images,hashes=self.imm(indices)
        enroll_person_from_images("imm_01_fixture",images,self.detector,self.net,
                                  self.path,self.version)
        return hashes

    def query(self,image,threshold):
        original=image.copy()
        with self.readonly_check():
            output=pipeline.recognize_faces_from_image(image,self.detector,self.net,
                                                       self.path,self.version,threshold)
        self.assertTrue(np.array_equal(image,original))
        return output

    def expected(self,image,threshold):
        from integration.mtcnn_adapter import mtcnn_to_facenet_detections
        detected=self.detector.detect(image)
        aligned=self.detector.extract_faces_rgb(image,detected,size=160,align=True)
        faces=mtcnn_to_facenet_detections(image,detected,aligned)
        registered={p["person_id"]:db.get_person(self.path,p["person_id"],self.version)["embedding"]
                    for p in db.list_people(self.path)}
        expected=[]
        for face in faces:
            embedding=extract_embedding(face["face_rgb"],self.net)
            expected.append((face,detect_unknown(embedding,registered,threshold)))
        return expected

    def test_a_single_enrollment_separate_query_accept_all_control(self):
        enrolled=self.enroll((0,))
        images,query_hashes=self.imm((1,))
        self.assertTrue(enrolled.isdisjoint(query_hashes))
        # -1 is a predeclared accept-all cosine control, not a calibrated threshold.
        output=self.query(images[0],-1.0)
        self.assertEqual(len(output),1)
        self.assertEqual(output[0]["status"],"recognized")
        self.assertEqual(output[0]["person_id"],"imm_01_fixture")
        self.assertEqual(output[0]["model_version"],self.version)
        print("REAL_RECOGNITION "+json.dumps({"scenario":"single_enrollment_separate_query",
            "enrollment_images":1,"query_faces":1,"threshold":-1.0,
            "status":"recognized","disjoint_source_images":True,"registry_readonly":True,
            "accuracy_evaluation":False}))

    def test_b_three_enrollment_images_heldout_query_matches_approved_policy(self):
        enrolled=self.enroll((0,1,2))
        images,query_hashes=self.imm((3,))
        self.assertTrue(enrolled.isdisjoint(query_hashes))
        # Fixed policy-test value chosen before inference; no accuracy assertion.
        output=self.query(images[0],.5)
        reference=self.expected(images[0],.5)
        self.assertEqual(len(output),1)
        decision=reference[0][1]
        self.assertEqual(output[0]["status"],"recognized" if decision["is_known"] else "unknown")
        self.assertEqual(output[0]["person_id"],decision["predicted_id"] if decision["is_known"] else None)
        difference=abs(output[0]["similarity"]-decision["similarity"])
        self.assertLessEqual(difference,1e-12)
        print("REAL_RECOGNITION "+json.dumps({"scenario":"three_image_gallery_heldout_query",
            "enrollment_images":3,"query_faces":1,"threshold":.5,"status":output[0]["status"],
            "cosine_max_abs_difference":difference,"policy_mismatches":0,
            "disjoint_source_images":True,"registry_readonly":True,"accuracy_evaluation":False}))

    def test_c_group_faces_independently_match_approved_policy(self):
        from MTCNN.preprocessing.image_processing import read_image
        self.enroll((0,))
        image=read_image(ROOT/"integration/tests/fixtures/mtcnn_multi.jpg")
        output=self.query(image,.5)
        reference=self.expected(image,.5)
        self.assertEqual(len(output),5)
        self.assertEqual(len(reference),5)
        maximum=0.0
        for index,(row,(face,decision)) in enumerate(zip(output,reference)):
            self.assertEqual(row["face_index"],index)
            self.assertEqual(row["bbox"],face["bbox"])
            self.assertEqual(row["confidence"],float(face["confidence"]))
            self.assertEqual(row["status"],"recognized" if decision["is_known"] else "unknown")
            self.assertEqual(row["person_id"],decision["predicted_id"] if decision["is_known"] else None)
            maximum=max(maximum,abs(row["similarity"]-decision["similarity"]))
        self.assertLessEqual(maximum,1e-12)
        print("REAL_RECOGNITION "+json.dumps({"scenario":"group_independent_faces",
            "query_faces":5,"policy_mismatches":0,"cosine_max_abs_difference":maximum,
            "order_preserved":True,"registry_readonly":True,"accuracy_evaluation":False}))

    def test_d_no_face_image_returns_empty(self):
        self.enroll((0,))
        self.assertEqual(self.query(np.zeros((240,320,3),np.uint8),.5),[])
        print("REAL_RECOGNITION "+json.dumps({"scenario":"no_face","query_faces":0,
            "registry_readonly":True}))

    def test_e_separate_query_unknown_strict_control(self):
        enrolled=self.enroll((0,))
        images,query_hashes=self.imm((4,))
        self.assertTrue(enrolled.isdisjoint(query_hashes))
        # 1.0 only accepts cosine equality. Predeclared rejection control.
        output=self.query(images[0],1.0)
        self.assertEqual(len(output),1)
        self.assertLess(output[0]["similarity"],1.0)
        self.assertEqual(output[0]["status"],"unknown")
        self.assertIsNone(output[0]["person_id"])
        self.assertEqual(output[0]["best_match_id"],"imm_01_fixture")
        print("REAL_RECOGNITION "+json.dumps({"scenario":"strict_unknown_control",
            "query_faces":1,"threshold":1.0,"status":"unknown",
            "candidate_similarity":output[0]["similarity"],"disjoint_source_images":True,
            "registry_readonly":True,"accuracy_evaluation":False}))

    def test_f_empty_registry_real_embedding_returns_unknown(self):
        from MTCNN.preprocessing.image_processing import read_image
        image=read_image(ROOT/"integration/tests/fixtures/mtcnn_single.png")
        output=self.query(image,.5)
        self.assertEqual(len(output),1)
        self.assertEqual(output[0]["status"],"unknown")
        self.assertIsNone(output[0]["person_id"])
        self.assertIsNone(output[0]["similarity"])
        print("REAL_RECOGNITION "+json.dumps({"scenario":"empty_valid_registry",
            "query_faces":1,"status":"unknown","candidate_exists":False,
            "registry_readonly":True}))


if __name__=="__main__":
    unittest.main()
