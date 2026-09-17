"""WMDP.py — Pipeline อ่านเลขมิเตอร์น้ำ SANWA จากภาพถ่าย (4 หลัก, จัดการ "เลขเหลื่อม" ได้)

ภาพรวมการไหลของข้อมูล (เรียกจากภายนอกผ่าน read_water_meter() เป็นจุดเข้าเดียว):
  1. init_ai(model_folder_path, output_folder) — โหลดโมเดลทั้งหมด (YOLO x2 + ViT ตัวจำแนกหลัก) ครั้งเดียว
     ก่อนเริ่มประมวลผล ต้องเรียกก่อนเรียก read_water_meter() เสมอ ผลคือตั้งค่า global model_yolo_main,
     model_yolo_digits, modelTB, processor และ CURRENT_OUT_DIR (โฟลเดอร์ปลายทางของทุกไฟล์ debug/output)
  2. read_water_meter(image, filename_ref) — ประมวลผล 1 ภาพ ไล่ผ่าน Step 1-7 (SECTION 7 ด้านล่าง)
     คืน dict {final_reading, error, meter_found, dial_found, screen_found, pattern, all_ids, viz_image}
     ผู้เรียก (เช่น TEST_app.py) เอา dict นี้ไปตัดสินใจต่อ เช่น แยกภาพลงโฟลเดอร์ตามหมวดหมู่ (ดูฟังก์ชัน
     run_single_process ใน TEST_app.py) — WMDP.py เองไม่ตัดสินใจเรื่องการจัดหมวดหมู่ภาพแทนผู้เรียก
  3. read_water_meter_batch(image_list, filename_refs) — [เสริม, SECTION 9] เรียก read_water_meter() วน
     หลายภาพของมิเตอร์ตัวเดียวกัน แล้วโหวตเสียงข้างมากทีละหลัก ให้ผลแม่นกว่าเชื่อภาพเดียว (ดู docstring ฟังก์ชัน)

  ทุกครั้งที่เรียก read_water_meter() ไฟล์ debug ของภาพนั้นจะถูกเขียนลง
  <CURRENT_OUT_DIR>/debug/<clean_name>/ เสมอ (ภาพทุกขั้นตอน + result.txt สรุปผล) โดย clean_name = ชื่อไฟล์
  ภาพ (ตัดนามสกุลออก) — ดู SECTION 6 (_save_pipeline_debug_outputs) สำหรับรายละเอียดว่าไฟล์ไหนคือขั้นตอนไหน

โครงสร้างไฟล์ (เรียงตามลำดับที่ประกาศ ไม่ใช่ลำดับที่ทำงานจริง — ลำดับทำงานจริงดู SECTION 7):
  SECTION 1  Global State & Stats Tracking      ตัวแปร global + คลาสเก็บสถิติ detection_report.txt
  SECTION 2  Geometry & Box Helpers              ฟังก์ชันช่วยเรื่องกรอบ/มุม/หมุนภาพ ใช้ร่วมกันหลายจุด
  SECTION 3  Digit Classification & Straddle     หัวใจของการ "แปลงเลขเหลื่อมเป็นเลขเต็ม" (ดู _resolve_digit_slot)
  SECTION 4  Digit Slot Detection (YOLO)         สแกนหากรอบหลักเลขแต่ละตัวบนจอ
  SECTION 5  Visualization Helpers               วาดภาพอธิบายผลรวม (99_annotated ที่ผู้ใช้ทั่วไปน่าจะดู)
  SECTION 6  Debug Image / Result File I/O       เซฟภาพ debug ทุกขั้นตอน + result.txt ลงดิสก์
  SECTION 7  Pipeline Steps                      ฟังก์ชันย่อยของแต่ละ Step 1-5 + read_water_meter() (จุดเข้า)
  SECTION 9  Multi-Photo Majority Vote           ฟีเจอร์เสริม โหวตหลายภาพต่อเหตุการณ์ถ่าย 1 ครั้ง
"""
import os
import cv2
import torch
import math
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

# ============================================================================
# SECTION 1: Global State & Stats Tracking
# ============================================================================
modelTB = None
processor = None
model_yolo_main = None
model_yolo_digits = None
CURRENT_OUT_DIR = "Test_copy"

# margin (confidence top-1 ลบ confidence top-2) ต่ำกว่านี้ = classifier ลังเลระหว่าง 2 คลาส ตั้ง flag
# low_confidence เตือนไว้ (ดู _step5_classify_digits) — ยังไม่มีค่าที่ "ถูกต้องแน่นอน" ตั้งไว้กว้างๆ ก่อน
# เผื่อดูสถิติจริงจาก 348 ภาพว่าเคสที่ตอบผิดกระจุกอยู่ต่ำกว่าค่านี้จริงไหมค่อยปรับทีหลัง
LOW_CONFIDENCE_MARGIN = 0.20

# วิธีตัดสินฝั่งชนะของเลขเหลื่อม (ดู _resolve_digit_slot) — ค่าเริ่มต้น "centroid" คือวิธีที่ใช้งานจริง
# (พิสูจน์แล้วว่าแม่นกว่า "area" อย่างมีนัยสำคัญ ดู docstring _resolve_digit_slot) ตัวแปรนี้มีไว้ให้สลับ
# เป็น "area" ชั่วคราวเพื่อรันเปรียบเทียบ/ทำรายงานเสนออาจารย์เท่านั้น — TEST_app.py เป็นผู้ตั้งค่านี้ก่อนรัน
# ไม่ควรแก้ default ในไฟล์นี้เว้นแต่ตั้งใจเปลี่ยนพฤติกรรมจริงของระบบ (ค่า: "centroid" หรือ "area")
STRADDLE_WINNER_METHOD = "centroid"

# ทดลองแล้วไม่ได้ผล (28 ส.ค. 2569): เคยลองเพิ่มสัญญาณ "edge_gap" (ระยะจากขอบนอกสุดของ crop ถึงหมึกแรก/
# สุดท้ายของแต่ละฝั่ง) หวังว่าเลขที่ไม่ใช่ค่าจริงจะถูกตัดชิดขอบมากกว่า — วัดค่าจริงแล้วพบว่า top/bottom
# edge_gap = 0.0 ทุกภาพทุกฝั่งเสมอ เพราะ YOLO digit-slot detector ตัดกรอบชิดเนื้อเลขอยู่แล้วโดยธรรมชาติ
# ไม่มี padding เหลือให้เทียบเลย จึงไม่มีข้อมูลอะไรให้ใช้ตั้งแต่ต้น — ลบโค้ดทิ้งแล้ว (ดู
# หาข้อที่เป็นพิษ/1_วิเคราะห์สาเหตุรายภาพ.txt กลุ่ม 1-2 สำหรับบริบทที่มาของไอเดียนี้)

class DetectionStats:
    def __init__(self):
        self.stats = {f"{i:04b}": 0 for i in range(16)}
        self.total_images = 0
        self.failed_files = []

    def update(self, pattern_str, filename):
        if pattern_str in self.stats:
            self.stats[pattern_str] += 1
            self.total_images += 1
            if pattern_str != "1111":
                self.failed_files.append((filename, pattern_str))

    def save_report(self):
        filename = os.path.join(CURRENT_OUT_DIR, "detection_report.txt")
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("="*45 + "\n")
                f.write(f"{'Four digit':<15} {'Detected':<10} {'(%)':<10}\n")
                f.write("="*45 + "\n")
                for i in range(16):
                    key = f"{i:04b}"
                    count = self.stats[key]
                    percent = (count / self.total_images * 100) if self.total_images > 0 else 0.0
                    f.write(f"{key:<15} {count:<10} {percent:.2f}\n")
                f.write("="*45 + "\n")
                f.write(f"Total Images: {self.total_images}\n\n")

                f.write("="*45 + "\n")
                f.write("FILES WITH INCOMPLETE DIGIT DETECTION\n")
                f.write("="*45 + "\n")
                if not self.failed_files:
                    f.write("None\n")
                else:
                    f.write(f"{'Filename':<30} {'Pattern':<10}\n")
                    f.write("-" * 45 + "\n")
                    for fname, pattern in self.failed_files:
                        f.write(f"{fname:<30} {pattern:<10}\n")
                f.write("="*45 + "\n")
        except Exception as e:
            print(f"Error saving report: {e}")

stat_tracker = DetectionStats()

def init_ai(model_folder_path, output_folder):
    global modelTB, processor, model_yolo_main, model_yolo_digits, CURRENT_OUT_DIR, stat_tracker
    CURRENT_OUT_DIR = output_folder
    os.makedirs(CURRENT_OUT_DIR, exist_ok=True)
    stat_tracker = DetectionStats()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name_lower = os.path.basename(model_folder_path).lower()

    try: processor = AutoImageProcessor.from_pretrained(model_folder_path)
    except Exception:  # bare except เดิมดัก KeyboardInterrupt/SystemExit ไปด้วย -> กด Ctrl+C ตอนโหลดโมเดลอาจไม่หยุดจริง
        if "vit" in model_name_lower: processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
        else: processor = AutoImageProcessor.from_pretrained("microsoft/resnet-18")

    modelTB = AutoModelForImageClassification.from_pretrained(model_folder_path, use_safetensors=True).to(device)
    modelTB.eval()

    if model_yolo_main is None:
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_yolo_main = YOLO(os.path.join(BASE_DIR, "model-1-semi.pt"))
        model_yolo_digits = YOLO(os.path.join(BASE_DIR, "digits.pt"))

# ============================================================================
# SECTION 2: Geometry & Box Helpers
# ============================================================================
def _label_crop(crop, text, color=(0, 255, 0)):
    """คืนสำเนาของ crop พร้อมข้อความกำกับไว้มุมซ้ายบน — ทำงานบนสำเนาเสมอ
    เพื่อไม่ให้กระทบภาพต้นฉบับที่ตัวแปรเดิมยังถูกใช้ประมวลผลต่อ (เช่น ส่งเข้า predict รอบถัดไป)"""
    labeled = crop.copy()
    cv2.putText(labeled, text, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(labeled, text, (2, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return labeled

def _best_box(boxes, clss, confs, target_cls, max_area_ratio=None, image_area=None):
    """คืนกล่องที่ confidence สูงสุดของ class ที่ต้องการ — ใช้แทนการวนลูปเลือกกล่อง "ตัวที่มาทีหลังทับ"
    ซึ่งเลือกกล่องแบบสุ่มตามลำดับที่โมเดลคืนมา ไม่ใช่กล่องที่น่าเชื่อถือที่สุด
    ถ้าระบุ max_area_ratio+image_area จะตัดกล่องที่ใหญ่ผิดปกติทิ้งก่อนเทียบ conf — กันกรณี Dial ปลอม
    (โมเดลจำแนกวงมิเตอร์ทั้งวงผิดเป็น class Dial ทั้งที่หน้าปัดเข็มจริงเล็กมากเทียบกับทั้งภาพ)"""
    best_box, best_conf = None, -1.0
    for b, c, cf in zip(boxes, clss, confs):
        if int(c) != target_cls: continue
        if max_area_ratio is not None and image_area:
            area = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
            if area > image_area * max_area_ratio: continue
        if cf > best_conf:
            best_box, best_conf = b, cf
    return best_box

def _meter_or_promoted_dial(boxes, clss, confs, image_area, max_dial_area_ratio=0.10):
    """หา Meter (class 0) ตามปกติ ถ้าไม่เจอเลย เช็คว่ากล่อง Dial (class 1) ที่ถูก _best_box คัดทิ้งเพราะ
    ใหญ่เกิน max_dial_area_ratio ของภาพ แท้จริงคือวงมิเตอร์ทั้งวงที่โมเดลจำแนกผิด class หรือไม่ — เคยเจอกรณีจริง
    ที่กรอบ "Dial" ใหญ่ผิดปกติ กับกรอบ Meter conf ต่ำที่โดน predict(conf=0.7) กรองทิ้งไปเอง มีพิกัดตรงกันเกือบเป๊ะ
    (พิกเซลเดียวกัน แค่โมเดลให้ label คนละตัวโดย conf ใกล้เคียงกัน) ถ้าเจอ ใช้กล่องนั้นแทน Meter
    เงื่อนไข activate แคบมาก (ต้องไม่เจอ Meter เลย + มี Dial ใหญ่ผิดปกติ) จึงไม่กระทบภาพที่ตรวจ Meter เจอตามปกติ"""
    bbox_m = _best_box(boxes, clss, confs, 0)
    if bbox_m is None:
        raw_dial = _best_box(boxes, clss, confs, 1)
        if raw_dial is not None:
            area = max(0.0, raw_dial[2] - raw_dial[0]) * max(0.0, raw_dial[3] - raw_dial[1])
            if area > image_area * max_dial_area_ratio:
                bbox_m = raw_dial
    return bbox_m

def _pad_box(b, shape, pad_ratio=0.15, min_pad=2):
    x1, y1, x2, y2 = b
    bw = x2 - x1; bh = y2 - y1
    pad_x = max(min_pad, int(bw * pad_ratio)); pad_y = max(min_pad, int(bh * pad_ratio))
    x1p = max(0, int(x1) - pad_x); y1p = max(0, int(y1) - pad_y)
    x2p = min(shape[1], int(x2) + pad_x); y2p = min(shape[0], int(y2) + pad_y)
    return x1p, y1p, x2p, y2p

def _rotate_to_target(image, pivot, ref_point, target_deg_std):
    """หมุนภาพรอบจุด pivot ให้จุด ref_point ไปอยู่ที่มุม target_deg_std ตามมุมมองมาตรฐาน
    (0=ขวา, 90=บน, 180=ซ้าย, 270=ล่าง — นับทวนเข็มนาฬิกาจากขวา) คืนค่า (rotated_image, rotate_by_degrees)"""
    cx_p, cy_p = pivot
    cx_r, cy_r = ref_point
    raw_angle = math.degrees(math.atan2(cy_r - cy_p, cx_r - cx_p))
    if raw_angle < 0: raw_angle += 360
    target_ydown = (360 - target_deg_std) % 360  # แปลงมุมมองมาตรฐาน (y ขึ้นบวก) เป็นมุมพิกัดภาพ (y ลงบวก)
    rotate_by = (raw_angle - target_ydown) % 360
    rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((cx_p, cy_p), rotate_by, 1.0),
                              (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
    return rotated, rotate_by

# ============================================================================
# SECTION 3: Digit Classification & Straddle Resolution
# ============================================================================
def _classify_crops(crops_bgr):
    """จำแนกทุก crop รวดเดียวเป็น batch เดียว คืน list ของ dict {"id", "confidence", "id2", "confidence2"}
    ยาวเท่าอินพุตเสมอ — "id"/"confidence" คือคลาสที่มั่นใจสูงสุด (top-1) กับความน่าจะเป็นของมัน (softmax, 0-1)
    "id2"/"confidence2" คือคลาสอันดับ 2 ไว้เทียบดูว่า classifier "ลังเล" ระหว่าง 2 คลาสแค่ไหน (ยิ่ง
    confidence-confidence2 ห่างกันน้อย ยิ่งเสี่ยงทายผิด) — เดิมคืนแค่ argmax ทิ้งข้อมูลความมั่นใจไปเฉยๆ
    ทำให้ตรวจไม่ได้เลยว่าเคสไหน classifier เดามั่วมา (ดู _step5_classify_digits ตรง low_confidence flag)
    เดิมกรอง crop ว่างทิ้งเงียบๆ ซึ่งอันตราย เพราะผู้เรียกใช้ zip() จับคู่ผลกับ slot ตามลำดับ —
    ถ้าความยาวหดลง id จะเลื่อนไปใส่ผิดหลักโดยไม่มีสัญญาณเตือนใดๆ จึงเปลี่ยนเป็นโยน error แทน
    (ตอนนี้ผู้เรียกกรอง crop ว่างออกก่อนแล้ว เงื่อนไขนี้จึงไม่ควรเกิด แต่กันไว้ไม่ให้พังเงียบในอนาคต)"""
    if any(c is None or c.size == 0 for c in crops_bgr):
        raise ValueError("_classify_crops received an empty crop; caller must filter first")
    pil_images = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops_bgr]
    if not pil_images: return []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if processor: inputs = processor(images=pil_images, return_tensors='pt').to(device)
    else:
        tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        inputs = {'pixel_values': torch.stack([tfm(img) for img in pil_images]).to(device)}
    with torch.no_grad():
        probs = modelTB(**inputs).logits.softmax(dim=-1)          # (N, 20) ความน่าจะเป็นแต่ละคลาส รวมกัน = 1
        top2_probs, top2_ids = probs.topk(2, dim=-1)               # (N, 2) เรียงจากมั่นใจสุดไปรองลงมา
    results = []
    for ids_row, probs_row in zip(top2_ids.tolist(), top2_probs.tolist()):
        results.append({
            "id": ids_row[0], "confidence": probs_row[0],
            "id2": ids_row[1], "confidence2": probs_row[1],
        })
    return results

def _split_straddle_crop(crop):
    """Split a straddling digit's own slot-crop into (top_band, bottom_band) via ink-projection valley.
    ตัวเลขในมิเตอร์นี้คือ "เลขขาวบนพื้นดำ" (ไม่ใช่ดำบนขาว) — หลัง CLAHE+threshold+INV แล้ว พิกเซลขาว(255)ที่ได้คือ
    "พื้นหลัง" ไม่ใช่ตัวเลข จึงกลับขั้วเป็น `ink_mask` ให้ขาว=เนื้อตัวเลขจริง เพื่อให้ตัวแปร/สูตรทั้งหมดด้านล่างสื่อ
    ความหมายตรงตัว (แก้บั๊กเดิมที่ใช้ profile ดิบแบบกลับด้านโดยไม่รู้ตัว 2 จุดพร้อมกัน):
      1) การหา valley (รอยต่อจริงระหว่าง 2 เลข) ต้องมองหาแถวที่ "เนื้อเลขน้อยที่สุด" (argmin) ไม่ใช่มากที่สุด —
         โค้ดเดิมดันเลือกแถวที่เนื้อเลขเยอะที่สุด (เช่น กลางคอคอดของเลข 8/9/0) เป็นเหตุให้ตัดภาพออกมาเป็นครึ่งเลข
         ทั้งที่ slot ต้นทางจับเลขมาเต็มทั้งคู่ (พิสูจน์แล้วกับ TEST_113: เดิมตัดกลางคอคอดเลข "8" แทนที่จะตัดตรง
         รอยต่อกับเลข "7" ด้านบน) — ถ่วงน้ำหนักให้เอียงเข้ากึ่งกลางภาพเล็กน้อยเป็นตัวตัดสินรอง กันเลือกช่องว่าง
         ภายในตัวเลขตัวเดียว (เช่น ในเลข 7 ที่มีพื้นที่โปร่งเยอะ) ที่ไม่ใช่รอยต่อจริง
      2) การคิดทั้ง ink area และ centroid ต้องคิดจากเนื้อตัวเลข ไม่ใช่พื้นหลัง
    Returns each band's ink area alongside its centroid. ตัวตัดสินผู้ชนะที่ _resolve_digit_slot ใช้จริงคือ
    ระยะจากกึ่งกลาง slot ถึง centroid ของแต่ละฝั่ง (ใกล้กึ่งกลางกว่า = โผล่ในกรอบเต็มกว่า = ชนะ) โดยมี
    ink area เป็นตัวตัดสินรองเฉพาะตอนระยะเท่ากันพอดี — ดูเหตุผลและตัวเลขที่วัดจริงได้ที่ _resolve_digit_slot

    ปรับปรุงจากเวอร์ชันก่อนหน้า 4 จุด:
      A) ช่วงค้นหา valley ขยายจาก 25%-75% เป็น 15%-85% ของความสูง crop — เผื่อรอยต่อจริงในบางภาพอยู่
         ใกล้ขอบกว่าที่คิด (ตัวเลขไม่ได้อยู่กึ่งกลาง slot เป๊ะเสมอไป)
      B) center_penalty ลดจาก 0.3 เหลือ 0.15 (ให้ ink profile จริงมีน้ำหนักตัดสินมากขึ้น) แต่แลกด้วยการหา
         "หลายจุดตัดสินใจ" (multi-candidate) แทนจุดเดียว — เรียงผู้สมัคร valley จากคะแนนดีสุดไปแย่สุด แล้วไล่
         เช็คทีละจุดว่าตัดแบ่งแล้วทั้งสองฝั่งมีเนื้อเลขจริงหรือไม่ (กันเคสตัดแล้วฝั่งใดฝั่งหนึ่งว่างเปล่า/มีแต่พื้นหลัง)
         ถ้าจุดที่คะแนนดีที่สุดตัดแล้วฝั่งใดฝั่งหนึ่งไม่มีเนื้อเลขเลย จะข้ามไปลองจุดถัดไปในลิสต์แทน
      C) ink_profile เดิมเป็น row-sum ดิบ (กระตุกง่ายเมื่อภาพมี noise) เพิ่ม smoothing (moving average กว้าง 3
         แถว) ก่อนนำไปคำนวณคะแนนหา valley — ค่าที่ smooth แล้วใช้แค่ตอนหา valley เท่านั้น ส่วน area/centroid
         ยังคำนวณจาก ink_profile ดิบเหมือนเดิม กันไม่ให้ smoothing ไปบิดเบือนตำแหน่งจุดศูนย์ถ่วงจริง
      D) threshold เปลี่ยนจาก Otsu (global) เป็น adaptive threshold (local) — ทนต่อแสงไม่สม่ำเสมอในภาพเดียวกัน
         ได้ดีกว่า (เช่น มุมหนึ่งของ crop โดนแดดจ้ากว่าอีกมุม) ซึ่ง Otsu แบบ threshold เดียวทั้งภาพจัดการไม่ได้"""
    gray_plain = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)  # เก็บไว้ก่อน CLAHE แยกต่างหาก ใช้ทำภาพ debug x1_grayscale
    # CLAHE ก่อน threshold — จอมิเตอร์กลางแจ้งมักมีแสงสะท้อน ทำให้ threshold เพี้ยนและ valley ผิดตำแหน่ง
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_plain)
    # (D) adaptive threshold แทน Otsu — blockSize ต้องเป็นเลขคี่ >= 3 และปรับตามขนาด crop จริง กันภาพเล็กเกินไปพัง
    # C=5 เคยลองแล้วแต่ "พิสูจน์จริงจากรัน 348 ภาพว่าแย่ลงสุทธิ" (84.5% -> 83.3%) เพราะสีเทาที่ผ่อนปรนเข้ามา
    # ไปเติมเต็มช่องว่างตรงรอยต่อจริงระหว่าง 2 เลขด้วย (ไม่ใช่แค่เนื้อเลขหนาขึ้นอย่างที่ตั้งใจ) ทำให้ valley
    # หลุดไปเจอจุดอื่นที่ผิดตำแหน่งแทน — เทียบ valley position ของเคสที่ min_frac ช่วยแก้ได้จริง (TEST_19/47/98)
    # พบว่า C=2 กับ C=5 ให้ตำแหน่ง valley เหมือนกันเป๊ะ แปลว่าการแก้ที่ได้ผลจริงคือ min_frac + ช่วงค้นหา
    # ไม่ใช่ C จึงคืนค่ากลับเป็น 2 (ค่าเดิม) ส่วนเรื่อง "สีเทาอ่อนนับเป็นหมึกขาว" ต้องหาวิธีอื่นที่ไม่กระทบ valley
    block_size = max(3, (gray.shape[0] // 3) | 1)  # | 1 บังคับให้เป็นเลขคี่เสมอ
    bg_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, block_size, 2)
    ink_mask = 255 - bg_mask  # กลับขั้ว: ขาว(255)=เนื้อตัวเลขจริง, ดำ(0)=พื้นหลัง
    ink_profile = ink_mask.sum(axis=1).astype(np.float64)  # สูง = แถวนั้นมีเนื้อเลขเยอะ (ดิบ, ใช้คำนวณ area/centroid)
    h = len(ink_profile)
    if h < 6: raise ValueError("crop too small to split")

    # (C) smoothing เฉพาะสำหรับหา valley — moving average กว้าง 3 แถว กัน noise ทำให้ argmin เลือกผิดจุด
    kernel = np.ones(3) / 3.0
    ink_profile_smooth = np.convolve(ink_profile, kernel, mode="same")

    # (A) ช่วงค้นหาขยายเป็น 20%-85% (ปรับจาก 15% ตามที่พบว่า 15% ยังปล่อยให้หลุดไปเจอขอบว่างเปล่าได้บ่อย)
    lo, hi = int(h * 0.20), int(h * 0.85)
    if hi <= lo: lo, hi = 1, h - 1
    center = h / 2.0
    max_ink = float(ink_profile_smooth.max()) or 1.0
    rows = np.arange(lo, hi)
    ink_score = ink_profile_smooth[lo:hi] / max_ink              # 1=เนื้อเลขล้วน, 0=พื้นหลังล้วน (รอยต่อจริง)
    center_penalty = np.abs(rows - center) / (h / 2.0)           # 0=อยู่กึ่งกลางพอดี, ~1=อยู่ริมภาพ
    # (B) น้ำหนัก center_penalty ลดเหลือ 0.15 + หาผู้สมัคร valley หลายจุดเรียงจากดีสุดไปแย่สุด
    combined_score = ink_score + 0.15 * center_penalty
    candidate_order = np.argsort(combined_score)  # ดีสุด(น้อยสุด) ก่อน

    # (B แก้เพิ่ม) เดิมเช็คแค่ band.sum() > 0 ซึ่งหลวมเกินไป — พบบั๊กจริงจากภาพตัวอย่าง (TEST_24/100/125 ฯลฯ)
    # ว่า valley มักไปเจอ "ขอบว่างเปล่า" ริมภาพ (แทบไม่มีหมึกเลย) แทนที่จะเจอรอยต่อจริงระหว่าง 2 เลข เพราะ
    # ขอบว่างมี ink_score ต่ำกว่ารอยต่อจริง (รอยต่อจริงยังมีหมึกเหลือติดจากส่วนโค้งของตัวเลขทั้งสองฝั่ง) ผลคือ
    # ฝั่งหนึ่งกลายเป็น "เกือบทั้งภาพ" (มีเนื้อ 2 เลขปนกัน) ทำให้ centroid เพี้ยนเข้าใกล้กึ่งกลางเทียมๆ จึงต้อง
    # บังคับว่าแต่ละฝั่งต้องมีสัดส่วนหมึกอย่างน้อย MIN_FRAC ของหมึกทั้งหมด ไม่ใช่แค่ > 0 — ไล่ผ่อนเกณฑ์ลงเป็นชั้นๆ
    # (0.20 -> 0.10 -> 0.0) ถ้าหาไม่เจอเลยในชั้นที่เข้มกว่า กันไม่ให้ภาพที่หมึกน้อยจริงๆ (เช่น เลข 1) พังไปด้วย
    total_ink = float(ink_profile.sum()) or 1.0

    def _try_valley(idx, min_frac):
        vr = lo + int(idx)
        if vr <= 0 or vr >= h - 1: return None
        top_sum = float(ink_profile[:vr].sum()); bottom_sum = float(ink_profile[vr:].sum())
        if top_sum < min_frac * total_ink or bottom_sum < min_frac * total_ink: return None
        return vr

    valley_row = None
    for min_frac in (0.20, 0.10, 0.0):
        for idx in candidate_order:
            valley_row = _try_valley(idx, min_frac)
            if valley_row is not None: break
        if valley_row is not None: break
    if valley_row is None:
        # ไม่มีจุดไหนผ่านแม้แต่เกณฑ์หลวมสุด (0.0) — fallback กลับไปใช้จุดคะแนนดีที่สุดแบบเดิม (ไม่เช็คเงื่อนไข)
        valley_row = lo + int(candidate_order[0])
        if valley_row <= 0 or valley_row >= h - 1: raise ValueError("no clear valley found")

    top_band = crop[0:valley_row, :]; bottom_band = crop[valley_row:h, :]
    if top_band.size == 0 or bottom_band.size == 0: raise ValueError("empty band after split")

    top_profile = ink_profile[:valley_row]; bottom_profile = ink_profile[valley_row:]
    top_area = float(top_profile.sum()); bottom_area = float(bottom_profile.sum())
    top_centroid = float(np.average(np.arange(valley_row), weights=top_profile)) if top_area > 0 else valley_row / 2
    bottom_centroid = valley_row + (float(np.average(np.arange(h - valley_row), weights=bottom_profile)) if bottom_area > 0 else (h - valley_row) / 2)


    # ภาพ debug 4 ขั้นตอน (grayscale/CLAHE/adaptive_inkmask/valley+centroid) — สร้างไว้ทุกครั้งที่ตัดสิน
    # เลขเหลื่อมจริง (ไม่ใช่แค่ตอนสร้างภาพประกอบนำเสนอ) ให้ _save_pipeline_debug_outputs เอาไปเซฟต่อเป็น
    # 04b_slot{i}_x1..x4 ในโฟลเดอร์ debug/<ชื่อภาพ>/ ของทุกรอบรันจริง เรียงต่อจาก 04b_slot{i}_raw ทันที
    marked = cv2.cvtColor(ink_mask, cv2.COLOR_GRAY2BGR)
    cv2.line(marked, (0, valley_row), (marked.shape[1], valley_row), (0, 0, 255), 2)
    cv2.line(marked, (0, int(center)), (marked.shape[1], int(center)), (0, 255, 0), 1)
    cv2.circle(marked, (marked.shape[1] // 2, int(top_centroid)), 3, (255, 0, 255), -1)
    cv2.circle(marked, (marked.shape[1] // 2, int(bottom_centroid)), 3, (0, 165, 255), -1)
    step_images = {
        "x1_grayscale": gray_plain,
        "x2_clahe": gray,
        "x3_adaptive_inkmask": ink_mask,
        "x4_valley_and_centroid": marked,
    }
    return top_band, bottom_band, top_area, bottom_area, top_centroid, bottom_centroid, center, step_images

def _resolve_digit_slot(crop, base_id):
    """ตัดสินค่าของ 1 หลัก จาก class id ที่จำแนกไว้แล้ว (base_id) — ไม่เรียก classifier ในนี้เอง
    ตัวเรียก (_step5_classify_digits) จะจำแนกทุก slot ของภาพเดียวกันเป็น batch เดียวรวดเดียวก่อน
    แล้วค่อยส่ง id ของแต่ละ slot เข้ามาที่นี่ทีละอัน — ประหยัดกว่าเรียก modelTB ทีละ 1 ภาพต่อ 1 หลัก
    ถ้าเป็นเลขเหลื่อม (odd class id) คู่ตัวเลือกรู้ได้จาก id เลย: v = id//2 (ตัวบน) และ (v+1)%10 (ตัวล่าง)
    ไม่ต้องจำแนกซ้ำสำหรับครึ่งเลข ผู้ชนะตัดสินจาก "ระยะจากกึ่งกลาง slot ถึง centroid ของเนื้อเลขแต่ละฝั่ง"
    เป็นหลัก (ใกล้กึ่งกลางกว่าชนะ) โดยมีพื้นที่หมึกเป็นตัวตัดสินรองเฉพาะตอนระยะเท่ากันพอดี
    หมายเหตุ: ค่าที่คืนคือ "ค่าที่ตัดสินแล้ว" ไม่ใช่ค่าดิบจากตัวจำแนก ผู้เรียกที่ต้องการ id ดิบให้ใช้ base_id
    คืน (value, is_straddle, is_90_straddle, resolved_is_zero, debug_images)
    debug_images มี top_value/bottom_value/winner/valley_row ไว้ใช้วาดภาพอธิบายด้วย"""
    debug = {}
    v = base_id // 2
    if base_id % 2 == 0:
        return v, False, False, False, debug

    v_next = (v + 1) % 10
    is_90 = (v == 9)
    try:
        top_band, bottom_band, top_area, bottom_area, top_c, bottom_c, center, step_images = _split_straddle_crop(crop)
        debug["top"] = top_band; debug["bottom"] = bottom_band
        debug["top_value"] = v; debug["bottom_value"] = v_next
        debug["valley_row"] = top_band.shape[0]
        debug.update(step_images)  # x1_grayscale/x2_clahe/x3_adaptive_inkmask/x4_valley_and_centroid
        # ตัดสินด้วย "ระยะจากกึ่งกลางภาพถึง centroid ของเนื้อเลขแต่ละฝั่ง" เป็นหลัก — ฝั่งที่ centroid
        # อยู่ใกล้กึ่งกลาง slot มากกว่าคือเลขที่โผล่อยู่ในกรอบหน้าต่างเต็มกว่า จึงเป็นค่าที่ควรอ่าน
        # (เดิมใช้ "พื้นที่หมึกเยอะกว่าชนะ" เป็นหลัก ซึ่งพิสูจน์จาก 348 ภาพจริงแล้วว่าอ่อนกว่าชัดเจน เพราะ
        # เลขแต่ละตัวมีปริมาณหมึกไม่เท่ากันโดยธรรมชาติ — เลข "1" มีหมึกน้อยกว่าเลข "8" มาก การเอาพื้นที่หมึก
        # ครึ่งบนไปแข่งครึ่งล่างจึงลำเอียงเข้าหาเลขอ้วนเสมอ ไม่ว่าเลขนั้นจะโผล่มาเต็มกรอบจริงหรือไม่
        # วัดจริงระดับ slot: พื้นที่หมึก 237/299 = 79.3% -> centroid 250/299 = 83.6%
        # วัดจริงระดับภาพทั้งชุด: 277/348 = 79.6% -> 285/348 = 81.9%)
        # พื้นที่หมึกถูกลดบทบาทเป็นตัวตัดสินรองเฉพาะตอนระยะ centroid เท่ากันพอดีเท่านั้น
        d_top = abs(top_c - center); d_bottom = abs(bottom_c - center)
        if STRADDLE_WINNER_METHOD == "area":
            # โหมดเปรียบเทียบ (ไม่ใช่ค่าเริ่มต้น) — ตัดสินด้วยพื้นที่หมึกเยอะกว่าชนะล้วนๆ ตามที่เคยใช้ก่อน
            # เปลี่ยนมาใช้ centroid วัดผลจริงแล้วว่า centroid แม่นกว่า (79.6% vs 81.9% ที่ระดับภาพ, ไล่ขึ้น
            # ไปถึง 88.2%+ หลังปรับจุดอื่นต่อ) เก็บโหมดนี้ไว้ให้ TEST_app.py เรียกรันเปรียบเทียบได้เท่านั้น
            top_wins = top_area >= bottom_area
        elif d_top != d_bottom:
            top_wins = d_top < d_bottom
        else:
            top_wins = top_area >= bottom_area
        debug["winner"] = "top" if top_wins else "bottom"
        value = v if top_wins else v_next
        return value, True, is_90, value == 0, debug
    except Exception:
        return v, True, is_90, False, debug

def apply_carry_logic_v2(slots):
    """slots: left-to-right list of {"value", "is_straddle", "is_90_straddle", "resolved_is_zero"}.
    Only a straddle caught exactly between 9 and 0, resolved as 0, ticks the digit to its left —
    matching real odometer/counter mechanics (a wheel only advances its neighbor on a 9->0 rollover).
    slot ที่ "ตัวเองกำลังเหลื่อมอยู่" ถือว่ากำลังหมุนพร้อมกับตัวขวาอยู่แล้ว ค่าที่ _resolve_digit_slot ตัดสินได้
    จึงเป็นค่าที่หมุนไปแล้วในตัว — ห้ามเอา carry ที่วิ่งเข้ามาบวกซ้ำอีก มิฉะนั้นจะเกินไป 1
    (เดิมกันไว้เฉพาะกรณีที่ slot นั้นเป็นเลขเหลื่อม 9/0 ที่ตัดสินได้ 0 ซึ่งเป็นแค่กรณีย่อยกรณีเดียวของหลักการนี้
     ทำให้พลาดกรณีทั่วไป เช่น TEST_181 หลักที่ 3 เหลื่อมคู่ (1,2) ตัดสินได้ 2 ถูกแล้ว แต่ carry จากหลักที่ 4
     ที่หมุน 9->0 ไปบวกซ้ำเป็น 3 ได้ 0430 ทั้งที่ค่าจริงคือ 0420 — วัดจริง 348 ภาพ: เพดานความแม่นยำ
     ขยับจาก 93.4% เป็น 96.8% และความแม่นยำจริงจาก 81.9% เป็น 83.0%)
    ส่วน carry ที่ "ส่งต่อ" ออกไปทางซ้ายยังคงเกิดเฉพาะตอนหมุนผ่าน 9->0 จริงเท่านั้น ตามกลไกเฟืองจริง"""
    final_digits = []
    carry = False
    for slot in reversed(slots):
        rolled = slot["is_90_straddle"] and slot["resolved_is_zero"]
        val = slot["value"]
        if carry and not slot.get("is_straddle", False): val = (val + 1) % 10
        final_digits.insert(0, str(val))
        carry = rolled
    return "".join(final_digits)

# ============================================================================
# SECTION 4: Digit Slot Detection (YOLO digits.pt)
# ============================================================================
def _scan_digits(screen_crop):
    """16-angle digit detection scan on a single screen crop.
    digits.pt ตรวจจับทั้ง "slot" ตรงๆ อยู่แล้ว — แม้เลขกำลังเหลื่อมอยู่ (ครึ่งบน-ล่างคนละเลข) ก็ยังได้กรอบเดียว
    ครอบทั้งคู่ (พิสูจน์แล้วจากภาพจริง: raw boxes เท่ากับ 4 พอดีตอนมีเลขเหลื่อมด้วย) จึงไม่ต้องมีขั้นตอนรวมกรอบ
    ซ้อนคอลัมน์หรือแบ่ง bucket ตำแหน่งอีก — ใช้กรอบดิบจากโมเดลเป็น 1 slot/หลักตรงๆ (เดิมมีขั้นตอนพวกนี้แต่พบว่า
    ไม่จำเป็นและเสี่ยงรวม/แบ่งกรอบผิดจนล้นข้ามหลักจริง) เลือกมุมที่เจอกรอบเยอะที่สุด (เจอครบ 4 หยุดทันที)
    Returns (boxes_sorted, best_screen) — boxes_sorted เรียงซ้าย→ขวาแล้ว"""
    best_boxes = np.empty((0, 4)); best_screen = screen_crop
    for angle in range(16):
        rotated_screen = screen_crop if angle == 0 else cv2.warpAffine(
            screen_crop, cv2.getRotationMatrix2D((screen_crop.shape[1]//2, screen_crop.shape[0]//2), angle, 1.0),
            (screen_crop.shape[1], screen_crop.shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        res_n = model_yolo_digits.predict(source=rotated_screen, conf=0.5, iou=0.5, verbose=False)
        boxes_raw = res_n[0].boxes.xyxy.cpu().numpy()

        if len(boxes_raw) > len(best_boxes):
            best_boxes = boxes_raw; best_screen = rotated_screen
        if len(boxes_raw) == 4: break

    if len(best_boxes) > 0:
        best_boxes = best_boxes[np.argsort(best_boxes[:, 0])]
    return best_boxes, best_screen

# ============================================================================
# SECTION 5: Visualization Helpers
# ============================================================================
def _build_summary_panel(viz_width, summary_lines, bg=(32, 32, 32)):
    """แถบสรุปท้ายภาพ (ต่อด้านล่าง viz_image) — สรุปข้อความว่าภาพนี้ตรวจจับอะไรได้บ้างในแต่ละขั้นตอน"""
    pad = 12
    line_h = 22
    panel_h = pad * 2 + line_h * len(summary_lines)
    panel = np.full((panel_h, viz_width, 3), bg, dtype=np.uint8)
    y = pad + 16
    for line in summary_lines:
        cv2.putText(panel, line, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        y += line_h
    return panel

def _draw_pipeline_visualization(rotated_image, bbox_m_rot, boxes_rot, clss_rot, confs_rot,
                                  best_candidate, slot_results, final_reading,
                                  meter_found, dial_found, screen_found):
    """วาดภาพอธิบายทั้ง pipeline ต่อจาก rotated_image เรียงตามลำดับขั้นตอนจริง:
    STEP1=Meter/Dial(จุดอ้างอิงหมุนภาพ), STEP4=Screen, STEP6=Digits, STEP7=แถบสรุปท้ายภาพ
    Blue=Meter, Red=Dial(ref เท่านั้น), Green=Screen, Yellow=Digits. คืน viz_image พร้อม reading_text"""
    viz_image = rotated_image.copy()

    # STEP1: Meter (สีน้ำเงิน)
    if bbox_m_rot is not None:
        cv2.rectangle(viz_image, (int(bbox_m_rot[0]), int(bbox_m_rot[1])), (int(bbox_m_rot[2]), int(bbox_m_rot[3])), (255, 0, 0), 2)
        cv2.putText(viz_image, "STEP1: Meter", (int(bbox_m_rot[0]), max(20, int(bbox_m_rot[1])-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    # STEP1: Dial (สีแดง) — ใช้เป็นจุดอ้างอิงหมุนภาพเท่านั้น ไม่เกี่ยวกับการตัดสินค่าเลขอีกต่อไป
    b_dial = _best_box(boxes_rot, clss_rot, confs_rot, 1, max_area_ratio=0.10, image_area=rotated_image.shape[0] * rotated_image.shape[1])
    if b_dial is not None:
        cv2.rectangle(viz_image, (int(b_dial[0]), int(b_dial[1])), (int(b_dial[2]), int(b_dial[3])), (0, 0, 255), 2)
        cv2.putText(viz_image, "STEP1: Dial(ref)", (int(b_dial[0]), max(20, int(b_dial[1])-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # STEP4: Screen (สีเขียว)
    sx1, sy1 = 0, 0
    if best_candidate is not None and bbox_m_rot is not None:
        sx1 = int(best_candidate[0]) + int(bbox_m_rot[0])
        sy1 = int(best_candidate[1]) + int(bbox_m_rot[1])
        sx2 = int(best_candidate[2]) + int(bbox_m_rot[0])
        sy2 = int(best_candidate[3]) + int(bbox_m_rot[1])
        cv2.rectangle(viz_image, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)
        cv2.putText(viz_image, "STEP4: Screen", (sx1, max(20, sy1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # STEP6: Digits (สีเหลือง) — วาดกรอบพร้อมเลขที่สรุปได้ของแต่ละหลัก (หลัง carry logic) กำกับไว้ด้านบนกรอบ
    if slot_results and sx1 > 0:
        for s in slot_results:
            b = s["box"]
            dx1, dy1 = int(b[0]) + sx1, int(b[1]) + sy1
            dx2, dy2 = int(b[2]) + sx1, int(b[3]) + sy1
            cv2.rectangle(viz_image, (dx1, dy1), (dx2, dy2), (0, 255, 255), 2)
            label = s.get("final_char", str(s["value"]))
            label_y = max(20, dy1 - 8)
            cv2.putText(viz_image, label, (dx1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
            cv2.putText(viz_image, label, (dx1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

            # เลขเหลื่อม: วาดกรอบ top/bottom (สีต่างกัน) ซ้อนในกรอบเหลือง — เส้นหนา=ตัวที่ชนะ(area มากกว่า), เส้นบาง=ตัวที่แพ้
            split = s.get("split")
            if s.get("is_straddle") and split and split.get("valley_row") is not None:
                px1, py1, px2, py2 = s["pad_box"]
                valley = split["valley_row"]
                vx1, vx2 = px1 + sx1, px2 + sx1
                vy_top, vy_mid, vy_bot = py1 + sy1, py1 + valley + sy1, py2 + sy1
                top_wins = split.get("winner") == "top"
                top_color, bottom_color = (255, 0, 255), (0, 165, 255)  # magenta=top, orange=bottom
                cv2.rectangle(viz_image, (vx1, vy_top), (vx2, vy_mid), top_color, 3 if top_wins else 1)
                cv2.rectangle(viz_image, (vx1, vy_mid), (vx2, vy_bot), bottom_color, 3 if not top_wins else 1)
                cv2.putText(viz_image, str(split.get("top_value")), (vx2 + 3, vy_top + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, top_color, 1)
                cv2.putText(viz_image, str(split.get("bottom_value")), (vx2 + 3, vy_bot - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, bottom_color, 1)

    # STEP: Final Reading (เลขผลลัพธ์ที่อ่านได้ — เขียนกำกับไว้บนภาพให้เห็นชัด)
    reading_text = f"READING: {final_reading:04d}" if final_reading is not None else "READING: FAILED"
    text_y = viz_image.shape[0] - 20
    cv2.putText(viz_image, reading_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
    cv2.putText(viz_image, reading_text, (10, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    # STEP7: แถบสรุปท้ายภาพ — สรุปว่าแต่ละขั้นตอนตรวจจับอะไรได้บ้าง
    try:
        straddle_count = sum(1 for s in slot_results if s.get("is_straddle"))
        # หมายเหตุ: ทุกบรรทัดในนี้ถูกวาดลงภาพจริงด้วย cv2.putText ซึ่งไม่รองรับฟอนต์ไทย (จะกลายเป็น "??????")
        # จึงต้องใช้ภาษาอังกฤษเท่านั้น — ให้เหมือนกับ label อื่นบนภาพทั้งหมดในไฟล์นี้ (คอมเมนต์โค้ดยังใช้ไทยได้ปกติ)
        summary_lines = [
            f"Meter: {'FOUND' if meter_found else 'not found'}   Dial(ref): {'FOUND' if dial_found else 'not found'}   Screen: {'FOUND' if screen_found else 'not found'}",
            f"Digits detected: {len(slot_results)}/4   Straddling: {straddle_count}",
            reading_text,
        ]
        summary_panel = _build_summary_panel(viz_image.shape[1], summary_lines)
        viz_image = np.vstack([viz_image, summary_panel])
    except Exception: pass

    return viz_image

# ============================================================================
# SECTION 6: Debug Image / Result File I/O
# ============================================================================
def _save_debug_images(clean_name, images):
    """เซฟภาพ debug ทุกภาพ — ถ้า key มี '/' (เช่น "sub/name") จะแยกไปเก็บในโฟลเดอร์ย่อยของ debug/<clean_name>/
    ให้เอง (เผื่อในอนาคตอยากแยกกลุ่มไฟล์ที่เกี่ยวข้องกันออกจากกอง debug หลัก)"""
    out_dir = os.path.join(CURRENT_OUT_DIR, "debug", clean_name)
    try:
        os.makedirs(out_dir, exist_ok=True)
        for name, img in images.items():
            try:
                if img is not None and img.size > 0:
                    full_path = os.path.join(out_dir, f"{name}.jpg")
                    os.makedirs(os.path.dirname(full_path), exist_ok=True)
                    cv2.imwrite(full_path, img)
            except Exception: pass
    except Exception: pass

def _save_pipeline_debug_outputs(clean_name, image, rotated_image, rotate_deg, meter_before, meter_img, screen_img,
                                  screen_candidates_debug, digit_overview_img, slot_results,
                                  digit_crops_debug, viz_image, screen_source, final_reading,
                                  current_pattern, all_ids, detect_before_rotate=None, detect_after_rotate=None):
    """รวมทุกอย่างที่ต้องเซฟลง debug/<clean_name>/ ของภาพนี้ไว้ที่เดียว: ภาพทุกขั้นตอน + result.txt สรุปผล
    เรียงลำดับไฟล์ตามลำดับการทำงานจริงของ pipeline (ตัวเลขนำหน้าไฟล์ = ลำดับขั้นตอน) ทุกจุดที่มีการ "ตัดภาพ"
    (crop) จะมีคู่ก่อนตัด (มีกรอบสี่เหลี่ยมวาดบอกตำแหน่งที่จะตัด) กับหลังตัด (ผลลัพธ์ที่ตัดออกมาจริง) เสมอ
    (ครอบด้วย try/except pass — debug I/O ล้มเหลวต้องไม่ทำให้ทั้ง pipeline พัง)"""
    try:
        rotated_labeled = _label_crop(rotated_image, f"rotated {rotate_deg:.1f} deg", color=(0, 255, 0))
        debug_images = {
            "00_original": image,          # ภาพต้นฉบับก่อนทำอะไรเลย
        }
        # 00b/00c: กรอบ Meter/Dial(เข็ม)/Screen ที่ตรวจจับได้ ก่อน-หลังหมุนภาพ (Step 1) — เห็นเปรียบเทียบกัน
        # ว่าหมุนไปแล้วกรอบขยับไปตรงไหน คนละจุดประสงค์กับ 02a/02b (ตัด crop จริง) — นี่แค่วาดกรอบทับดูเฉยๆ
        if detect_before_rotate is not None: debug_images["00b_detect_before_rotate"] = detect_before_rotate
        if detect_after_rotate is not None: debug_images["00c_detect_after_rotate"] = detect_after_rotate
        debug_images["01_rotated"] = rotated_labeled  # หลัง Step 1-2: หมุนภาพให้ตั้งตรงแล้ว (ยังไม่ตัด) — กำกับมุมที่หมุนไว้มุมซ้ายบน
        # Step 3: ตัด Meter ออกจาก rotated_image — คู่ก่อน/หลังตัด
        if meter_before is not None: debug_images["02a_meter_before"] = meter_before
        debug_images["02b_meter_after"] = meter_img
        # Step 4: ตัด Screen 2 รอบซ้อนจาก meter_img — คู่ก่อน/หลังตัดของทั้ง 2 รอบ (03a/03b, 03c/03d)
        # เก็บเฉพาะกรอบที่ระบบเลือกใช้จริงเท่านั้น (ดู _step4_detect_screen — กรอบผู้สมัครอื่นไม่ถูกบันทึก)
        debug_images.update(screen_candidates_debug)
        debug_images["03e_screen_final"] = screen_img  # screen_img สุดท้ายที่ pipeline ใช้ต่อจริง (ซ้ำกับ 03b/03d เผื่อรอบ 2 หาไม่เจอ)

        # Step 5: ก่อนตัดหลักเลขทั้ง 4 — ภาพรวมกรอบทุกหลักบน screen_img (คู่ "ก่อนตัด" ของหลักทั้งชุด)
        debug_images["04a_digits_before"] = digit_overview_img
        for i, (s, dbg) in enumerate(zip(slot_results, digit_crops_debug), start=1):
            # 04b: ผลจำแนก "ดิบ" จากตัวจำแนกก่อนตัดสินใจใดๆ — โชว์คลาสเลขเหลื่อม/เลขนิ่งที่ทายได้ตรงๆ ก่อนเสมอ
            base_id = s.get("base_id")
            conf_pct = f" conf={s['confidence']*100:.0f}%" if "confidence" in s else ""
            low_conf_warn = " ⚠LOW-CONF" if s.get("low_confidence") else ""
            if s["is_straddle"]:
                pair = f"({base_id // 2},{(base_id // 2 + 1) % 10})"
                raw_tag = f"cls{base_id} straddle pair{pair}{conf_pct}{low_conf_warn}"
            else:
                raw_tag = f"cls{base_id} clean={base_id // 2}{conf_pct}{low_conf_warn}"
            debug_images[f"04b_slot{i}_raw_{raw_tag}"] = _label_crop(dbg["crop"], raw_tag)
            # 04b_x1-x4: ขั้นตอนขาวดำของ _split_straddle_crop (grayscale/CLAHE/adaptive_inkmask/valley+centroid)
            # เรียงต่อจาก 04b_raw ทันที (ก่อน 04c_top) มีเฉพาะ slot ที่เป็นเลขเหลื่อมเท่านั้น (dict มีคีย์
            # เหล่านี้ก็ต่อเมื่อ _split_straddle_crop ทำงานสำเร็จ ไม่ throw exception กลางทาง)
            for xkey in ("x1_grayscale", "x2_clahe", "x3_adaptive_inkmask", "x4_valley_and_centroid"):
                if xkey in dbg:
                    debug_images[f"04b_slot{i}_{xkey}"] = dbg[xkey]
            # 04c/04d: ครึ่งบน/ล่างที่แยกออกมาตัดสิน (เฉพาะเลขเหลื่อม)
            if "top" in dbg:
                top_tag = f"top={dbg.get('top_value')}" + (" WIN" if dbg.get("winner") == "top" else "")
                debug_images[f"04c_slot{i}_top"] = _label_crop(dbg["top"], top_tag, color=(255, 0, 255))
            if "bottom" in dbg:
                bottom_tag = f"bottom={dbg.get('bottom_value')}" + (" WIN" if dbg.get("winner") == "bottom" else "")
                debug_images[f"04d_slot{i}_bottom"] = _label_crop(dbg["bottom"], bottom_tag, color=(0, 165, 255))
            # 04e: ผลลัพธ์สุดท้ายหลังตัดสินใจ (หลังกฎกลเข้าหาตรงกลาง+carry) — มาทีหลังสุดเสมอ ตามลำดับดูง่าย
            label = s.get("final_char", str(s["value"]))
            debug_images[f"04e_slot{i}_final_val{label}"] = _label_crop(dbg["crop"], f"final={label}", color=(0, 255, 0))
        debug_images["99_annotated"] = viz_image  # ภาพสรุปทุกขั้นตอนซ้อนกัน + แถบสรุปท้ายภาพ
        _save_debug_images(clean_name, debug_images)

        result_path = os.path.join(CURRENT_OUT_DIR, "debug", clean_name, "result.txt")
        with open(result_path, "w", encoding="utf-8") as rf:
            reading_str = f"{final_reading:04d}" if final_reading is not None else "อ่านไม่สำเร็จ (FAILED)"
            straddle_positions = [i + 1 for i, s in enumerate(slot_results) if s.get("is_straddle")]

            # ============================================================
            # ส่วนที่ 1: สรุปแบบเข้าใจง่าย (อ่านก่อน ไม่ต้องรู้ศัพท์เทคนิคก็เข้าใจได้)
            # ============================================================
            rf.write("=" * 70 + "\n")
            rf.write("สรุปผลอ่านมิเตอร์ (ฉบับเข้าใจง่าย)\n")
            rf.write("=" * 70 + "\n")
            rf.write(f"ไฟล์ภาพ: {clean_name}\n")
            rf.write(f"ผลลัพธ์ที่อ่านได้: {reading_str}\n")
            rf.write(f"ตรวจพบหลักตัวเลขครบ 4 หลักหรือไม่: {'ครบ (4/4)' if current_pattern == '1111' else f'ไม่ครบ (แพทเทิร์น {current_pattern})'}\n")
            low_conf_positions = [i + 1 for i, s in enumerate(slot_results) if s.get("low_confidence")]
            if low_conf_positions:
                th_pos_lc = {1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่"}
                rf.write(f"⚠ คำเตือน: หลักที่ {', '.join(th_pos_lc.get(p, str(p)) for p in low_conf_positions)} "
                         f"ตัวจำแนกลังเลระหว่าง 2 คลาส (คะแนนใกล้กันมาก) ควรตรวจสอบด้วยตาอีกครั้ง — "
                         f"ดูรายละเอียดที่ส่วนที่ 2 ด้านล่าง\n")
            if not straddle_positions:
                rf.write("เลขเหลื่อม (เลข 2 ตัวซ้อนกันเพราะล้อกำลังหมุนตอนถ่ายภาพ): ไม่มีเลย ทุกหลักเป็นเลขนิ่งชัดเจน\n")
            else:
                th_pos = {1: "หนึ่ง", 2: "สอง", 3: "สาม", 4: "สี่"}
                rf.write(f"เลขเหลื่อม: พบ {len(straddle_positions)} หลัก ที่ตำแหน่งที่ {', '.join(str(p) for p in straddle_positions)} (นับจากซ้าย)\n")
                for i, s in enumerate(slot_results, start=1):
                    if not s.get("is_straddle"): continue
                    sp = s.get("split") or {}
                    top_v, bot_v = sp.get("top_value"), sp.get("bottom_value")
                    final_v = s.get("value")
                    if s.get("gear_forced"):
                        why = (f"เห็นเลข {top_v} กับ {bot_v} ซ้อนกัน — ระบบไม่ได้วัดจากรูปทรงเอง แต่ใช้ 'กฎกลเข้าหาตรงกลาง' "
                               f"บังคับแทน เพราะหลักถัดไปทางขวากำลังพลิกจาก 9 เป็น 0 พอดี แปลว่าหลักนี้ต้องกำลังหมุนอยู่แน่นอน "
                               f"จึงเลือก {final_v} (ค่าที่หมุนไปแล้ว) โดยไม่ต้องเดาจากรูปภาพเลย")
                    elif sp.get("winner"):
                        winner_v = top_v if sp.get("winner") == "top" else bot_v
                        loser_v = bot_v if sp.get("winner") == "top" else top_v
                        rows = ["บน", "ล่าง"] if sp.get("winner") == "top" else ["ล่าง", "บน"]
                        why = (f"เห็นเลข {top_v} กับ {bot_v} ซ้อนกัน — วัดแล้วเลข {winner_v} (ฝั่ง{rows[0]}ของภาพ) "
                               f"โผล่อยู่กึ่งกลางกรอบมากกว่าเลข {loser_v} (ฝั่ง{rows[1]}) จึงเลือก {winner_v}")
                    else:
                        why = f"เห็นเลข {top_v} กับ {bot_v} ซ้อนกัน แต่แยกวิเคราะห์รูปทรงไม่สำเร็จ จึงใช้ค่าเริ่มต้น {final_v}"
                    rf.write(f"  - หลักที่{th_pos.get(i, i)}: {why}\n")
            rf.write("\n")

            # ============================================================
            # ส่วนที่ 2: รายละเอียดทีละหลัก ซ้าย -> ขวา (ละเอียดขึ้น เห็นค่าดิบคู่กับค่าตัดสินแล้ว)
            # ============================================================
            rf.write("=" * 70 + "\n")
            rf.write("รายละเอียดทีละหลัก (ซ้าย -> ขวา)\n")
            rf.write("=" * 70 + "\n")
            for i, s in enumerate(slot_results, start=1):
                base_id = s.get("base_id")
                conf_str = ""
                if "confidence" in s:
                    conf_str = (f" [ความมั่นใจ: คลาส{base_id}={s['confidence']*100:.1f}% vs "
                                f"คลาส{s['id2']}={s['confidence2']*100:.1f}%"
                                f"{' — ⚠ลังเลใกล้กันมาก' if s.get('low_confidence') else ''}]")
                if s.get("is_straddle"):
                    pair = (base_id // 2, (base_id // 2 + 1) % 10)
                    decided_by = "กฎกลเข้าหาตรงกลาง (บังคับ)" if s.get("gear_forced") else "ระยะ centroid ถึงกึ่งกลางกรอบ"
                    rf.write(f"หลักที่ {i}: คลาสดิบ={base_id} (เลขเหลื่อม คู่ตัวเลือก {pair[0]}/{pair[1]}) "
                             f"-> ตัดสินโดย: {decided_by} -> ค่าสุดท้าย = {s['value']}{conf_str}\n")
                else:
                    rf.write(f"หลักที่ {i}: คลาสดิบ={base_id} (เลขนิ่ง ไม่ต้องตัดสินใจ) -> ค่าสุดท้าย = {s['value']}{conf_str}\n")
            rf.write("\n")

            # ============================================================
            # ส่วนที่ 3: ข้อมูลดิบ/เทคนิค (สำหรับ debug เจาะลึก)
            # ============================================================
            rf.write("=" * 70 + "\n")
            rf.write("ข้อมูลดิบ/เทคนิค\n")
            rf.write("=" * 70 + "\n")
            rf.write(f"Rotated By: {rotate_deg:.2f} degrees (0 = ไม่ได้หมุนเลย)\n")
            rf.write(f"Screen Source (round1=ขยาย2.5เท่า, round2=เจอซ้ำไม่ขยาย): {screen_source}\n")
            rf.write(f"Final Reading: {final_reading if final_reading is not None else 'FAILED'}\n")
            rf.write(f"Pattern (4 slots detected): {current_pattern}\n")
            rf.write(f"Slot Values (left->right): {[s['value'] for s in slot_results]}\n")
            rf.write(f"Straddle Slots (any pair): {straddle_positions}\n")
            rf.write(f"Straddle Slots (9<->0 rollover): {[i+1 for i, s in enumerate(slot_results) if s['is_90_straddle']]}\n")
            rf.write(f"Gear-rule Forced Slots: {[i+1 for i, s in enumerate(slot_results) if s.get('gear_forced')]}\n")
            # all_ids = ค่าที่ "ตัดสินแล้ว" เข้ารหัสเป็น value*2+parity ไม่ใช่ class id ดิบจากตัวจำแนก
            # (ป้ายเดิมเขียนว่า "Raw Class IDs" ซึ่งผิด และเคยทำให้ตีความผลวิเคราะห์ผิดมาแล้ว)
            rf.write(f"Resolved IDs (all_ids = value*2+parity): {all_ids}\n")
            rf.write(f"Raw Class IDs from classifier: {[s.get('base_id') for s in slot_results]}\n")
            # Low-confidence: margin (conf top1 - conf top2) < LOW_CONFIDENCE_MARGIN — แค่ธงเตือน ไม่ได้แก้คำตอบ
            # เอง (ดู docstring _classify_crops) — ยังไม่มีขั้นตอนไหนใน pipeline ตรวจสอบย้อนกลับความถูกต้องของ
            # base_id เลย ถ้าทายผิดตั้งแต่ต้น ทุกอย่างหลังจากนี้เดินหน้าผิดทั้งสาย ธงนี้ไว้ดักจับเคสเสี่ยงเท่านั้น
            rf.write(f"Low Confidence Slots (margin < {LOW_CONFIDENCE_MARGIN:.2f}): {[i+1 for i, s in enumerate(slot_results) if s.get('low_confidence')]}\n")
            conf_pairs = []
            for s in slot_results:
                if "confidence" in s:
                    conf_pairs.append(f"{s['confidence']*100:.0f}/{s['confidence2']*100:.0f}")
                else:
                    conf_pairs.append("n/a")
            rf.write(f"Classifier Confidence per slot (top1%/top2%): {conf_pairs}\n")
    except Exception: pass

# ============================================================================
# SECTION 7: Pipeline Steps (แต่ละขั้นตอนของ read_water_meter แยกเป็นฟังก์ชันย่อย)
# ============================================================================
def _step1_find_orientation_refs(image):
    """[Step 1] ตรวจหา Meter+Dial บนภาพต้นฉบับ เพื่อใช้เป็นจุดอ้างอิงหมุนภาพ
    คืน (boxes, clss, confs, bbox_m, bbox_d, meter_found, dial_found)"""
    res1 = model_yolo_main.predict(source=image, conf=0.7, classes=[0, 1], verbose=False)
    boxes = res1[0].boxes.xyxy.cpu().numpy(); clss = res1[0].boxes.cls.cpu().numpy(); confs = res1[0].boxes.conf.cpu().numpy()
    image_area = image.shape[0] * image.shape[1]
    bbox_m = _meter_or_promoted_dial(boxes, clss, confs, image_area)
    # Dial ตัวจริงเป็นหน้าปัดเข็มเล็กๆ ไม่ควรกินพื้นที่เกิน ~10% ของภาพ — กันกรณีโมเดลจำแนกวงมิเตอร์ทั้งวงผิดเป็น Dial
    bbox_d = _best_box(boxes, clss, confs, 1, max_area_ratio=0.10, image_area=image_area)
    return boxes, clss, confs, bbox_m, bbox_d, bbox_m is not None, bbox_d is not None

def _rotate_by_reference(image, bbox_m, bbox_d):
    """หมุนภาพให้ตั้งตรงโดยใช้ Meter/Dial เป็นจุดอ้างอิงหลัก ถ้าขาดตัวใดตัวหนึ่งไปใช้ Screen ช่วยหามุมแทน
    คืน (rotated_image, rotate_deg) — rotate_deg คือมุมที่หมุนไปจริง (0.0 ถ้าไม่ได้หมุนเลย เช่นไม่เจอจุด
    อ้างอิงอะไรเลย) ใช้แปะกำกับไว้บนภาพ debug (01_rotated, 02a_meter_before) ให้เห็นว่าหมุนไปเท่าไหร่"""
    if bbox_d is not None and bbox_m is not None:
        # เจอครบทั้งคู่ — หมุนโดยใช้ Meter->Dial เป็นแกนอ้างอิง (Dial ไปอยู่ขวา 0 องศา)
        cx_m, cy_m = (bbox_m[0]+bbox_m[2])/2, (bbox_m[1]+bbox_m[3])/2
        cx_d, cy_d = (bbox_d[0]+bbox_d[2])/2, (bbox_d[1]+bbox_d[3])/2
        rotated_image, rotate_deg = _rotate_to_target(image, (cx_m, cy_m), (cx_d, cy_d), 0)
        return rotated_image, rotate_deg

    # เจอแค่ Meter หรือ Dial อย่างเดียว — หา Screen มาช่วยกำหนดการหมุนแทน
    res_scr = model_yolo_main.predict(source=image, conf=0.1, classes=[2], verbose=False)
    scr_boxes = res_scr[0].boxes.xyxy.cpu().numpy()
    scr_confs = res_scr[0].boxes.conf.cpu().numpy()
    if len(scr_boxes) == 0:
        return image, 0.0

    si = int(np.argmax(scr_confs))
    bbox_s = scr_boxes[si]
    cx_s, cy_s = (bbox_s[0]+bbox_s[2])/2, (bbox_s[1]+bbox_s[3])/2
    if bbox_m is not None:
        # เจอแต่ Meter — ใช้ Meter เป็นจุดหมุน หมุนให้ Screen อยู่ข้างบนตรงๆ (90 องศา)
        cx_p, cy_p = (bbox_m[0]+bbox_m[2])/2, (bbox_m[1]+bbox_m[3])/2
        rotated_image, rotate_deg = _rotate_to_target(image, (cx_p, cy_p), (cx_s, cy_s), 90)
    else:
        # เจอแต่ Dial — ใช้ Dial เป็นจุดหมุน หมุนให้ Screen อยู่ด้านซ้าย (135 องศา)
        cx_p, cy_p = (bbox_d[0]+bbox_d[2])/2, (bbox_d[1]+bbox_d[3])/2
        rotated_image, rotate_deg = _rotate_to_target(image, (cx_p, cy_p), (cx_s, cy_s), 135)
    return rotated_image, rotate_deg

def _step2_3_detect_and_crop_meter(rotated_image, bbox_m):
    """[Step 2] ตรวจ Meter+Dial ซ้ำบนภาพที่หมุนแล้ว (แม่นขึ้นเพราะภาพตั้งตรง) แล้ว [Step 3] crop เฉพาะ Meter
    คืน (boxes_rot, clss_rot, confs_rot, bbox_m_rot, meter_img)"""
    res_rot = model_yolo_main.predict(source=rotated_image, conf=0.7, classes=[0, 1], verbose=False)
    boxes_rot = res_rot[0].boxes.xyxy.cpu().numpy()
    clss_rot = res_rot[0].boxes.cls.cpu().numpy()
    confs_rot = res_rot[0].boxes.conf.cpu().numpy()
    bbox_m_rot = _meter_or_promoted_dial(boxes_rot, clss_rot, confs_rot, rotated_image.shape[0] * rotated_image.shape[1])

    meter_img = rotated_image
    if bbox_m_rot is not None:
        mx1, my1, mx2, my2 = map(int, bbox_m_rot)
        meter_img = rotated_image[max(0, my1):my2, max(0, mx1):mx2]
    elif bbox_m is not None:
        mx1, my1, mx2, my2 = map(int, bbox_m)
        meter_img = rotated_image[max(0, my1):my2, max(0, mx1):mx2]

    return boxes_rot, clss_rot, confs_rot, bbox_m_rot, meter_img

def _draw_box_before(source_img, box, tag, color=(0, 255, 0)):
    """คืนสำเนาของ source_img พร้อมกรอบสี่เหลี่ยม + ป้ายกำกับวาดทับไว้ ณ ตำแหน่ง box —
    ใช้เป็นภาพ "ก่อนตัด" (before-crop) คู่กับภาพ "หลังตัด" (crop จริง) ในทุกจุดที่ตัดภาพในไฟล์นี้
    เพื่อให้ debug เห็นชัดว่าระบบเล็งตัดตรงไหนของภาพต้นทาง ก่อนจะได้ผลลัพธ์ที่ตัดแล้วมา"""
    vis = source_img.copy()
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
    cv2.putText(vis, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(vis, tag, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    return vis

def _draw_angle_diagram(vis, center, rotate_deg, radius=None):
    """วาดไดอะแกรมมุมที่จุด center: เส้นอ้างอิงแนวนอน (เทา, มุม 0), เส้นบอกมุมที่หมุนไปจริง (เหลือง),
    เส้นโค้ง (arc) เชื่อมระหว่าง 2 เส้น พร้อมตัวเลของศากำกับ — ให้เห็นเป็นภาพ "เอียงจากแนวตั้งฉาก/แนวนอน
    ไปกี่องศา" ไม่ใช่แค่ตัวเลขข้อความเฉยๆ
    ทิศทาง: ใช้ธรรมเนียมเดียวกับ cv2.getRotationMatrix2D ที่ _rotate_to_target เรียกใช้จริง — มุมบวก
    (rotate_deg > 0) หมุนทวนเข็มนาฬิกาในพิกัดภาพปกติ (แกน y ปกติ ไม่ใช่แกน y ของภาพที่กลับหัว) จึงวาดเส้น
    ที่หมุนไปด้วยการคำนวณ (cos, -sin) กัน y ของภาพที่ชี้ลงกลับทิศให้ตรงกับธรรมเนียมนั้น
    ถ้าไม่ระบุ radius จะคำนวณตามขนาดภาพจริงแทนค่าคงที่ตายตัว (ภาพถ่ายมือถือมักมีความละเอียดสูงมาก เช่น
    2268x4032 — เส้นผ่าศูนย์กลางคงที่ 90px แทบมองไม่เห็น) ให้เป็นสัดส่วนกับด้านสั้นของภาพแทน กันดูเล็กเกินไป"""
    if radius is None:
        radius = max(60, int(min(vis.shape[0], vis.shape[1]) * 0.12))
    thick = max(2, radius // 45)
    font_scale = max(0.7, radius / 130)
    cx, cy = int(center[0]), int(center[1])
    rad = math.radians(rotate_deg)
    ex, ey = int(cx + radius * math.cos(rad)), int(cy - radius * math.sin(rad))
    cv2.circle(vis, (cx, cy), max(4, thick + 2), (255, 255, 255), -1)
    cv2.line(vis, (cx - radius, cy), (cx + radius, cy), (160, 160, 160), thick)   # เส้นอ้างอิง 0 องศา (เทา)
    cv2.line(vis, (cx, cy), (ex, ey), (0, 215, 255), thick + 1)                   # เส้นมุมที่หมุนจริง (เหลือง)
    cv2.ellipse(vis, (cx, cy), (int(radius * 0.6), int(radius * 0.6)), 0, -rotate_deg, 0, (0, 215, 255), thick)
    label = f"{rotate_deg:+.1f} deg"
    lx, ly = cx + radius + 15, cy
    cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thick + 3)
    cv2.putText(vis, label, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 215, 255), thick)
    return vis

def _draw_orientation_detections(image, bbox_m, bbox_d, bbox_s, stage_tag, rotate_deg=None):
    """วาดกรอบ Meter (น้ำเงิน) / Dial=เข็ม (แดง) / Screen (เขียว) ทับบน image ภาพเดียว — ใช้ทำภาพ debug
    คู่ "ก่อนหมุน/หลังหมุน" (00b/00c) แสดงว่าตอน Step 1 ตรวจจับอะไรได้บ้างจากภาพจริง ก่อนที่จะรู้ผลลัพธ์
    การหมุนเลยด้วยซ้ำ — ใช้สีเดียวกับ _draw_pipeline_visualization (STEP1/STEP4) เพื่อให้อ่านสอดคล้องกัน
    ทั้งระบบ กรอบไหนหาไม่เจอ (None) แค่ข้ามไป ไม่วาด ไม่ error
    stage_tag: ข้อความภาษาอังกฤษกำกับมุมซ้ายบนของภาพ เช่น "BEFORE ROTATE" / "AFTER ROTATE" (cv2.putText
    ไม่รองรับฟอนต์ไทย เหมือน label อื่นทั้งหมดในไฟล์นี้)
    rotate_deg: ถ้าระบุ (ไม่ใช่ None) จะวาดไดอะแกรมมุมเพิ่ม (ดู _draw_angle_diagram) จากจุดกึ่งกลาง Meter
    ถ้ามี bbox_m ไม่งั้นใช้จุดกึ่งกลางภาพแทน — ใช้เฉพาะภาพ "หลังหมุน" (00c) เพื่อโชว์ว่าหมุนไปกี่องศาจากต้นฉบับ"""
    vis = image.copy()
    cv2.putText(vis, stage_tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
    cv2.putText(vis, stage_tag, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    for box, label, color in ((bbox_m, "Meter", (255, 0, 0)), (bbox_d, "Dial(needle)", (0, 0, 255)),
                               (bbox_s, "Screen", (0, 255, 0))):
        if box is None: continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, label, (x1, max(50, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(vis, label, (x1, max(50, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
    if rotate_deg is not None:
        if bbox_m is not None:
            center = ((bbox_m[0] + bbox_m[2]) / 2, (bbox_m[1] + bbox_m[3]) / 2)
        else:
            center = (vis.shape[1] / 2, vis.shape[0] / 2)
        vis = _draw_angle_diagram(vis, center, rotate_deg)
    return vis

def _detect_screen_box_only(image):
    """หากรอบ Screen (class 2) แบบดิบๆ บน image ที่ให้มา ไม่ผ่านขั้นตอน crop/ขยายใดๆ — ใช้เฉพาะสำหรับสร้าง
    ภาพ debug คู่ก่อน/หลังหมุน (00b/00c) เท่านั้น ไม่เกี่ยวกับ Screen ที่ pipeline หลักใช้ตัดจริง (ดู
    _step4_detect_screen ซึ่งทำงานบน meter_img ที่ crop แล้ว คนละจุดประสงค์กัน) คืนกรอบ conf สูงสุด หรือ
    None ถ้าไม่เจอเลย"""
    res = model_yolo_main.predict(source=image, conf=0.1, classes=[2], verbose=False)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    if len(boxes) == 0: return None
    confs = res[0].boxes.conf.cpu().numpy()
    return boxes[int(np.argmax(confs))]

def _step4_detect_screen(meter_img):
    """[Step 4] หากรอบ Screen บน meter_img — เลือกจาก YOLO confidence ตรงๆ แล้วขยายกรอบเป็น 2.5 เท่าก่อน crop (รอบ 1)
    จากนั้นเอา crop รอบ 1 มาหา Screen ซ้ำอีกทีแบบไม่ขยาย เพื่อ refine ขอบจอให้แม่นขึ้น (รอบ 2)
    ถ้า YOLO เจอกรอบมากกว่า 1 กรอบในรอบใดรอบหนึ่ง จะเลือกใช้แค่กรอบ conf สูงสุดเสมอ — กรอบอื่นที่ไม่ถูกเลือก
    จะไม่ถูกบันทึกลง debug เลย (debug เก็บเฉพาะภาพที่ระบบ "เลือกใช้จริง" เท่านั้น ไม่เก็บผู้สมัครที่ตกรอบ)
    คืน (screen_img, screen_found, screen_source, best_candidate, screen_candidates_debug)
    screen_candidates_debug มีคู่ก่อนตัด/หลังตัดของทั้ง 2 รอบ: 03a/03b = รอบ 1, 03c/03d = รอบ 2"""
    screen_img = meter_img
    screen_found = False
    best_candidate = None
    screen_source = "none"  # บอกว่า screen ที่ใช้จริงมาจากรอบไหน: round1 (ขยาย 2.5 เท่า) หรือ round2 (เจอซ้ำ ไม่ขยาย)
    screen_candidates_debug = {}  # เก็บเฉพาะภาพก่อน/หลังตัดของกรอบที่ระบบเลือกใช้จริงในแต่ละรอบ

    res_s = model_yolo_main.predict(source=meter_img, conf=0.1, iou=0.85, classes=[2], verbose=False)
    detected_boxes = res_s[0].boxes
    if len(detected_boxes) == 0:
        return screen_img, screen_found, screen_source, best_candidate, screen_candidates_debug

    confs = detected_boxes.conf.cpu().numpy()
    best_i = int(np.argmax(confs))  # เลือกกรอบ conf สูงสุดตรงๆ — กรอบอื่น (ถ้ามี) ไม่ถูกใช้ต่อ จึงไม่เก็บ debug
    best_box = detected_boxes[best_i].xyxy[0].cpu().numpy()
    bx1, by1, bx2, by2 = _pad_box(best_box, meter_img.shape, pad_ratio=0.75)  # รอบ 1: ขยาย 2.5 เท่า

    # 03a: "ก่อนตัด" — meter_img เต็มภาพ พร้อมกรอบที่กำลังจะตัด (ก่อนขยาย 2.5 เท่า) วาดทับไว้
    screen_candidates_debug["03a_screen_round1_before"] = _draw_box_before(
        meter_img, best_box, f"round1 conf={float(confs[best_i]):.2f}")

    crop = meter_img[by1:by2, bx1:bx2]
    if crop.size == 0:
        return screen_img, screen_found, screen_source, best_candidate, screen_candidates_debug

    screen_found = True
    best_candidate = np.array([bx1, by1, bx2, by2], dtype=np.float64)
    screen_img = crop
    screen_source = "round1_expanded_2.5x"
    # 03b: "หลังตัด" — ผลลัพธ์ที่ตัดออกมาจริงในรอบ 1 (ขยาย 2.5 เท่าแล้ว)
    screen_candidates_debug["03b_screen_round1_after"] = _label_crop(crop, f"conf={float(confs[best_i]):.2f}")

    # --- รอบ 2: เอาภาพที่ขยาย 2.5 เท่าแล้ว (crop) มาหา screen ซ้ำอีกรอบ — ครั้งนี้ไม่ขยาย crop ตรงกรอบที่เจอเลย ---
    res_s2 = model_yolo_main.predict(source=crop, conf=0.1, iou=0.85, classes=[2], verbose=False)
    detected_boxes2 = res_s2[0].boxes
    if len(detected_boxes2) > 0:
        confs2 = detected_boxes2.conf.cpu().numpy()
        best_j = int(np.argmax(confs2))  # เช่นเดียวกับรอบ 1 — เลือกกรอบ conf สูงสุดเท่านั้น กรอบอื่นไม่ถูกเก็บ debug
        best_box2 = detected_boxes2[best_j].xyxy[0].cpu().numpy()
        rx1, ry1, rx2, ry2 = int(best_box2[0]), int(best_box2[1]), int(best_box2[2]), int(best_box2[3])  # รอบ 2: ไม่ขยาย

        # 03c: "ก่อนตัด" — ภาพรอบ 1 (crop) พร้อมกรอบที่รอบ 2 กำลังจะตัดซ้อนวาดไว้
        screen_candidates_debug["03c_screen_round2_before"] = _draw_box_before(
            crop, best_box2, f"round2 conf={float(confs2[best_j]):.2f}", color=(0, 200, 255))

        refined = crop[ry1:ry2, rx1:rx2]
        if refined.size > 0:
            screen_img = refined
            best_candidate = np.array([bx1 + rx1, by1 + ry1, bx1 + rx2, by1 + ry2], dtype=np.float64)
            screen_source = "round2_no_expand"
            # 03d: "หลังตัด" — ผลลัพธ์สุดท้ายที่ใช้จริงเป็น screen_img ของทั้ง pipeline
            screen_candidates_debug["03d_screen_round2_after"] = _label_crop(refined, f"conf={float(confs2[best_j]):.2f}")

    return screen_img, screen_found, screen_source, best_candidate, screen_candidates_debug

def _build_digit_overview(screen_img, boxes_n):
    """ภาพรวมทุกกรอบที่โมเดล digits เจอในรอบมุมที่ดีที่สุด (แต่ละกรอบ = 1 slot/หลักตรงๆ) — ไว้ debug"""
    digit_overview_img = screen_img.copy()
    for rb in boxes_n:
        ox1, oy1, ox2, oy2 = map(int, rb)
        cv2.rectangle(digit_overview_img, (ox1, oy1), (ox2, oy2), (0, 255, 255), 1)
    overview_text = f"digit slots found: {len(boxes_n)}/4"
    cv2.putText(digit_overview_img, overview_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(digit_overview_img, overview_text, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    return digit_overview_img

def _step5_classify_digits(boxes_n, screen_img):
    """[Step 5] จำแนกเลขทีละหลักจากกรอบดิบของ digits.pt ตรงๆ ไม่มี padding เพิ่ม (ตรงกับกรอบใน 89_digit_overview.jpg
    พอดี) แก้ไขเลขเหลื่อมผ่าน _resolve_digit_slot จากนั้นใช้ "กฎกลเข้าหาตรงกลาง" ข้ามหลักทับผลอีกชั้น
    (หลักที่ไม่ใช่ขวาสุดจะหมุนได้ก็ต่อเมื่อหลักขวากำลังพลิก 9->0) แล้วรวมเป็นเลขสุดท้ายด้วย apply_carry_logic_v2
    คืน (slot_results, digit_crops_debug, all_ids, final_reading, final_str)"""
    slot_results = []
    digit_crops_debug = []
    all_ids = []
    final_reading = None
    final_str = ""
    if len(boxes_n) == 0:
        return slot_results, digit_crops_debug, all_ids, final_reading, final_str

    # เตรียม crop ของทุก slot ก่อน (ตัด slot ที่ crop ว่างทิ้ง) แล้วจำแนกทุก slot รวดเดียวเป็น batch เดียว
    # ผ่าน modelTB — แทนที่จะเรียกทีละภาพต่อ 1 หลัก (4 รอบ inference ต่อภาพ) ลดเหลือ 1 รอบ เร็วขึ้นชัดเจน
    # โดยผลลัพธ์ไม่เปลี่ยน เพราะ modelTB.eval() แล้ว (batchnorm ใช้ running stats ไม่ใช่สถิติของ batch นี้)
    slots_to_classify = []
    for b in boxes_n:
        x1p, y1p, x2p, y2p = _pad_box(b, screen_img.shape, pad_ratio=0.0, min_pad=0)
        crop = screen_img[y1p:y2p, x1p:x2p]
        if crop.size == 0: continue
        slots_to_classify.append((b, (x1p, y1p, x2p, y2p), crop))

    if not slots_to_classify:
        return slot_results, digit_crops_debug, all_ids, final_reading, final_str

    cls_results = _classify_crops([crop for _, _, crop in slots_to_classify])
    for (b, pad_box, crop), cls in zip(slots_to_classify, cls_results):
        base_id = cls["id"]
        value, is_straddle, is_90, resolved_zero, sub_debug = _resolve_digit_slot(crop, base_id)
        all_ids.append(value * 2 + (1 if is_straddle else 0))
        # margin ระหว่าง top-1/top-2 เล็ก = classifier ลังเลระหว่าง 2 คลาสนี้ เสี่ยงทายคลาสผิดตั้งแต่ต้น
        # (ไม่มีขั้นตอนไหนหลังจากนี้ตรวจสอบย้อนกลับได้ — ธง low_confidence นี้ไว้เตือนเฉยๆ ไม่ได้แก้คำตอบให้)
        conf_margin = cls["confidence"] - cls["confidence2"]
        low_confidence = conf_margin < LOW_CONFIDENCE_MARGIN
        slot_results.append({
            "value": value, "is_90_straddle": is_90, "resolved_is_zero": resolved_zero, "box": b,
            "is_straddle": is_straddle, "pad_box": pad_box, "base_id": int(base_id),
            "split": sub_debug if is_straddle else None,
            "confidence": cls["confidence"], "confidence_margin": conf_margin,
            "low_confidence": low_confidence, "id2": cls["id2"], "confidence2": cls["confidence2"],
        })
        digit_crops_debug.append({"crop": crop, **sub_debug})

    # --- กลไกเฟือง: หลักที่ไม่ใช่ขวาสุดจะ "หมุน" ได้ก็ต่อเมื่อหลักขวาของมันกำลังพลิก 9->0 เท่านั้น ---
    # ถ้าหลักขวาอยู่แถว 9/0 จริง แปลว่าการหมุนของหลักนี้กำลังจะเสร็จ ค่าที่ถูกต้องคือ "ตัวล่าง" (ค่าที่หมุนไปแล้ว)
    # เชื่อถือได้กว่าการวัดรูปทรงของ slot นี้เอง เพราะเป็นข้อบังคับเชิงกลไก ไม่ใช่การประมาณจากพิกเซล
    # (วัดจริง: ตรง 14 จาก 15 ครั้ง เทียบกับ centroid ที่ตรง 12 จาก 17 — ทั้งชุด 348 ภาพ 83.0% -> 84.5%)
    # จงใจแก้เฉพาะทิศทางนี้ทิศเดียว ส่วนกรณีหลักขวาไม่ได้อยู่แถว 9/0 ปล่อยให้ centroid ตัดสินตามเดิม
    # เพราะข้อมูลจริงมีแค่ 2 ตัวอย่างและแบ่งกันคนละครึ่ง ยังไม่พอสรุปว่าควรบังคับไปทางไหน
    for si in range(len(slot_results) - 1):
        s = slot_results[si]
        if not s["is_straddle"]: continue
        if slot_results[si + 1]["base_id"] // 2 not in (9, 0): continue
        v_top = s["base_id"] // 2
        s["value"] = (v_top + 1) % 10
        s["resolved_is_zero"] = (s["value"] == 0)
        if s["split"] is not None: s["split"]["winner"] = "bottom"
        s["gear_forced"] = True  # บันทึกไว้ว่าค่านี้ถูกกฎกลเข้าหาตรงกลางบังคับทับ ไม่ใช่ผลจาก centroid ล้วนๆ (ใช้ทำ result.txt ให้ละเอียดขึ้น)
        all_ids[si] = s["value"] * 2 + 1

    # --- cross-check ระหว่างหลัก: หลักที่ไม่ใช่ขวาสุดจะทดได้ก็ต่อเมื่อหลักขวาของมันอยู่แถว 9/0 ---
    for si in range(len(slot_results) - 1):
        s = slot_results[si]
        if s["is_90_straddle"] and s["resolved_is_zero"]:
            if slot_results[si + 1]["value"] not in (9, 0):
                s["is_90_straddle"] = False

    if len(slot_results) == 4:
        final_str = apply_carry_logic_v2(slot_results)
    else:
        final_str = "".join(str(s["value"]) for s in slot_results)
    try: final_reading = int(final_str)
    except: pass

    # ผูกเลขสรุปสุดท้าย (หลัง carry logic) กลับเข้าแต่ละ slot ไว้ใช้กำกับภาพ debug
    for s, ch in zip(slot_results, final_str):
        s["final_char"] = ch

    return slot_results, digit_crops_debug, all_ids, final_reading, final_str

# ============================================================================
# SECTION 8: Main Entry Point
# ============================================================================
def _blank_error_image(text, size=(400, 400)):
    """ภาพ placeholder สีดำพร้อมข้อความ error — ใช้แทน viz_image ตอนที่ image_input เองก็ใช้งานไม่ได้
    (เช่น None หรือ array ว่าง) จึงไม่มีภาพต้นฉบับให้ทำสำเนาแปะข้อความลงไปเหมือนกรณี error อื่นๆ"""
    img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.putText(img, text, (10, size[0]//2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    return img

def read_water_meter(image_input, filename_ref=None):
    meter_found = False
    dial_found = False
    screen_found = False
    current_pattern = "0000"
    final_reading = None
    # all_ids = ค่าที่ตัดสินแล้วของแต่ละหลัก เข้ารหัสเป็น value*2 + (1 ถ้าเป็นเลขเหลื่อม)
    # ไม่ใช่ class id ดิบจากตัวจำแนก — คงรูปแบบนี้ไว้เพราะ TEST_app.py พึ่ง parity bit นี้อยู่
    all_ids = []
    # คำนวณไว้นอก try ตั้งแต่ต้น (ไม่ใช่แค่ในเคสสำเร็จ) เพื่อให้ debug/<clean_name>/ ใช้ชื่อเดียวกันได้แน่นอน
    # ทุก exit point รวมถึงตอน exception ด้วย
    clean_name = os.path.splitext(os.path.basename(filename_ref))[0] if filename_ref else "unknown"

    # กันภาพที่ใช้งานไม่ได้ตั้งแต่ต้น (image_input เป็น None หรือ array ว่าง) — เช่นตอน cv2.imread() อ่านไฟล์
    # เสีย/พังแล้วคืน None แล้วผู้เรียกลืมเช็คก่อนส่งเข้ามา (เคยพบว่า except Exception ด้านล่างเรียก
    # image_input.copy() ซ้ำตอนสร้าง viz_image ของ error case ซึ่งพังซ้ำถ้า image_input เป็น None จริง —
    # exception หลุดออกไปนอกฟังก์ชันทั้งที่มี try/except ครอบไว้ ทำให้ทั้ง batch พังจากภาพเสียใบเดียว)
    if image_input is None or not hasattr(image_input, "copy") or getattr(image_input, "size", 1) == 0:
        return {
            "final_reading": None, "error": "ภาพต้นฉบับใช้งานไม่ได้ (None หรือว่างเปล่า)",
            "meter_found": False, "dial_found": False, "screen_found": False,
            "pattern": "0000", "all_ids": [], "viz_image": _blank_error_image("ERROR: INVALID IMAGE INPUT"),
        }

    try:
        image = image_input.copy()

        # [Step 1] Initial Detect & Angle
        boxes, clss, confs1, bbox_m, bbox_d, meter_found, dial_found = _step1_find_orientation_refs(image)

        if bbox_m is None and bbox_d is None:
            # ไม่เจอทั้ง Meter และ Dial — ไม่มีจุดอ้างอิงให้หมุนภาพเลย หยุดตรงนี้แจ้ง error ทันที
            err_img = image.copy()
            err_text = "ERROR: NOT FOUND METER"
            cv2.putText(err_img, err_text, (10, err_img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5)
            cv2.putText(err_img, err_text, (10, err_img.shape[0]-50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            return {
                "final_reading": None,
                "error": "ไม่เห็นมิเตอร์ ลองทำความสะอาดมิเตอร์",
                "meter_found": False, "dial_found": False, "screen_found": False,
                "pattern": "0000", "all_ids": [], "viz_image": err_img
            }

        # ภาพ debug "ก่อนหมุน" (00b) — โชว์ว่า Step 1 ตรวจเจอ Meter/Dial(เข็ม)/Screen ตรงไหนบ้างบนภาพต้นฉบับ
        # จริงๆ ก่อนที่จะรู้ผลการหมุนเลยด้วยซ้ำ Screen ไม่ได้เป็นส่วนหนึ่งของ Step 1 ปกติ (ตรวจแค่ Meter/Dial)
        # จึงต้องเรียก _detect_screen_box_only เพิ่มอีกครั้งเฉพาะเพื่อทำภาพนี้ (ไม่กระทบผลลัพธ์จริงของ pipeline)
        bbox_s_before = _detect_screen_box_only(image)
        detect_before_rotate = _draw_orientation_detections(image, bbox_m, bbox_d, bbox_s_before, "BEFORE ROTATE")

        rotated_image, rotate_deg = _rotate_by_reference(image, bbox_m, bbox_d)

        # [Step 2-3] Rotation Raw & Detection -> Meter Crop
        boxes_rot, clss_rot, confs_rot, bbox_m_rot, meter_img = _step2_3_detect_and_crop_meter(rotated_image, bbox_m)
        # "ก่อนตัด" ของ Step 3 — rotated_image เต็มภาพ พร้อมกรอบ Meter ที่กำลังจะตัดวาดทับไว้ (ไว้ debug คู่กับ meter_img)
        # กำกับมุมที่หมุนไปด้วย (rotate_deg) ให้เห็นว่าภาพนี้หมุนมาจากต้นฉบับกี่องศา
        meter_before = _draw_box_before(rotated_image, bbox_m_rot, f"meter crop (rotated {rotate_deg:.1f} deg)") if bbox_m_rot is not None else None

        # ภาพ debug "หลังหมุน" (00c) — ตรวจซ้ำเหมือนกันแต่บนภาพที่หมุนแล้ว เทียบกับ 00b ให้เห็นชัดว่าหมุน
        # ไปแล้วกรอบขยับไปตรงไหน (bbox_m_rot มาจาก _step2_3_detect_and_crop_meter ที่เรียกไปแล้วด้านบน)
        bbox_d_rot = _best_box(boxes_rot, clss_rot, confs_rot, 1, max_area_ratio=0.10,
                                image_area=rotated_image.shape[0] * rotated_image.shape[1])
        bbox_s_after = _detect_screen_box_only(rotated_image)
        detect_after_rotate = _draw_orientation_detections(rotated_image, bbox_m_rot, bbox_d_rot, bbox_s_after,
                                                            "AFTER ROTATE", rotate_deg=rotate_deg)

        # [Step 4] Screen Detection & Crop
        screen_img, screen_found, screen_source, best_candidate, screen_candidates_debug = _step4_detect_screen(meter_img)

        boxes_n, screen_img = _scan_digits(screen_img)
        digit_overview_img = _build_digit_overview(screen_img, boxes_n)

        # pattern ตอนนี้บอกแค่ "เจอกี่หลักจาก 4" ไม่ได้บอกตำแหน่งที่หายไปแล้ว (เดิมทำได้ผ่าน _slots_from_boxes ที่ตัดออก
        # เพราะ bucket ตำแหน่งผิดได้ง่ายเมื่อแถวเลขไม่กินเต็มความกว้าง crop) — ใช้พอสำหรับ stat_tracker/รายงานสรุป
        current_pattern = ("1" * min(len(boxes_n), 4)).ljust(4, "0")
        stat_tracker.update(current_pattern, clean_name)

        # [Step 5] Classification — resolve straddling digits via ink-projection center analysis
        slot_results, digit_crops_debug, all_ids, final_reading, _ = _step5_classify_digits(boxes_n, screen_img)

        # [Step 6-7] Visualization: ตีกรอบอธิบายการทำงาน + แถบสรุปท้ายภาพ
        viz_image = _draw_pipeline_visualization(
            rotated_image, bbox_m_rot, boxes_rot, clss_rot, confs_rot, best_candidate,
            slot_results, final_reading, meter_found, dial_found, screen_found,
        )

        # --- บันทึกภาพทุกขั้นตอนแยกโฟลเดอร์ต่อภาพ (debug) ---
        _save_pipeline_debug_outputs(
            clean_name, image, rotated_image, rotate_deg, meter_before, meter_img, screen_img, screen_candidates_debug,
            digit_overview_img, slot_results, digit_crops_debug, viz_image, screen_source,
            final_reading, current_pattern, all_ids,
            detect_before_rotate=detect_before_rotate, detect_after_rotate=detect_after_rotate,
        )

        return {
            "final_reading": final_reading,
            "error": None,
            "meter_found": meter_found,
            "dial_found": dial_found,
            "screen_found": screen_found,
            "pattern": current_pattern,
            "all_ids": all_ids, # ค่าที่ตัดสินแล้ว (value*2+parity) ใช้ตรวจว่าหลักไหนเป็นเลขเหลื่อม
            "raw_ids": [s.get("base_id") for s in slot_results], # class id ดิบจากตัวจำแนกก่อนตัดสินใจใดๆ (0-19 ต่อ slot)
            "is_straddle_slots": [s.get("is_straddle", False) for s in slot_results], # ธงบอกว่า slot ไหนเป็นเลขเหลื่อม เรียงซ้าย->ขวา
            "gear_forced_slots": [s.get("gear_forced", False) for s in slot_results], # slot ไหนถูกกฎกลเข้าหาตรงกลางบังคับผล (ไม่ใช่ผลจาก centroid ล้วนๆ)
            "rotate_deg": rotate_deg, # มุมที่หมุนภาพไปจริง (องศา) ก่อนเริ่มตรวจจับ
            "viz_image": viz_image, # ส่งภาพที่ตีกรอบแล้วออกไปเซฟ
        }
    except Exception as e:
        # image_input ผ่านการเช็คว่าใช้งานได้ (มี .copy(), ไม่ว่าง) ไปแล้วตั้งแต่ต้นฟังก์ชัน จึง .copy() ซ้ำ
        # ที่นี่ได้อย่างปลอดภัย — แต่กันเผื่ออีกชั้นด้วย fallback เผื่อ .copy() เองล้มเหลวจากเหตุอื่น (เช่น
        # หน่วยความจำเต็ม) ไม่ให้ exception จากจุดนี้หลุดออกไปทำให้ทั้ง batch พังซ้ำสอง
        try: viz_image = image_input.copy()
        except Exception: viz_image = _blank_error_image("ERROR: PIPELINE FAILED")
        return {
            "final_reading": None, "error": str(e), "meter_found": False,
            "dial_found": False, "screen_found": False, "pattern": "0000",
            "all_ids": [], "viz_image": viz_image
        }

# ============================================================================
# SECTION 9: Multi-Photo Majority Vote
# ============================================================================
def read_water_meter_batch(image_list, filename_refs=None):
    """[เสริม] อ่านมิเตอร์จากภาพหลายใบของ "มิเตอร์ตัวเดียวกัน ค่าเดียวกัน" (เช่น ถ่ายรัวหลายรูปติดกัน
    ในการอ่านครั้งเดียว) แล้วรวมคำตอบด้วยการโหวตเสียงข้างมากทีละหลัก แทนที่จะเชื่อภาพใดภาพหนึ่งเดี่ยวๆ

    ที่มา: ตรวจชุดทดสอบจริง 348 ภาพ พบว่าแท้จริงมีแค่ 88 "เหตุการณ์ถ่าย" ที่ไม่ซ้ำกัน (ภาพส่วนใหญ่เป็นการ
    ถ่ายซ้ำมิเตอร์ตัวเดียวกันหลายใบ) และ error ในกลุ่มถ่ายซ้ำสหสัมพันธ์กันสูงมาก (68/86 กลุ่มตอบเหมือนกัน
    ทุกภาพ ทั้งตอนถูกและตอนผิด) เพราะภาพในกลุ่มเดียวกันแทบไม่มีอะไรต่างกันเลย (มุม/แสง/สภาพเดียวกัน)
    วัดจริง: ความแม่นยำระดับภาพเดี่ยว 84.5% -> ความแม่นยำระดับเหตุการณ์ (โหวตทีละหลัก) 89.8%
    ทดสอบ 2 วิธีโหวตแล้วให้ผลเท่ากันทุกกลุ่มในชุดทดสอบ (0 กลุ่มต่างกัน) จึงเลือกแบบทีละหลัก เพราะทนทาน
    กว่าโดยทฤษฎี — ยังนับคะแนนได้แม้ไม่มี 2 ภาพไหนตอบทั้งคำตอบตรงกันเป๊ะ ขอแค่หลักส่วนใหญ่ตรงกันก็พอ

    เงื่อนไขการใช้: ภาพทุกใบใน image_list ต้องเป็นภาพของมิเตอร์ตัวเดียวกัน ค่าเดียวกัน ณ เวลาเดียวกัน —
    ฟังก์ชันนี้ไม่ตรวจสอบเงื่อนไขนี้ให้ เป็นหน้าที่ผู้เรียก (เช่น ถ่ายรัว 3-5 รูปติดกันในแอปเดียวกัน)

    Args:
        image_list: list ของภาพ (numpy array, BGR) ของมิเตอร์ตัวเดียวกัน
        filename_refs: list ชื่อไฟล์อ้างอิงคู่กัน (สำหรับ debug/log) ยาวเท่า image_list หรือ None

    Returns: dict เพิ่มเติมจาก read_water_meter ปกติ
        final_reading: ผลโหวตสุดท้าย (int) หรือ None ถ้าทุกภาพอ่านไม่ได้เลย
        agreement: สัดส่วนภาพที่ตอบตรงกับผลโหวต (0.0-1.0) — ใกล้ 1 = มั่นใจสูง ใกล้ 0.25 = กลุ่มเห็นต่างกันมาก
        member_results: ผลลัพธ์ดิบของแต่ละภาพ (list ของ dict จาก read_water_meter ตามลำดับ image_list)
        used_for_vote: ดัชนีของภาพที่ถูกนำไปโหวตจริง (กรองภาพที่ตรวจจับไม่ครบ 4 หลักออกก่อน ถ้ามีให้เลือก)
    """
    n = len(image_list)
    if filename_refs is None: filename_refs = [None] * n

    member_results = [read_water_meter(img, filename_ref=fr) for img, fr in zip(image_list, filename_refs)]

    # ให้น้ำหนักภาพที่ตรวจจับครบ 4 หลัก (pattern == "1111") มากกว่า — ภาพที่จับไม่ครบมักมีเหตุผลอื่น
    # (บังแสง/เบลอ) ที่ทำให้ผลไม่น่าเชื่อถือ ใช้เป็น fallback เฉพาะตอนไม่มีภาพไหนจับครบเลยสักใบ
    full = [i for i, r in enumerate(member_results) if r.get("final_reading") is not None and r.get("pattern") == "1111"]
    used = full if full else [i for i, r in enumerate(member_results) if r.get("final_reading") is not None]

    if not used:
        return {
            "final_reading": None, "agreement": 0.0, "member_results": member_results,
            "used_for_vote": [], "error": "ทุกภาพในกลุ่มอ่านไม่ได้เลย",
        }

    readings = [member_results[i]["final_reading"] for i in used]
    if len(set(readings)) == 1:
        return {
            "final_reading": readings[0], "agreement": 1.0, "member_results": member_results,
            "used_for_vote": used, "error": None,
        }

    # โหวตทีละหลัก (zero-pad เป็นความกว้าง 4 หลักให้ตรงกันก่อนเทียบ) เสมอกัน -> ใช้หลักจาก "ผลโหวตทั้งคำตอบ"
    # เป็นตัวตัดสินรอง (ทดสอบแล้วว่าสองวิธีนี้ไม่เคยขัดกันในชุดทดสอบจริง จึงใช้เป็น tie-break ที่ปลอดภัย)
    from collections import Counter
    width = 4
    strs = [f"{v:0{width}d}" for v in readings]
    whole_mode = Counter(readings).most_common(1)[0][0]
    whole_mode_str = f"{whole_mode:0{width}d}"
    out_digits = []
    for pos in range(width):
        c = Counter(s[pos] for s in strs)
        top_count = c.most_common(1)[0][1]
        tied = [ch for ch, cnt in c.items() if cnt == top_count]
        out_digits.append(tied[0] if len(tied) == 1 else whole_mode_str[pos])
    final_reading = int("".join(out_digits))

    agree = sum(1 for r in readings if r == final_reading) / len(readings)
    return {
        "final_reading": final_reading, "agreement": agree, "member_results": member_results,
        "used_for_vote": used, "error": None,
    }
