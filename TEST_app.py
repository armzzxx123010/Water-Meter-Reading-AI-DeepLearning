import os
import sys
import cv2
import time
import glob
import re
import json # สำหรับทำไฟล์ให้ Gemini อ่าน
import csv # สำหรับทำตารางละเอียดทุกภาพ (4_ตารางละเอียดทุกภาพ.csv)

# บังคับ stdout ให้ flush ทุกบรรทัดทันที (line-buffered) แทนที่จะรอ buffer เต็มหรือโปรแกรมจบก่อนค่อย
# แสดงผล — ปกติ Python จะ full-buffer stdout เองอัตโนมัติทันทีที่ไม่ได้ต่อกับ terminal จริงๆ (เช่น รันผ่าน
# VSCode debugger/debugpy, redirect ไปไฟล์ด้วย >, หรือ pipe) ทำให้ progress ที่ print ไว้ทุกภาพ (ดู
# "[{idx+1}/{total_imgs}] ...") ไม่โผล่ออกมาเลยจนกว่าโปรแกรมจะรันจบทั้งหมด ดูเหมือนโปรแกรม "ค้าง" ทั้งที่
# จริงๆ กำลังประมวลผลอยู่ปกติ — เคยเกิดปัญหานี้จริงตอนรันผ่าน F5 debugger แล้วเข้าใจผิดว่าค้าง
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass  # บาง environment (เช่น stdout ถูกแทนที่ด้วย object อื่นที่ไม่มี reconfigure) ก็แค่ข้ามไป ไม่ทำให้พัง

# 1. โหลด AI Module
try:
    import WMDP as wmdp
    print("INFO: Loaded WMDP successfully")
except ImportError:
    print("ERROR: WMDP.py not found")
    sys.exit(1)

# 2. ตั้งค่า Path
INPUT_FOLDER = r"C:\Users\Acer Nitro 5\Desktop\Water-meter-AI-Sanwa-master\Water-meter-AI-Sanwa-master edit\UsedTest"
MODELS_BASE_DIR = r"C:\Users\Acer Nitro 5\Desktop\Water-meter-AI-Sanwa-master\Water-meter-AI-Sanwa-master edit\VitnadResnet"
MAIN_OUTPUT_FOLDER = "Test_copy"
GROUND_TRUTH_PATH = r"C:\Users\Acer Nitro 5\Desktop\Water-meter-AI-Sanwa-master\Water-meter-AI-Sanwa-master edit\check.xlsx"

CURRENT_MODEL = "Vit-tiny"

def natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_)]

def load_ground_truth(path):
    """โหลดค่าจริง (ground truth) จาก check.xlsx ไว้เทียบความแม่นยำ — คืน dict {clean_filename: {"digits":[d1,d2,d3,d4], "answer": "0433"}}
    ถ้าไฟล์ไม่มีหรือเปิดไม่ได้ คืน dict ว่าง (รายงานความแม่นยำจะข้ามส่วนที่ต้องพึ่งค่าจริงไปเอง ไม่ทำให้ทั้งสคริปต์พัง)"""
    if not os.path.exists(path):
        print(f"WARNING: ไม่พบไฟล์ ground truth ที่ {path} — จะข้ามส่วนคำนวณความแม่นยำในรายงาน")
        return {}
    try:
        import openpyxl
        wb = openpyxl.load_workbook(path, data_only=True)
        ws = wb["Sheet1"]
        gt = {}
        for row in ws.iter_rows(min_row=5, values_only=True):
            if row[0] is None: continue
            name = str(row[0]).strip()
            digits = row[2:6]
            answer = row[7]
            if answer is None or any(d is None for d in digits): continue
            gt[name] = {"digits": [int(d) for d in digits], "answer": str(answer).strip()}
        return gt
    except Exception as e:
        print(f"WARNING: อ่าน ground truth ไม่สำเร็จ ({e}) — จะข้ามส่วนคำนวณความแม่นยำในรายงาน")
        return {}

def run_single_process(model_name=None, decision_method=None):
    """รันทดสอบทั้งชุด 348 ภาพกับโมเดล 1 ตัว — ถ้าไม่ระบุ model_name ใช้ CURRENT_MODEL (ค่าเริ่มต้นเดิม)
    ตั้งเป็นพารามิเตอร์แทนตัวแปร global เดิม เพื่อให้เรียกวนซ้ำได้หลายโมเดลในการรันครั้งเดียว (ดู run_all_models)
    decision_method: "centroid" (ค่าเริ่มต้นจริงของระบบ) หรือ "area" (โหมดเปรียบเทียบ ใช้พื้นที่หมึกตัดสิน
    ฝั่งชนะเลขเหลื่อมแทน) — ตั้งค่า wmdp.STRADDLE_WINNER_METHOD ก่อนรัน และเติมชื่อโฟลเดอร์ output ต่อท้าย
    ด้วย "_area" กันไปทับผลลัพธ์ของโหมด centroid เดิม (ดู run_area_comparison)"""
    model_name = model_name or CURRENT_MODEL
    decision_method = decision_method or "centroid"
    wmdp.STRADDLE_WINNER_METHOD = decision_method
    if not os.path.exists(INPUT_FOLDER): return print("ERROR: Input folder not found")

    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    if not image_files: return print("ERROR: No images found")

    image_files.sort(key=lambda x: natural_key(os.path.basename(x)))

    method_tag = "" if decision_method == "centroid" else f"_{decision_method}"
    print(f"\n🚀 เริ่มทดสอบโมเดล: {model_name} (วิธีตัดสินเลขเหลื่อม: {decision_method})")
    model_path = os.path.join(MODELS_BASE_DIR, model_name)
    out_folder = os.path.join(MAIN_OUTPUT_FOLDER, model_name + method_tag)

    wmdp.init_ai(model_path, out_folder)
    ground_truth = load_ground_truth(GROUND_TRUTH_PATH)

    # ==========================================
    # สร้าง 3 โฟลเดอร์สำหรับแยกภาพตามหมวดหมู่ผลลัพธ์
    # (ชื่อโฟลเดอร์ตอนสร้างยังไม่มีจำนวนภาพ เพราะยังไม่รู้ผลจนกว่าจะรันครบ — จะถูก rename ใส่จำนวนภาพ
    # ต่อท้ายชื่อหลังประมวลผลเสร็จทั้งหมดแล้วเท่านั้น ดูขั้นตอน "เติมจำนวนภาพลงชื่อโฟลเดอร์" ท้ายไฟล์)
    # ==========================================
    dir_full = os.path.join(out_folder, "1_เลขเต็ม")
    dir_half = os.path.join(out_folder, "2_เลขเหลื่อม")
    dir_inc = os.path.join(out_folder, "3_ภาพที่จับไม่ครบ")
    os.makedirs(dir_full, exist_ok=True)
    os.makedirs(dir_half, exist_ok=True)
    os.makedirs(dir_inc, exist_ok=True)

    total_imgs = len(image_files)

    count_full = 0  # นับจำนวนภาพในแต่ละหมวด ไว้เติมท้ายชื่อโฟลเดอร์ตอนจบ (เลขเหลื่อม/จับไม่ครบนับผ่าน stat_half/stat_inc อยู่แล้ว)

    # ตัวแปรสำหรับ Report เลขเหลื่อม
    stat_half = {
        "total_images": 0,
        "slots_count": {1:0, 2:0, 3:0, 4:0}, # หลักที่ 1, 2, 3, 4
        "half_per_image": {1:0, 2:0, 3:0, 4:0}, # มีเหลื่อม 1 ตัวกี่ภาพ, 2 ตัวกี่ภาพ...
        "value_freq": {str(k): 0 for k in range(10)}, # เลข 0-9 ที่เหลื่อม
        "images_per_pos": {1: [], 2: [], 3: [], 4: []}, # ชื่อไฟล์ภาพที่เหลื่อม แยกตามหลัก 1-4
    }

    # ตัวแปรสำหรับ Report ภาพที่จับไม่ครบ
    stat_inc = {
        "total_images": 0,
        "errors_list": []
    }

    # ตัวแปรสำหรับ Report ความแม่นยำ + ข้อผิดพลาด (ต้องมี ground_truth ถึงจะคำนวณได้)
    clean_value_freq = {str(k): 0 for k in range(10)}  # ความถี่ค่าของ "เลขนิ่ง" (ไม่ใช่เลขเหลื่อม) ทุก slot ทุกภาพที่จับครบ 4 หลัก
    per_pos_correct = {1: 0, 2: 0, 3: 0, 4: 0}
    per_pos_wrong = {1: 0, 2: 0, 3: 0, 4: 0}
    per_pos_total = {1: 0, 2: 0, 3: 0, 4: 0}
    overall_correct = 0
    overall_total_with_truth = 0
    wrong_images = []  # list ของ dict: filename, predicted, actual, per-position reasons
    detail_rows = []  # ตารางละเอียดที่สุด: 1 แถวต่อ 1 หลัก ต่อ 1 ภาพ (4 แถว/ภาพ) สำหรับไฟล์ 4_ตารางละเอียดทุกภาพ.csv

    for idx, img_path in enumerate(image_files):
        fname = os.path.basename(img_path)
        t_start = time.time()
        img = cv2.imread(img_path)
        if img is None: continue

        res = wmdp.read_water_meter(img, filename_ref=fname)
        dur = (time.time() - t_start) * 1000

        pattern = res.get('pattern', '0000')
        all_ids = res.get('all_ids', [])
        viz_image = res.get('viz_image')

        # --- วิเคราะห์หมวดหมู่ภาพ ---
        missing_digits = 4 - sum(int(c) for c in pattern)
        category = ""

        if pattern != "1111" or len(all_ids) < 4:
            category = "3_ภาพที่จับไม่ครบ"
        else:
            # ตรวจสอบเลขเหลื่อม (idx % 2 != 0)
            is_half = any(v % 2 != 0 for v in all_ids)
            if is_half:
                category = "2_เลขเหลื่อม"
            else:
                category = "1_เลขเต็ม"
                count_full += 1

        # --- เซฟภาพลงโฟลเดอร์ ---
        if viz_image is not None:
            save_path = os.path.join(out_folder, category, fname)
            cv2.imwrite(save_path, viz_image)

        # --- เก็บสถิติแยกตามโฟลเดอร์ ---
        if category == "2_เลขเหลื่อม":
            stat_half["total_images"] += 1
            half_in_this_img = 0
            for slot_idx, val_id in enumerate(all_ids):
                if val_id % 2 != 0: # เป็นเลขเหลื่อม
                    stat_half["slots_count"][slot_idx + 1] += 1
                    half_in_this_img += 1
                    base_val = str(val_id // 2)
                    stat_half["value_freq"][base_val] += 1
                    stat_half["images_per_pos"][slot_idx + 1].append(fname)
            if half_in_this_img > 0:
                stat_half["half_per_image"][half_in_this_img] += 1

        elif category == "3_ภาพที่จับไม่ครบ":
            stat_inc["total_images"] += 1
            # เตรียมข้อมูลให้ Gemini
            err_cause = []
            if not res.get('meter_found'): err_cause.append("METER_NOT_DETECTED (อาจเบลอมาก หรือไม่อยู่ในเฟรม)")
            if not res.get('screen_found'): err_cause.append("SCREEN_NOT_DETECTED (อาจมีแสงสะท้อนรุนแรง หรือมุมเอียงเกินไป)")
            if missing_digits > 0: err_cause.append(f"DIGITS_MISSING: หายไป {missing_digits} ตัว (อาจเพราะเงาบัง ฝุ่น หรือเลขกำลังหมุนเปลี่ยนหลักจนโมเดลหาไม่เจอ)")

            stat_inc["errors_list"].append({
                "filename": fname,
                "missing_count": missing_digits,
                "detected_pattern": pattern,
                "ai_diagnostics": {
                    "meter_detected": res.get("meter_found"),
                    "screen_detected": res.get("screen_found"),
                    "dial_detected": res.get("dial_found")
                },
                "gemini_analysis_context": "System failed to read complete 4 digits. " + " | ".join(err_cause)
            })

        # --- นับความถี่ "เลขนิ่ง" (ไม่ใช่เลขเหลื่อม) ทุก slot ของภาพที่จับครบ 4 หลัก ---
        if pattern == "1111" and len(all_ids) == 4:
            for v in all_ids:
                if v % 2 == 0:  # เลขนิ่ง (parity bit = 0)
                    clean_value_freq[str(v // 2)] += 1

        # --- เทียบกับค่าจริง (ground truth) เพื่อคำนวณความแม่นยำ + หาสาเหตุที่ผิด ---
        clean_name = os.path.splitext(fname)[0]
        truth = ground_truth.get(clean_name)
        reading_for_gt = res.get('final_reading')
        raw_ids = res.get('raw_ids') or []
        is_straddle_slots = res.get('is_straddle_slots') or []
        gear_forced_slots = res.get('gear_forced_slots') or []
        if truth is not None and pattern == "1111" and len(all_ids) == 4 and reading_for_gt is not None:
            # ใช้ final_reading (ผ่าน carry logic แล้ว) เป็นค่าทายจริง — ห้ามคำนวณจาก all_ids//2 ตรงๆ
            # เพราะ all_ids ไม่รวมผลของการทดเลขข้ามหลัก (carry) ตอนหลักขวาพลิกผ่าน 9->0 จะได้ค่าผิดจากที่ระบบอ่านจริง
            predicted_str = f"{reading_for_gt:04d}"
            predicted_digits = [int(ch) for ch in predicted_str]
            truth_digits = truth["digits"]
            truth_str = truth["answer"]

            overall_total_with_truth += 1
            is_fully_correct = (predicted_str == truth_str)
            if is_fully_correct: overall_correct += 1

            pos_reasons = []
            for pos in range(4):  # pos 0-3 = หลักที่ 1-4
                per_pos_total[pos + 1] += 1
                if pos >= len(predicted_digits) or pos >= len(truth_digits): continue
                if predicted_digits[pos] == truth_digits[pos]:
                    per_pos_correct[pos + 1] += 1
                else:
                    per_pos_wrong[pos + 1] += 1
                    is_str = is_straddle_slots[pos] if pos < len(is_straddle_slots) else False
                    if is_str and pos < len(raw_ids) and raw_ids[pos] is not None:
                        base = raw_ids[pos] // 2
                        pair = (base, (base + 1) % 10)
                        if truth_digits[pos] in pair:
                            reason = f"หลักที่{pos+1}: เลือกฝั่งเลขเหลื่อมผิด (คู่ {pair[0]}/{pair[1]} ทายได้ {predicted_digits[pos]} จริงคือ {truth_digits[pos]})"
                        else:
                            reason = f"หลักที่{pos+1}: โมเดลทายคู่เลขเหลื่อมผิดตั้งแต่ต้น (เสนอคู่ {pair[0]}/{pair[1]} แต่ค่าจริง {truth_digits[pos]} อยู่นอกคู่นี้)"
                    else:
                        reason = f"หลักที่{pos+1}: เลขนิ่งทายผิดตรงๆ (ทายได้ {predicted_digits[pos]} จริงคือ {truth_digits[pos]})"
                    pos_reasons.append(reason)

            if not is_fully_correct:
                wrong_images.append({
                    "filename": fname, "predicted": predicted_str, "actual": truth_str,
                    "reasons": pos_reasons,
                })

        # --- สร้างแถวตารางละเอียดที่สุด (4 แถวต่อภาพ = 1 แถวต่อหลัก) ไม่ว่าภาพนั้นจะถูก/ผิด/จับไม่ครบ/ไม่มีเฉลยก็ตาม ---
        final_str_full = f"{reading_for_gt:04d}" if reading_for_gt is not None else None
        truth_digits_full = truth["digits"] if truth is not None else None
        truth_str_full = truth["answer"] if truth is not None else None
        image_overall = "ไม่ทราบ (ไม่มีเฉลย)" if truth_str_full is None else (
            "ไม่ทราบ (อ่านไม่สำเร็จ/จับไม่ครบ)" if final_str_full is None else
            ("ถูก" if final_str_full == truth_str_full else "ผิด")
        )
        for pos in range(4):
            slot_no = pos + 1
            raw_id = raw_ids[pos] if pos < len(raw_ids) else None
            is_str = is_straddle_slots[pos] if pos < len(is_straddle_slots) else False
            gear = gear_forced_slots[pos] if pos < len(gear_forced_slots) else False
            pair_str = ""
            if raw_id is not None and is_str:
                base = raw_id // 2
                pair_str = f"{base}/{(base + 1) % 10}"
            predicted_digit = final_str_full[pos] if final_str_full is not None and pos < len(final_str_full) else "-"
            actual_digit = str(truth_digits_full[pos]) if truth_digits_full is not None and pos < len(truth_digits_full) else "ไม่มีเฉลย"

            if truth_digits_full is None:
                slot_correct = "ไม่ทราบ (ไม่มีเฉลย)"
                reason = ""
            elif final_str_full is None:
                slot_correct = "ไม่ทราบ (อ่านไม่สำเร็จ/จับไม่ครบ)"
                reason = "ขั้นตอนตรวจจับล้มเหลวก่อนถึงขั้นอ่านเลข (ดูคอลัมน์ พบมิเตอร์/พบหน้าปัด/พบจอตัวเลข)"
            elif predicted_digit == actual_digit:
                slot_correct = "ถูก"
                reason = ""
            else:
                slot_correct = "ผิด"
                if is_str and raw_id is not None:
                    base = raw_id // 2
                    pair = (base, (base + 1) % 10)
                    if int(actual_digit) in pair:
                        reason = f"เลือกฝั่งเลขเหลื่อมผิด (คู่ {pair[0]}/{pair[1]} ทายได้ {predicted_digit} จริงคือ {actual_digit})"
                    else:
                        reason = f"โมเดลทายคู่เลขเหลื่อมผิดตั้งแต่ต้น (เสนอคู่ {pair[0]}/{pair[1]} แต่ค่าจริง {actual_digit} อยู่นอกคู่นี้)"
                else:
                    reason = f"เลขนิ่งทายผิดตรงๆ (ทายได้ {predicted_digit} จริงคือ {actual_digit})"

            if gear:
                gear_note = "ถูกบังคับด้วยกฎกลเข้าหาตรงกลาง (gear rule: หลักขวาใกล้ 9/0 จึงบังคับใช้ค่าฝั่งล่าง)"
                reason = f"{reason} | {gear_note}" if reason else gear_note

            detail_rows.append({
                "ชื่อภาพ": fname,
                "พบมิเตอร์": "พบ" if res.get("meter_found") else "ไม่พบ",
                "พบหน้าปัด": "พบ" if res.get("dial_found") else "ไม่พบ",
                "พบจอตัวเลข": "พบ" if res.get("screen_found") else "ไม่พบ",
                "หมุนกี่องศา": f"{res.get('rotate_deg', 0.0):.2f}",
                "แพทเทิร์นที่จับได้": pattern,
                "หลักที่": slot_no,
                "คลาสดิบ_raw_id": raw_id if raw_id is not None else "-",
                "เป็นเลขเหลื่อม": "ใช่" if is_str else "ไม่ใช่",
                "คู่เลขเหลื่อม": pair_str,
                "ถูกบังคับกฎเฟือง": "ใช่" if gear else "ไม่ใช่",
                "เลขที่ทายได้": predicted_digit,
                "เลขจริง": actual_digit,
                "ผลลัพธ์หลักนี้": slot_correct,
                "เหตุผล/หมายเหตุ": reason,
                "ค่าที่อ่านได้ทั้งภาพ": final_str_full if final_str_full is not None else "อ่านไม่สำเร็จ",
                "ค่าจริงทั้งภาพ": truth_str_full if truth_str_full is not None else "ไม่มีเฉลย",
                "ภาพนี้ถูกทั้งหมดไหม": image_overall,
            })

        reading = res.get('final_reading')
        status = f"READING: {reading:04d}" if reading is not None else "FAILED"
        print(f"[{idx+1}/{total_imgs}] {fname} -> {status} ({dur:.0f}ms) | Pattern: {pattern} -> [{category}]")

    # ==========================================
    # สร้าง Report 0: ความแม่นยำโดยรวม + ข้อผิดพลาด + สาเหตุ (ไฟล์รวมทุกอย่าง อยู่นอกโฟลเดอร์ย่อย)
    # ==========================================
    report_acc_path = os.path.join(out_folder, "0_รายงานความแม่นยำและข้อผิดพลาด.txt")
    with open(report_acc_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"รายงานความแม่นยำและข้อผิดพลาด — โมเดล: {model_name} (วิธีตัดสินเลขเหลื่อม: {decision_method})\n")
        f.write("=" * 70 + "\n\n")

        # --- 1. ความแม่นยำโดยรวม ---
        f.write("1. ความแม่นยำโดยรวม\n")
        f.write("-" * 70 + "\n")
        if not ground_truth:
            f.write(f"ไม่มีไฟล์ ground truth ({GROUND_TRUTH_PATH}) จึงคำนวณความแม่นยำไม่ได้\n")
            f.write("(ยังคงมีรายงานส่วนอื่น เช่น ภาพที่จับไม่ครบ / ความถี่เลขซ้ำ ตามปกติด้านล่าง)\n\n")
        elif overall_total_with_truth == 0:
            f.write("ไม่มีภาพไหนที่ตรวจจับครบ 4 หลัก และมีค่าจริงให้เทียบเลย\n\n")
        else:
            acc_pct = 100 * overall_correct / overall_total_with_truth
            f.write(f"ถูกทั้งภาพ: {overall_correct}/{overall_total_with_truth} = {acc_pct:.1f}%\n")
            f.write(f"(นับเฉพาะภาพที่ตรวจจับครบ 4 หลัก และมีค่าจริงในไฟล์ check.xlsx ให้เทียบ — "
                    f"ภาพที่จับไม่ครบดูหัวข้อ 4 แยกต่างหาก ไม่ถูกนับว่า 'ผิด' ในหัวข้อนี้)\n\n")

            # --- 2. ความแม่นยำแยกตามหลัก ---
            f.write("2. ความแม่นยำแยกตามหลัก (ซ้าย -> ขวา) — หลักไหนผิดบ่อยที่สุด\n")
            f.write("-" * 70 + "\n")
            worst_pos, worst_wrong = None, -1
            for pos in range(1, 5):
                total_p = per_pos_total[pos]
                wrong_p = per_pos_wrong[pos]
                pct_p = 100 * per_pos_correct[pos] / total_p if total_p else 0.0
                f.write(f"   หลักที่ {pos}: ถูก {per_pos_correct[pos]}/{total_p} = {pct_p:.1f}% (ผิด {wrong_p} ครั้ง)\n")
                if wrong_p > worst_wrong: worst_pos, worst_wrong = pos, wrong_p
            if worst_pos is not None and worst_wrong > 0:
                f.write(f"   -> หลักที่ผิดบ่อยที่สุด: หลักที่ {worst_pos} ({worst_wrong} ครั้ง)\n")
            f.write("\n")

            # --- 3. รายการภาพที่ผิด พร้อมสาเหตุรายหลัก ---
            f.write(f"3. รายการภาพที่อ่านผิด ({len(wrong_images)} ภาพ) พร้อมสาเหตุ\n")
            f.write("-" * 70 + "\n")
            if not wrong_images:
                f.write("ไม่มีภาพไหนอ่านผิดเลย (ในกลุ่มที่ตรวจจับครบ 4 หลัก)\n\n")
            else:
                for w in wrong_images:
                    f.write(f"   {w['filename']}: ทายได้ {w['predicted']} | ค่าจริง {w['actual']}\n")
                    for r in w["reasons"]:
                        f.write(f"      - {r}\n")
                f.write("\n")

        # --- 4. ภาพที่ต้องใช้เงื่อนไขพิเศษ (จับไม่ครบ/ไม่ชัด/บางส่วนหาไม่เจอ) ---
        f.write(f"4. ภาพที่ต้องใช้เงื่อนไขพิเศษ — ตรวจจับไม่ครบ/ไม่ชัด/หาบางอย่างไม่เจอ ({stat_inc['total_images']} ภาพ)\n")
        f.write("-" * 70 + "\n")
        if not stat_inc["errors_list"]:
            f.write("ไม่มีภาพไหนตรวจจับไม่ครบเลย\n\n")
        else:
            for e in stat_inc["errors_list"]:
                f.write(f"   {e['filename']}: แพทเทิร์นที่จับได้ {e['detected_pattern']} (หายไป {e['missing_count']} หลัก)\n")
                f.write(f"      - {e['gemini_analysis_context']}\n")
            f.write("\n(รายละเอียดเชิงลึกสำหรับให้ AI วิเคราะห์ต่อ อยู่ในไฟล์ ข้อมูลข้อผิดพลาดสำหรับ_AI_Gemini.json ในโฟลเดอร์ 3)\n\n")

        # --- 5. เลขที่ซ้ำบ่อยๆ ทั้งเลขเหลื่อมและเลขนิ่ง(เต็ม) ---
        f.write("5. เลขที่พบซ้ำบ่อยที่สุด (0-9)\n")
        f.write("-" * 70 + "\n")
        f.write("   ก) ตัวเลขที่เกิดอาการ 'เหลื่อม' บ่อยที่สุด (เห็นเลข 2 ตัวซ้อนกัน):\n")
        half_sorted = sorted(stat_half["value_freq"].items(), key=lambda kv: -kv[1])
        if any(v > 0 for _, v in half_sorted):
            for k, v in half_sorted:
                if v > 0: f.write(f"      เลข {k}: {v} ครั้ง\n")
        else:
            f.write("      ไม่พบข้อมูลเลขเหลื่อมในชุดทดสอบนี้\n")
        f.write("   ข) ตัวเลข 'นิ่ง' (ไม่เหลื่อม) ที่พบบ่อยที่สุด ในภาพที่จับครบ 4 หลัก:\n")
        clean_sorted = sorted(clean_value_freq.items(), key=lambda kv: -kv[1])
        if any(v > 0 for _, v in clean_sorted):
            for k, v in clean_sorted:
                if v > 0: f.write(f"      เลข {k}: {v} ครั้ง\n")
        else:
            f.write("      ไม่พบข้อมูล\n")
        f.write("\n")

    # ==========================================
    # สร้าง Report 3: ตารางละเอียดที่สุด ทุกภาพ ทุกหลัก (CSV เปิดด้วย Excel ได้)
    # 1 แถว = 1 หลัก (4 แถวต่อภาพ) มีข้อมูลทุกขั้นตอนของ pipeline + ถูก/ผิด + เหตุผล
    # ==========================================
    report_detail_path = os.path.join(out_folder, "4_ตารางละเอียดทุกภาพ.csv")
    if detail_rows:
        fieldnames = list(detail_rows[0].keys())
        with open(report_detail_path, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(detail_rows)

    # ==========================================
    # สร้าง Report 3b: เนื้อหาเดียวกับ Report 3 (CSV) แต่เป็น .txt อ่านง่ายแบบจัดกลุ่มทีละภาพ
    # (เผื่อไม่อยากเปิด Excel — เปิดด้วย Notepad ได้เลย)
    # ==========================================
    report_detail_txt_path = os.path.join(out_folder, "4b_ตารางละเอียดทุกภาพ.txt")
    with open(report_detail_txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"ตารางละเอียดทุกภาพ ทุกหลัก — โมเดล: {model_name}\n")
        f.write("(เนื้อหาเดียวกับ 4_ตารางละเอียดทุกภาพ.csv แต่จัดกลุ่มเป็นภาพ-ต่อ-ภาพให้อ่านง่ายด้วย Notepad)\n")
        f.write("=" * 70 + "\n\n")
        for i in range(0, len(detail_rows), 4):
            img_rows = detail_rows[i:i + 4]
            if not img_rows: continue
            r0 = img_rows[0]
            f.write("-" * 70 + "\n")
            f.write(f"{r0['ชื่อภาพ']}\n")
            f.write(f"  พบมิเตอร์: {r0['พบมิเตอร์']} | พบหน้าปัด: {r0['พบหน้าปัด']} | พบจอตัวเลข: {r0['พบจอตัวเลข']} | "
                    f"หมุน: {r0['หมุนกี่องศา']} องศา | แพทเทิร์นที่จับได้: {r0['แพทเทิร์นที่จับได้']}\n")
            f.write(f"  ค่าที่อ่านได้ทั้งภาพ: {r0['ค่าที่อ่านได้ทั้งภาพ']} | ค่าจริงทั้งภาพ: {r0['ค่าจริงทั้งภาพ']} | "
                    f"ภาพนี้ถูกทั้งหมดไหม: {r0['ภาพนี้ถูกทั้งหมดไหม']}\n\n")
            for row in img_rows:
                pair_txt = row['คู่เลขเหลื่อม'] if row['คู่เลขเหลื่อม'] else "-"
                reason_txt = row['เหตุผล/หมายเหตุ'] if row['เหตุผล/หมายเหตุ'] else "-"
                f.write(f"    หลักที่ {row['หลักที่']}: คลาสดิบ={row['คลาสดิบ_raw_id']} | เหลื่อม={row['เป็นเลขเหลื่อม']} | "
                        f"คู่={pair_txt} | กฎเฟือง={row['ถูกบังคับกฎเฟือง']} | ทายได้={row['เลขที่ทายได้']} | "
                        f"จริง={row['เลขจริง']} | ผลลัพธ์={row['ผลลัพธ์หลักนี้']}\n")
                f.write(f"       เหตุผล/หมายเหตุ: {reason_txt}\n")
            f.write("\n")

    # ==========================================
    # สร้าง Report 4: ไฟล์แยกเฉพาะ "ภาพที่ผิด + เหตุผล" อย่างเดียว (เอาไว้เปิดดูภายหลังโดยไม่ต้องเปิดรายงานรวม)
    # ==========================================
    report_wrong_path = os.path.join(out_folder, "5_ภาพที่ผิดและเหตุผล.txt")
    with open(report_wrong_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write(f"ภาพที่อ่านผิด + เหตุผล — โมเดล: {model_name}\n")
        f.write("(ไฟล์นี้แยกมาจากรายงานรวม 0_รายงานความแม่นยำและข้อผิดพลาด.txt เพื่อเปิดดูเฉพาะภาพที่มีปัญหาได้ง่ายภายหลัง)\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"ก) ภาพที่ตรวจจับครบ 4 หลัก แต่ทายผิดจากเฉลย ({len(wrong_images)} ภาพ)\n")
        f.write("-" * 70 + "\n")
        if not wrong_images:
            f.write("ไม่มี\n\n")
        else:
            for w in wrong_images:
                f.write(f"   {w['filename']}: ทายได้ {w['predicted']} | ค่าจริง {w['actual']}\n")
                for r in w["reasons"]:
                    f.write(f"      - {r}\n")
            f.write("\n")

        f.write(f"ข) ภาพที่ตรวจจับไม่ครบ 4 หลัก (เปรียบเทียบกับเฉลยไม่ได้) ({stat_inc['total_images']} ภาพ)\n")
        f.write("-" * 70 + "\n")
        if not stat_inc["errors_list"]:
            f.write("ไม่มี\n")
        else:
            for e in stat_inc["errors_list"]:
                f.write(f"   {e['filename']}: แพทเทิร์นที่จับได้ {e['detected_pattern']} (หายไป {e['missing_count']} หลัก)\n")
                f.write(f"      - {e['gemini_analysis_context']}\n")

    # ==========================================
    # สร้าง Report 1: นับเลขเหลื่อม (ในโฟลเดอร์ 2)
    # ==========================================
    # สร้างโฟลเดอร์ซ้ำอีกครั้งก่อนเขียนไฟล์ (กันเคส Desktop sync กับ OneDrive ลบโฟลเดอร์ว่างทิ้งไปเอง
    # ระหว่างที่ยังไม่มีไฟล์ข้างในเลย ทำให้ makedirs ตอนต้นสคริปต์ใช้ไม่ได้ผลตอนมาเปิดไฟล์ทีหลัง)
    os.makedirs(dir_half, exist_ok=True)
    report_half_path = os.path.join(dir_half, "สถิติเลขเหลื่อม.txt")
    with open(report_half_path, "w", encoding="utf-8") as f:
        f.write("สรุปสถิติภาพที่มีเลขเหลื่อม (Half Digits Report)\n")
        f.write("="*50 + "\n")
        f.write(f"1. ภาพที่มีเลขเหลื่อมทั้งหมด: {stat_half['total_images']} ภาพ\n\n")
        
        f.write("2. จำนวนเลขเหลื่อมในแต่ละหลัก (จากซ้ายไปขวา):\n")
        for i in range(1, 5):
            f.write(f"   - หลักที่ {i}: {stat_half['slots_count'][i]} ครั้ง\n")
            
        f.write("\n3. ความถี่ของการพบเลขเหลื่อมใน 1 ภาพ:\n")
        for i in range(1, 5):
            f.write(f"   - ภาพที่มีเลขเหลื่อม {i} ตัว: {stat_half['half_per_image'][i]} ภาพ\n")
            
        if stat_half['total_images'] > 0:
            most_freq_val = max(stat_half['value_freq'], key=stat_half['value_freq'].get)
            f.write(f"\n4. ตัวเลขที่เกิดอาการเหลื่อมบ่อยที่สุด (0-9): เลข '{most_freq_val}' (เกิด {stat_half['value_freq'][most_freq_val]} ครั้ง)\n")
            f.write("   - รายละเอียดความถี่ทั้งหมด:\n")
            for k, v in stat_half['value_freq'].items():
                if v > 0: f.write(f"      เลข {k}: {v} ครั้ง\n")
        else:
            f.write("\n4. ไม่พบข้อมูลเลขเหลื่อมในชุดทดสอบนี้\n")

        f.write("\n5. รายชื่อภาพที่เหลื่อม แยกตามหลัก (1-4):\n")
        for i in range(1, 5):
            names = stat_half["images_per_pos"][i]
            f.write(f"   - หลักที่ {i}: {len(names)} ภาพ\n")
            if names:
                # พิมพ์ทีละ 10 ชื่อต่อบรรทัด ให้อ่านง่าย ไม่ยาวเกินไป
                for j in range(0, len(names), 10):
                    f.write("      " + ", ".join(names[j:j+10]) + "\n")
            else:
                f.write("      (ไม่มี)\n")

    # ==========================================
    # สร้าง Report 2: ข้อผิดพลาด (ในโฟลเดอร์ 3) สำหรับ Gemini
    # ==========================================
    os.makedirs(dir_inc, exist_ok=True)  # กันเคสเดียวกับ dir_half ด้านบน
    report_inc_path = os.path.join(dir_inc, "ข้อมูลข้อผิดพลาดสำหรับ_AI_Gemini.json")
    # ใช้ JSON เพื่อให้ Gemini อ่านแล้วเข้าใจ 100%
    gemini_data = {
        "report_type": "INCOMPLETE_DIGIT_DETECTION_ANALYSIS",
        "total_incomplete_images": stat_inc["total_images"],
        "instructions_for_gemini": "Please analyze this JSON data to understand why the water meter AI failed to read complete digits. Provide a summary of the most common failure points (e.g., missing screen vs missing digits) and suggest physical environment improvements.",
        "error_details": stat_inc["errors_list"]
    }
    with open(report_inc_path, "w", encoding="utf-8") as f:
        json.dump(gemini_data, f, ensure_ascii=False, indent=4)

    # ==========================================
    # เติมจำนวนภาพลงชื่อโฟลเดอร์ (ทำเป็นขั้นตอนสุดท้ายเสมอ — ต้องรอรู้จำนวนจริงหลังประมวลผลครบทุกภาพก่อน
    # ถึงจะ rename ได้ ภาพ/รายงานทั้งหมดที่เซฟไปก่อนหน้านี้อยู่ในโฟลเดอร์ชื่อเดิม ตามด้วยไปกับการ rename เอง)
    # ==========================================
    count_half = stat_half["total_images"]
    count_inc = stat_inc["total_images"]

    def _safe_rename_with_count(src, count):
        """rename src -> src_{count}ภาพ อย่างปลอดภัย:
        - ถ้าปลายทางมีอยู่แล้ว (ค้างจากรอบรันก่อนหน้าที่ rename ไม่สำเร็จ) ให้ลบทิ้งก่อนแล้วค่อย rename
        - ถ้า src หายไปเฉยๆ (เช่น โดน OneDrive sync ลบโฟลเดอร์ว่างทิ้ง) ให้สร้างใหม่แบบว่างเปล่าแล้ว rename แทน
          (กันสคริปต์พังตอนจบทั้งที่ประมวลผลภาพครบและรายงานอื่นเขียนเสร็จหมดแล้ว)"""
        import shutil
        dst = src + f"_{count}ภาพ"
        if os.path.exists(dst):
            shutil.rmtree(dst)
        if not os.path.exists(src):
            os.makedirs(src, exist_ok=True)
        os.rename(src, dst)

    _safe_rename_with_count(dir_full, count_full)
    _safe_rename_with_count(dir_half, count_half)
    _safe_rename_with_count(dir_inc, count_inc)

    print(f"\n✅ ประมวลผลเสร็จสิ้น! บันทึกภาพและรายงานลงในโฟลเดอร์: {out_folder}")
    print(f"   1_เลขเต็ม: {count_full} ภาพ | 2_เลขเหลื่อม: {count_half} ภาพ | 3_ภาพที่จับไม่ครบ: {count_inc} ภาพ")
    acc_pct = None
    if ground_truth and overall_total_with_truth > 0:
        acc_pct = 100 * overall_correct / overall_total_with_truth
        print(f"   ความแม่นยำโดยรวม: {overall_correct}/{overall_total_with_truth} = {acc_pct:.1f}% "
              f"(ดูรายละเอียดที่ 0_รายงานความแม่นยำและข้อผิดพลาด.txt)")

    # คืนค่าสรุปสั้นๆ ไว้ให้ run_all_models()/run_area_comparison() เก็บไปทำตารางเปรียบเทียบ โดยไม่ต้องเปิด
    # ไฟล์ report ซ้ำ
    return {
        "model_name": model_name, "decision_method": decision_method,
        "out_folder": out_folder, "report_path": report_acc_path,
        "overall_correct": overall_correct if (ground_truth and overall_total_with_truth > 0) else None,
        "overall_total": overall_total_with_truth if (ground_truth and overall_total_with_truth > 0) else None,
        "accuracy_pct": acc_pct,
    }

# ============================================================================
# เปรียบเทียบหลายโมเดลในการรันครั้งเดียว (Full-sequence accuracy)
# ============================================================================
# รายชื่อโฟลเดอร์โมเดล (ต้องตรงกับชื่อโฟลเดอร์ย่อยใต้ MODELS_BASE_DIR ทุกตัวอักษร) — เพิ่ม/ลดรายการนี้
# ได้ตามโมเดลที่ extract ไว้จริงใน VitnadResnet/ (ดูโฟลเดอร์ที่มี config.json + model.safetensors)
MODELS_TO_COMPARE = ["Vit-tiny", "ResNet34", "Vit-base"]
COMPARE_REPORT_FOLDER = "123"  # โฟลเดอร์ปลายทางสำหรับรายงานเปรียบเทียบ (ตามที่ตกลงกันไว้)

def run_all_models(model_names=None):
    """รันทดสอบทั้งชุด 348 ภาพซ้ำทีละโมเดลตาม model_names (ค่าเริ่มต้น = MODELS_TO_COMPARE) ในการรัน
    สคริปต์ครั้งเดียว — แต่ละโมเดลได้ผลลัพธ์เต็มรูปแบบแยกโฟลเดอร์ของตัวเอง (Test_copy/<ชื่อโมเดล>/ เหมือน
    Test_copy/Vit-tiny เดิมทุกไฟล์) จากนั้นคัดลอกรายงาน 0_รายงานความแม่นยำและข้อผิดพลาด.txt ของแต่ละโมเดล
    ไปไว้ในโฟลเดอร์ COMPARE_REPORT_FOLDER ด้วย พร้อมสร้างตารางสรุปเปรียบเทียบ Full-sequence accuracy
    ระหว่างโมเดลทั้งหมดไว้ในไฟล์เดียว"""
    import shutil
    model_names = model_names or MODELS_TO_COMPARE
    os.makedirs(COMPARE_REPORT_FOLDER, exist_ok=True)
    summaries = []
    for name in model_names:
        model_dir = os.path.join(MODELS_BASE_DIR, name)
        if not os.path.exists(model_dir):
            print(f"⚠️  ข้ามโมเดล '{name}' — ไม่พบโฟลเดอร์ {model_dir}")
            continue
        result = run_single_process(name)
        if not result:
            print(f"⚠️  โมเดล '{name}' รันไม่สำเร็จ (ไม่มีภาพให้ทดสอบ หรือ input folder หาไม่เจอ)")
            continue
        summaries.append(result)
        # คัดลอกรายงานฉบับเต็มของโมเดลนี้ไปไว้ที่โฟลเดอร์เปรียบเทียบ ตั้งชื่อแยกตามโมเดลกันไฟล์ทับกัน
        dst = os.path.join(COMPARE_REPORT_FOLDER, f"0_รายงานความแม่นยำและข้อผิดพลาด_{name}.txt")
        if os.path.exists(result["report_path"]):
            shutil.copy2(result["report_path"], dst)
            print(f"   -> คัดลอกรายงานไปที่ {dst}")

    # ตารางสรุปเปรียบเทียบ Full-sequence accuracy ของทุกโมเดลที่รันสำเร็จ ไว้ไฟล์เดียวดูง่าย
    if summaries:
        cmp_path = os.path.join(COMPARE_REPORT_FOLDER, "0_เปรียบเทียบ Full-sequence accuracy ระหว่างโมเดล.txt")
        with open(cmp_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write("เปรียบเทียบ Full-sequence Accuracy ระหว่างโมเดล (รันชุดทดสอบเดียวกันทุกโมเดล)\n")
            f.write("=" * 70 + "\n\n")
            best = max((s for s in summaries if s["accuracy_pct"] is not None),
                       key=lambda s: s["accuracy_pct"], default=None)
            for s in summaries:
                if s["accuracy_pct"] is None:
                    f.write(f"{s['model_name']:<12} ไม่มีไฟล์ ground truth ให้เทียบ\n")
                    continue
                mark = "  <-- แม่นยำสุด" if best and s is best else ""
                f.write(f"{s['model_name']:<12} ถูกทั้งภาพ {s['overall_correct']}/{s['overall_total']} "
                        f"= {s['accuracy_pct']:.1f}%{mark}\n")
            f.write(f"\nรายงานฉบับเต็มของแต่ละโมเดล (per-digit accuracy, รายชื่อภาพที่ผิด+สาเหตุ ฯลฯ) "
                    f"อยู่ในไฟล์ 0_รายงานความแม่นยำและข้อผิดพลาด_<ชื่อโมเดล>.txt ในโฟลเดอร์เดียวกันนี้\n")
        print(f"\n📊 บันทึกตารางเปรียบเทียบไว้ที่ {cmp_path}")
    return summaries

# ============================================================================
# เปรียบเทียบวิธีตัดสินฝั่งชนะเลขเหลื่อม: centroid (ค่าเริ่มต้นจริงของระบบ) vs area (พื้นที่หมึก)
# ============================================================================
def run_area_comparison(model_name=None):
    """รันชุดทดสอบ 348 ภาพซ้ำ 2 รอบด้วยโมเดลจำแนกตัวเดียวกัน (model_name, ค่าเริ่มต้น = CURRENT_MODEL)
    ต่างกันแค่วิธีตัดสินฝั่งชนะของเลขเหลื่อม: รอบ 1 ใช้ "centroid" (ระยะจาก centroid ถึงกึ่งกลางกรอบ —
    วิธีที่ระบบใช้งานจริง) รอบ 2 ใช้ "area" (พื้นที่หมึกเยอะกว่าชนะ — วิธีเดิมก่อนเปลี่ยนมาใช้ centroid)
    ผลลัพธ์แต่ละวิธีแยกโฟลเดอร์ (Test_copy/<model>/ กับ Test_copy/<model>_area/) ไม่ทับกัน จากนั้นคัดลอก
    รายงานทั้ง 2 + ตารางเปรียบเทียบไปไว้ที่ COMPARE_REPORT_FOLDER (ตามที่อาจารย์ขอไว้)"""
    import shutil
    model_name = model_name or CURRENT_MODEL
    os.makedirs(COMPARE_REPORT_FOLDER, exist_ok=True)
    summaries = []
    for method in ("centroid", "area"):
        result = run_single_process(model_name, decision_method=method)
        if not result:
            print(f"⚠️  วิธี '{method}' รันไม่สำเร็จ")
            continue
        summaries.append(result)
        dst = os.path.join(COMPARE_REPORT_FOLDER,
                            f"0_รายงานความแม่นยำและข้อผิดพลาด_{model_name}_{method}.txt")
        if os.path.exists(result["report_path"]):
            shutil.copy2(result["report_path"], dst)
            print(f"   -> คัดลอกรายงานไปที่ {dst}")

    if summaries:
        cmp_path = os.path.join(COMPARE_REPORT_FOLDER,
                                 f"0_เปรียบเทียบ centroid vs area ({model_name}).txt")
        with open(cmp_path, "w", encoding="utf-8") as f:
            f.write("=" * 70 + "\n")
            f.write(f"เปรียบเทียบวิธีตัดสินฝั่งชนะเลขเหลื่อม — โมเดล: {model_name}\n")
            f.write("(รันชุดทดสอบ 348 ภาพเดียวกัน เปลี่ยนแค่วิธีตัดสิน centroid vs area)\n")
            f.write("=" * 70 + "\n\n")
            best = max((s for s in summaries if s["accuracy_pct"] is not None),
                       key=lambda s: s["accuracy_pct"], default=None)
            for s in summaries:
                label = "centroid (ใช้งานจริงในระบบ)" if s["decision_method"] == "centroid" \
                         else "area (พื้นที่หมึก, โหมดเปรียบเทียบ)"
                if s["accuracy_pct"] is None:
                    f.write(f"{label:<35} ไม่มีไฟล์ ground truth ให้เทียบ\n")
                    continue
                mark = "  <-- แม่นยำสุด" if best and s is best else ""
                f.write(f"{label:<35} ถูกทั้งภาพ {s['overall_correct']}/{s['overall_total']} "
                        f"= {s['accuracy_pct']:.1f}%{mark}\n")
            f.write(f"\nรายงานฉบับเต็มของแต่ละวิธี (per-digit accuracy, รายชื่อภาพที่ผิด+สาเหตุ ฯลฯ) "
                    f"อยู่ในไฟล์ 0_รายงานความแม่นยำและข้อผิดพลาด_{model_name}_<วิธี>.txt ในโฟลเดอร์เดียวกันนี้\n")
        print(f"\n📊 บันทึกตารางเปรียบเทียบไว้ที่ {cmp_path}")
    return summaries

if __name__ == "__main__":
    run_single_process(decision_method="area")  # รันครั้งเดียว วิธี "area" (พื้นที่หมึก) อย่างเดียว
    # ไม่รัน centroid คู่ด้วย — ผลลัพธ์ไปที่ Test_copy/<model>_area/ (ดูรายงาน 0_รายงานความแม่นยำฯ ในนั้น)
    # จะกลับไปรันแบบ centroid ปกติ (ค่าเริ่มต้นจริงของระบบ) ให้เรียก run_single_process() เฉยๆ แทน
    # หรือรันเทียบหลายโมเดลให้เรียก run_all_models() หรือเทียบ centroid vs area พร้อมกันให้เรียก
    # run_area_comparison() (ฟังก์ชันทั้งหมดยังอยู่ด้านบน ไม่ได้ลบทิ้ง)