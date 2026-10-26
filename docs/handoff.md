# Handoff: Manga Translator

## สถานะ (2026-10-26)
MVP ครบ pipeline ทีละหน้า ภาษาต้นฉบับญี่ปุ่นและอังกฤษ แปลด้วย Ollama `qwen3.5:4b` มี UI หน้าต่างเดียวและ CLI
pytest ผ่าน 24 ข้อ (รวม Ollama จริงและ LaMa บน GPU) ส่วน `tools/ui_smoke.py` เปิดหน้าต่างจริง รันครบ 5 ขั้น แก้คำแปล แล้ว Undo/Redo ได้

## โครงสร้าง
- `manga_translator/core/`: project (PageDoc/Region + page.json), detect, ocr, translate, inpaint, typeset, pipeline (Models โหลดทีละตัว + 5 ขั้น), settings
- `manga_translator/ui/`: main_window, canvas (กล่อง + แปรง mask), region_panel, settings_dialog, worker (QThread)
- `models/` ไม่เข้า git ดาวน์โหลดด้วย `tools/download_models.py` (hash อยู่ใน manifest.json)

## ตัวเลขที่วัดได้ (RTX 5060 8GB, หน้าสังเคราะห์ 900×1300)
- detect 1.3 วิ (ONNX บน CPU), manga-ocr โหลดครั้งแรก ~20 วิ อ่าน 3 กล่องเร็ว, Ollama ครั้งแรก ~80 วิ (โหลดโมเดล) ครั้งต่อไป ~3 วิ, clean 0.1 วิ, typeset ~1 วิ
- VRAM ของ torch สูงสุด 0.44 GB (ไม่รวม Ollama)

## จุดที่ต้องระวัง
- `opencv-python` (ที่ rapidocr ดึงมา) ชนกับ `opencv-python-headless`/PySide6 ให้ติดตั้งเฉพาะตัว headless
- `cv2.imread` เปิด path ที่มีอักษรไทย/ญี่ปุ่นไม่ได้ ให้ใช้ `project.imread/imwrite` (np.fromfile + imdecode)
- ห้ามส่ง Thai ที่มีเครื่องหมายวรรคตอนให้ newmm ทั้งก้อน (`เหรอ...` จะกลายเป็น `เห|รอ.`) ใน `typeset.tokenize` จึงแยกส่วนที่เป็นไทยออกก่อน
- คำพูดภาษาปากใส่เพิ่มได้ที่ `THAI_EXTRA_WORDS`
- RapidOCR ต้องเติมขอบ 20px รอบภาพที่ crop ไม่งั้นคำจะแตก (`YoU s`)
- `bubble_rect` ใช้ flood fill จากกลางกล่องบนภาพ cleaned ถ้าเกิน 20 เท่าของกล่องหรือเกือบเต็มหน้า จะถือว่ารั่วออกนอกบอลลูนและใช้กล่องเดิมแทน

## ยังไม่ได้ทำ / ไอเดียรอบหน้า
- ทดสอบกับหน้ามังงะจริง (ต้องรอผู้ใช้ให้ภาพ)
- ใส่ข้อความแนวตั้งสำหรับปลายทางภาษาญี่ปุ่น/จีน (ตอนนี้วางแนวนอนทั้งหมด)
- คิวแปลทั้งบทพร้อมทำต่อจากที่ค้างได้ และส่งบริบทข้ามหน้าให้ตัวแปล
- ฟอนต์ไทยแนวการ์ตูน (OFL) ใส่ไว้ใน `fonts/`
