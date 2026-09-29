# YOLOE Studio

Realtime open-vocabulary detection & segmentation с онлайн-обучением.

## Возможности

- 🎯 Open-vocabulary детекция через YOLOE-26s-seg
- ⚡ Async pipeline: read → infer → result (не блокирует UI)
- 🔄 Онлайн-обучение: per-class conf, few-shot refinement, экспорт кропов
- 📐 ROI-зоны, ByteTrack трекинг
- 💾 Экспорт детекций в JSONL + CSV
- 🎨 Современный тёмный UI (Gradio)
- ⌨️ 15 горячих клавиш
- 🩺 Health-check, метрики p50/p95

## Установка

```bash
pip install ultralytics gradio opencv-python torch numpy pyyaml huggingface_hub
python download_model.py
python app.py
