# Drawing-to-3D

**Drawing-to-3D** — локальное приложение для преобразования инженерных PDF-чертежей в 3D CAD-модели.

Использует CV/OCR, локальные мультимодальные LLM через Ollama и FreeCAD. Поддерживает экспорт в STEP, FCStd и STL.

## 🚀 Быстрый запуск

**Требования:** Windows x64, PowerShell, интернет и WinGet (Microsoft App Installer).

### 1. Скачать проект

```powershell
git clone https://github.com/Ruziy/Drawing-to-3D.git
cd Drawing-to-3D
```

Или скачайте ZIP через **Code → Download ZIP**, распакуйте и откройте PowerShell в папке проекта.

### 2. Установить зависимости и LLM

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL.ps1 -Model qwen3.5:9b
```

Установщик автоматически подготовит необходимые компоненты: Python 3.11, библиотеки, Tesseract OCR, FreeCAD и Ollama с выбранной моделью.

Другие варианты:

```powershell
# Gemma 12B
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL.ps1 -Model gemma4:12b

# Несколько моделей
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL.ps1 -Model "qwen3.5:9b,gemma4:12b"

# Без LLM — только CV/OCR
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL.ps1 -WithoutLLM
```

Уже установленные модели Ollama повторно не скачиваются.

### 3. Запустить приложение

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\START.ps1
```

Откройте в браузере:

**http://127.0.0.1:8000**

Загрузите PDF-чертёж, выберите модель и нажмите **«Анализировать и построить 3D»**.

## ⚙️ Дополнительные команды

```powershell
# Запуск на другом порту
powershell -NoProfile -ExecutionPolicy Bypass -File .\START.ps1 -Port 8001

# Принудительный режим без LLM
powershell -NoProfile -ExecutionPolicy Bypass -File .\START.ps1 -Classic

# Предпросмотр плана установки без скачивания
powershell -NoProfile -ExecutionPolicy Bypass -File .\INSTALL.ps1 -Model qwen3.5:9b -Plan
```

Для остановки сервера нажмите `Ctrl+C`.

## 📦 Результаты

- **STEP / FCStd** — CAD-модели.
- **STL** — треугольная 3D-сетка.
- **JSON** — извлечённые размеры, план построения и результаты проверок.

> **Важно:** качество реконструкции зависит от чертежа и распознавания. Успешное создание 3D-модели не гарантирует полного геометрического соответствия оригиналу. Для работы крупных LLM требуется достаточный объём RAM/VRAM.
