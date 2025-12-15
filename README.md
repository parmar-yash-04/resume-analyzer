# Resume Analyzer

A Flask-based web application that analyzes resume text using a pre-trained machine learning model to predict job categories.

## Prerequisites

- Python 3.x
- `pip` (Python package installer)
- **Tesseract-OCR**: Required for image-based resumes.
    - **Windows**: Download and install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki).
    - **Linux**: `sudo apt-get install tesseract-ocr`
    - **macOS**: `brew install tesseract`

## Installation

1.  Navigate to the project directory:
    ```bash
    cd "c:/Users/ASUS/Downloads/Resume analyser"
    ```

2.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Important**: Ensure Tesseract is in your system PATH. If not, you may need to configure the path in `app.py`:
    ```python
    # Example for Windows
    # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    ```

## Running the Application

1.  Run the application using Gunicorn:
    ```bash
    gunicorn app:app
    ```
    *Note: If you are on Windows and `gunicorn` is not supported or difficult to set up, you can run the Flask development server directly:*
    ```bash
    python app.py
    ```

2.  Open your web browser and go to:
    ```
    http://127.0.0.1:8000
    ```
    (or `http://127.0.0.1:5000` if running with `python app.py`)

## Usage

1.  Click the upload area or drag and drop a resume file.
2.  Supported formats: **PDF, PNG, JPG, JPEG**.
3.  Click "Analyze Resume".
4.  The predicted job role will be displayed below the form.
