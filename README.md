# Adobe 1b: Persona-Driven Document Intelligence System

## Overview
This project extracts and prioritizes document sections based on a user persona and job-to-be-done, using semantic analysis and hierarchical understanding. It supports PDF input and outputs structured JSON results.

## Requirements
- Python 3.10 (only)
- pip (Python package manager)

## Setup
1. **Clone the repository:**
   ```bash
   git clone https://github.com/Gaya3-m/Adobe-1b.git
   cd Adobe-1b
   ```
2. **(Recommended) Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # Or
   source venv/bin/activate  # On Linux/Mac
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Input Format
- Place your PDF files in the project directory.
- Prepare an `input.json` file specifying the documents, persona, and job-to-be-done. Example:

```json
{
  "documents": [
    {"filename": "South of France - Cities.pdf"},
    {"filename": "South of France - Cuisine.pdf"}
    // ... more PDFs ...
  ],
  "persona": {"role": "Travel Planner"},
  "job_to_be_done": {"task": "Plan a trip of 4 days for a group of 10 college friends."}
}
```

## How to Run
1. **Generate output:**
   ```bash
   python main.py --input_json input.json --output output.json
   ```
   - This will create `output.json` with extracted and ranked sections/subsections.

2. **Evaluate output:**
   ```bash
   python evaluate.py output.json output_ref.json
   ```
   - Compares your output to the reference and prints precision, recall, F1, and rank correlation.

## Notes
- Only Python 3.10 is supported.
- All dependencies are listed in `requirements.txt`.
- For first-time use, the model files for `sentence-transformers` and `keybert` will be downloaded automatically.
- If you encounter issues with missing packages, ensure your virtual environment is activated and run the install command again.

## Project Structure
- `main.py` - Main pipeline for document analysis
- `evaluate.py` - Evaluation script
- `requirements.txt` - Python dependencies
- `input.json` - Input configuration (user-provided)
- `output.json` - Output results (generated)
- `output_ref.json` - Reference output for evaluation
- `.gitignore` - Files and folders to ignore in git

## Example
```bash
python main.py --input_json input.json --output output.json
python evaluate.py output.json output_ref.json
``` 
