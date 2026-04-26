# Web Application

This folder contains the diabetic retinopathy web demo that connects trained Hybrid CNN-ViT model to an interactive webpage.

## Run

From the project root:

```bash
cd "Web Application"
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://127.0.0.1:5000
```

## Notes

- The app loads `../hybrid_best.h5` from the project root.
- The prediction classes are mapped as:
  - `0`: No DR
  - `1`: Mild NPDR
  - `2`: Moderate NPDR
  - `3`: Severe NPDR
  - `4`: Proliferative DR
- This project is for academic demonstration and research only.
