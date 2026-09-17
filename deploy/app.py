"""
Hugging Face Spaces entrypoint for the RailPulse cascade dashboard.

How to deploy (permanent public link, free):
  1. Create a Space at https://huggingface.co/new-space  (SDK: "Docker" NOT needed
     - use "Blank" or "Streamlit"? -> pick SDK = "Docker"? no; pick "Blank").
     Easiest: create the Space with SDK "Gradio", then REPLACE app.py with this file.
  2. Push the project so the repo root looks like:

        app.py                 <- this file
        requirements.txt       <- deploy/requirements.txt (copy to root)
        src/                   <- the whole src folder
        data/  models/  results/   <- commit these too (see note below)

  3. On first boot the app self-heals: if models/bundle.joblib is missing or
     was trained with different library versions, it RETRAINS automatically
     (~2-3 min on the free CPU). Afterwards it serves the dashboard.

Note on the model bundle: models/bundle.joblib is ~32 MB. Committing it via
plain git may require Git LFS on HF. To keep things simple you can OMIT the
bundle - the app will train it on the Space itself (free tier has 2 vCPU).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "src"))

import uvicorn  # noqa: E402
from app import app  # noqa: E402  (FastAPI app serving the RailPulse dashboard at "/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 7860)))
