# FieldCare backend

The backend exposes the trained EfficientNet model as an HTTP API.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

By default the newest best checkpoint under `outputs/`, `output/`, or
`models/train/` is loaded. Override it with `FIELDCARE_CHECKPOINT`, or choose a
device with `FIELDCARE_DEVICE=cpu`, `cuda`, or `auto`.

`GET /health` reports model readiness. `POST /predict` accepts one JPG, JPEG,
or PNG file in the multipart field named `file`, up to 5 MB.

`GET /treatments/{class_key}` returns cause factors, treatment or correction
steps, prevention guidance, sources, and the agricultural-extension disclaimer
for every class produced by the model.

Successful predictions are stored in SQLite. `GET /history` returns saved scans,
`GET /history/stats` returns summary counts, and `GET /history/{scan_id}/image`
returns the original uploaded image. The database defaults to
`data/fieldcare.db`; set `FIELDCARE_DB_PATH` to use a different location.
