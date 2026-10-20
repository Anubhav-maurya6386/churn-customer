export PYTHONPATH := src

install:   ; pip install -r requirements.txt
data:      ; python -m churn.data generate
train:     ; python -m churn.train
predict:   ; python -m churn.predict
drift:     ; python -m churn.data generate --drift --n 2000 --out data/raw/new_batch_drifted.csv && python -m churn.monitor --input data/raw/new_batch_drifted.csv
test:      ; python -m pytest -q
api:       ; uvicorn churn.api:app --port 8000
dashboard: ; streamlit run app/dashboard.py
all: data train predict test
