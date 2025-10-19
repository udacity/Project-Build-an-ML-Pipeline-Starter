# --- Quick commands for your pipeline & serving ---

PY := /Users/aaronburton/miniconda3/envs/nyc_airbnb_dev/bin/python
MLFLOW := mlflow
TRACKING := sqlite:///mlflow.db
ARTIFACT_DIR := ./mlruns

export MLFLOW_TRACKING_URI := $(TRACKING)
export MLFLOW_ARTIFACT_URI := $(ARTIFACT_DIR)

.PHONY: all download clean split train test register ui serve-prod ping-prod predict-prod

all:
	$(MLFLOW) run . -P steps=all

download:
	$(MLFLOW) run . -P steps=download

clean:
	$(MLFLOW) run . -P steps=basic_cleaning

split:
	$(MLFLOW) run . -P steps=data_split

train:
	$(MLFLOW) run . -P steps=train_random_forest

test:
	$(MLFLOW) run . -P steps=test_regression_model

register:
	$(MLFLOW) run . -P steps=register_model

ui:
	$(MLFLOW) ui --backend-store-uri $(TRACKING) --host 127.0.0.1 --port 5001

# Serve the Production alias on :5004
serve-prod:
	$(MLFLOW) models serve -m "models:/nyc_airbnb_price_model@prod" --env-manager local --port 5004

# Health check Production server
ping-prod:
	curl -s http://127.0.0.1:5004/ping && echo

# Example prediction to Production server
predict-prod:
	curl -s http://127.0.0.1:5004/invocations \
	  -H 'Content-Type: application/json' \
	  -d '{"dataframe_split":{"columns":["id","name","host_id","host_name","neighbourhood_group","neighbourhood","latitude","longitude","room_type","minimum_nights","number_of_reviews","last_review","reviews_per_month","calculated_host_listings_count","availability_365"],"data":[[123,"Cozy studio",1,"Alice","Brooklyn","Williamsburg",40.71,-73.96,"Entire home/apt",2,10,"2023-09-01",0.2,1,200]]}}' && echo
