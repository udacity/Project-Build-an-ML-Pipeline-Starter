SHELL := /bin/bash
PORT ?= 5000
WANDB_ENTITY ?= yannicknkongolo7-wgu
WANDB_PROJECT ?= nyc_airbnb

.PHONY: train
train:
	# Train using the artifact from your nyc_airbnb project
	conda run -n components \
		env WANDB_ENTITY=$(WANDB_ENTITY) WANDB_PROJECT=$(WANDB_PROJECT) \
		mlflow run -e main src/train_random_forest \
		  -P trainval_artifact='yannicknkongolo7-wgu/nyc_airbnb/clean_sample.csv:latest' \
		  -P val_size=0.2 \
		  -P rf_config='rf_config.json' \
		  -P max_tfidf_features=500 \
		  -P output_artifact='rf_model' \
		  -P random_seed=42 \
		  -P stratify_by='neighbourhood_group' \
		  --env-manager=local

.PHONY: serve
serve:
	conda run -n components mlflow models serve \
	  -m src/train_random_forest/random_forest_dir \
	  --env-manager=local -p $(PORT)

.PHONY: stop
stop:
	- kill $$(lsof -ti :$(PORT)) || true

.PHONY: predict_http
predict_http:
	conda run -n components python scripts/predict_http.py \
	  --csv src/basic_cleaning/clean_sample.csv \
	  --n_rows 3 \
	  --url http://127.0.0.1:$(PORT)/invocations
