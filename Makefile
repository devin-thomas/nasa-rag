IMAGE_NAME := nasa-rag
CHROMA_DIR := chroma_db_openai
COLLECTION := nasa_space_missions_text

.PHONY: install index stats run evaluate test docker-build docker-run

install:
	python -m pip install -r requirements.txt

index:
	python embedding_pipeline.py --data-path ./data_text --chroma-dir ./$(CHROMA_DIR) --collection-name $(COLLECTION)

stats:
	python embedding_pipeline.py --chroma-dir ./$(CHROMA_DIR) --collection-name $(COLLECTION) --stats-only

run:
	streamlit run chat.py

evaluate:
	python batch_evaluate.py --chroma-dir ./$(CHROMA_DIR) --collection-name $(COLLECTION) --output ./evaluation_report.json

test:
	python -m pytest

docker-build:
	docker build -t $(IMAGE_NAME) .

docker-run:
	docker run --rm -p 8501:8501 --env-file .env -v $$(pwd)/$(CHROMA_DIR):/app/$(CHROMA_DIR) $(IMAGE_NAME)
