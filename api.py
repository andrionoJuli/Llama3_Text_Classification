from fastapi import FastAPI, HTTPException
from classifier import Classify
from utils import parse_labels, map_labels_to_ids

import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

text_classifier = Classify()


@app.post("/generate_labels/")
async def classify_texts(input_text: str):
    labels = text_classifier(input_text)
    logger.info(f"Classified labels: {labels}")
    try:
        logger.info("Parsing labels")
        parsed_labels = parse_labels(labels)
        logger.info(f"Parsed labels: {parsed_labels}")
        logger.info("Mapping labels to ids")
        result = map_labels_to_ids(parsed_labels)
        logger.info(f"Mapped labels based on id: {result}")
        return result
    except Exception as e:
        logger.error(f"Error during text classification: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")





