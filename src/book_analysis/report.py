#!/usr/bin/env python3
"""Print a report on gutenberg.books as JSON: how many books have a summary, which job arrays wrote them, and their genres."""

import json
import logging
import os

from pymongo import MongoClient
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)

# How many of the last summarized books to list
LAST_SUMMARIZED = 3


def runs_by_job(books):
    """Return one entry per Slurm job array: its job ID and how many books its tasks summarized."""
    return list(books.aggregate([
        {"$unwind": "$runs"},                       # one row per run instead of one per book
        {"$match": {"runs.job": {"$ne": None}}},    # only the runs in a Slurm job array
        # "4211873_5" -> "4211873": the job ID of the array, without the task ID
        {"$group": {"_id": {"$arrayElemAt": [{"$split": ["$runs.job", "_"]}, 0]}, "books": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]))


def genres(books):
    """Return how many books have each genre, the most common first."""
    return list(books.aggregate([
        {"$match": {"llm_analysis": {"$exists": True}}},
        {"$group": {"_id": "$llm_analysis.genre", "books": {"$sum": 1}}},
        {"$sort": {"books": -1}},
    ]))


def last_summarized(books):
    """Return title, author, genre, and summary of the books that were summarized last."""
    projection = {"title": 1, "author": 1, "llm_analysis.genre": 1, "llm_analysis.summary": 1}
    return list(books.find({"llm_analysis": {"$exists": True}}, projection).sort("llm_analysis.at", -1).limit(LAST_SUMMARIZED))


def main():
    """Print the report as one JSON document."""
    # Atlas settings from your environment variables, as in Lab 05
    client = MongoClient(os.environ["MONGODB_ATLAS_URL"], username=os.environ["MONGODB_ATLAS_USER"],
                         password=os.environ["MONGODB_ATLAS_PWD"])
    try:
        books = client["gutenberg"]["books"]
        result = {
            "documents": books.count_documents({}),
            "summarized": books.count_documents({"llm_analysis": {"$exists": True}}),
            "runs_by_job": runs_by_job(books),
            "genres": genres(books),
            "last_summarized": last_summarized(books),
        }
        print(json.dumps(result, indent=2, default=str))
    except PyMongoError as e:
        logger.error("Cannot read from MongoDB: %s", e)
        raise
    finally:
        client.close()  # closes the connections to Atlas, also after an error


if __name__ == "__main__":
    main()
