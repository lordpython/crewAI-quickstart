[tool.poetry]
name = "crewai-streamlit-sequential-quickstart"
version = "0.0.1"
description = ""
authors = ["Alex Fazio <alessandro.fazio@me.com>"]

[tool.poetry.scripts]
crcrewai-streamlit-sequential-quickstart = "main:main"

[tool.poetry.dependencies]
python = ">=3.10,<=3.13"
pydantic = "*"
crewai = "*"
streamlit = "*"
pandas = "*"
watchdog = "*"
setuptools = "*"
python-decouple = "*"
langchain-community = "*"
langchain_groq = "*"
langchain-anthropic = "*"

[[tool.poetry.packages]]
include = "*.py"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.masonry.api"