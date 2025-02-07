#!/usr/bin/env python
from random import randint
import os
from langtrace_python_sdk import langtrace
from pydantic import BaseModel
from typing import Optional, Dict
from crewai.flow.flow import Flow, listen, start
from .crews.researchers.researchers import Researchers
from .crews.writers.writers import Writers
import logging
from tenacity import retry, wait_exponential, stop_after_attempt

api_key = os.getenv('LANGTRACE_API_KEY')
langtrace.init(api_key=api_key)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EduFlow(Flow):
    def __init__(self, input_variables: Optional[Dict] = None):
        super().__init__()
        self.input_variables = input_variables or {}
        self._validate_input()
        logger.info(f"Initialized EduFlow with variables: {self.input_variables}")

    def _validate_input(self):
        if not self.input_variables.get("topic"):
            raise ValueError("Topic is required in input_variables")

    @start()
    def generate_reseached_content(self):
        try:
            logger.info("Starting research phase")
            research_output = Researchers().crew().kickoff(self.input_variables)
            if not research_output or not research_output.raw:
                raise ValueError("Research crew produced no output")
            logger.info(f"Research phase completed. Output preview: {research_output.raw[:100]}...")
            return research_output.raw
        except Exception as e:
            logger.error(f"Research phase failed: {str(e)}", exc_info=True)
            raise

    @listen(generate_reseached_content)
    def generate_educational_content(self, research_content):
        try:
            logger.info("Starting writing phase")
            if not research_content:
                raise ValueError("No research content received from previous phase")
            
            combined_input = {
                **self.input_variables,
                "research_content": research_content
            }
            
            writer_output = Writers().crew().kickoff(combined_input)
            if not writer_output or not writer_output.raw:
                raise ValueError("Writer crew produced no output")
            
            logger.info(f"Writing phase completed. Output preview: {writer_output.raw[:100]}...")
            return writer_output.raw
        except Exception as e:
            logger.error(f"Writing phase failed: {str(e)}", exc_info=True)
            raise

    @listen(generate_educational_content)
    def save_to_markdown(self, content):
        try:
            logger.info("Starting save phase")
            if not content:
                raise ValueError("No content received to save")
            
            output_dir = os.path.abspath("output")
            os.makedirs(output_dir, exist_ok=True)
            
            topic = self.input_variables.get("topic")
            file_name = f"{topic}.md".replace(" ", "_").lower()
            output_path = os.path.join(output_dir, file_name)
            
            logger.info(f"Writing content to {output_path}")
            logger.debug(f"Content preview: {content[:100]}...")
            
            with open(output_path, "w", encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"Content saved successfully to {output_path}")
            return {"status": "success", "file_path": output_path}
        except Exception as e:
            logger.error(f"Save phase failed: {str(e)}", exc_info=True)
            raise


def kickoff(topic: Optional[str] = None):
    if not topic:
        topic = input('Please enter your topic here: ').strip()
        if not topic:
            raise ValueError("Topic cannot be empty")
    
    input_variables = {"topic": topic}
    edu_flow = EduFlow(input_variables)
    return edu_flow.kickoff()

if __name__ == "__main__":
    kickoff()