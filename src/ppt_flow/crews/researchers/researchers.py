from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool
from pydantic import BaseModel
from typing import List
from src.ppt_flow.llm_config import get_llm
# from tenacity import retry, wait_exponential, stop_after_attempt


@CrewBase
class Researchers():
	"""Researchers crew"""

	agents_config = 'config/agents.yaml'
	tasks_config = 'config/tasks.yaml'

	def __init__(self, model_name=None):
	        super().__init__()
	        self.llm = get_llm(model_name) 

	@agent
	def topic_explorer(self) -> Agent:
		search_tool= SerperDevTool()
		return Agent(
			config=self.agents_config['topic_explorer'],
			tools=[search_tool],
			llm= self.llm,
			verbose=True
		)

	@agent
	def indepth_researcher(self) -> Agent:
		search_tool= SerperDevTool()
		return Agent(
			config=self.agents_config['indepth_researcher'],
			tools=[search_tool],
			llm= self.llm,
			verbose=True,
			memory=True
		)
		

	@task
	def topic_exploration_task(self) -> Task:
		return Task(
			config=self.tasks_config['topic_exploration_task'],
			output_file= 'slides.md'
		)

	@task
	def detailed_research_task(self) -> Task:
		return Task(
			config=self.tasks_config['detailed_research_task'],
			output_file='depth.md'
		)
	

	@crew
	def crew(self) -> Crew:
		"""Creates the Researchers crew"""
		# To learn how to add knowledge sources to your crew, check out the documentation:
		# https://docs.crewai.com/concepts/knowledge#what-is-knowledge

		return Crew(
			agents=self.agents, # Automatically created by the @agent decorator
			tasks=self.tasks, # Automatically created by the @task decorator
			process=Process.sequential,
			verbose=True,
			# process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
		)

