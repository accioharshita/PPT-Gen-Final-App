import requests
import re
from src.ppt_flow.llm_config import llm
from crewai_tools import SerperDevTool
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from tenacity import retry, wait_exponential, stop_after_attempt
import logging

logger = logging.getLogger(__name__)

def validate_presentation_content(content: str) -> bool:
    """Validates that the content meets the 30-slide requirement with proper formatting."""
    try:
        # Check if content is just a confirmation message
        if len(content) < 500 or "has been reviewed" in content.lower():
            logger.error("Content appears to be a confirmation message instead of actual slides")
            return False

        # Check for exactly 30 slides
        slides = re.findall(r'### Slide \d+:', content)
        if len(slides) != 30:
            logger.error(f"Expected 30 slides, found {len(slides)}")
            return False

        # Check for 6 bullet points per slide
        slide_contents = content.split('### Slide')[1:]
        for i, slide in enumerate(slide_contents, 1):
            bullet_points = re.findall(r'^\s*[•\-\*]\s+.+', slide, re.MULTILINE)
            if len(bullet_points) != 6:
                logger.error(f"Slide {i} has {len(bullet_points)} bullet points instead of 6")
                return False

        return True
    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        return False



def check_link(url):
    """Returns True if the link is valid, False if it's broken (404)."""
    try:
        response = requests.get(url, timeout=5)
        return response.status_code < 400  # Treat any 4xx or 5xx as broken
    except requests.RequestException:
        return False

def extract_links(text):
    """Extracts all links from markdown content."""
    return re.findall(r'\[.*?\]\((https?://.*?)\)', text)

@retry(wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
def find_better_example(query):
    """Search for a new relevant example or case study with a valid link."""
    search_tool = SerperDevTool()
    results = search_tool.run(query + " case study OR example")

    if results and "organic" in results and len(results["organic"]) > 0:
        for result in results["organic"]:
            if "link" in result and "snippet" in result:
                return {
                    "link": result["link"],
                    "summary": result["snippet"]  # Short description of the example
                }
    return None


def validate_and_replace_examples(content):
    """Checks all links, and if broken, replaces them with a new example or case study."""
    links = extract_links(content)
    for link in links:
        if not check_link(link):
            print(f"Broken link found: {link}")
            query = link.split("/")[-1]  # Extract a keyword from the URL
            
            # Instead of searching for the same example, find a new relevant case study
            better_example = find_better_example(query)
            
            if better_example:
                new_text = f"\n**New Example:** {better_example['summary']} \n[Read More]({better_example['link']})"
                print(f"Replacing broken example with a new one: {better_example['link']}")
                content = content.replace(link, better_example['link'])  # Replace broken link
                content += new_text  # Append the new example
            else:
                print(f"No replacement found for {link}. Keeping placeholder.")
                content = content.replace(link, "[NO_VALID_SOURCE_FOUND]")
    
    return content

@CrewBase
class Writers():
    """Writers crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def slide_content_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['slide_content_writer'],
            llm=llm,
            verbose=True,
            memory=True
        )

    @agent
    def final_reviewer(self) -> Agent:
        return Agent(
            config=self.agents_config['final_reviewer'],
            tools=[SerperDevTool()],  # Add Serper tool
            llm=llm,
            verbose=True,
            memory=True
        )

    @task
    @retry(wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
    # def content_writing_task(self) -> Task:
    #     return Task(
    #         config=self.tasks_config['content_writing_task'],
    #         output_file='write-1.md'
    #     )
    def content_writing_task(self) -> Task:
        def validate_content(content):
            if not validate_presentation_content(content):
                raise ValueError("Generated content does not meet requirements")
            return content

        return Task(
            config=self.tasks_config['content_writing_task'],
            output_file='write-1.md',
            function=validate_content
        )

    # @task
    # def review_task(self) -> Task:
    #     """Reviews content, finds and replaces broken links with new relevant examples."""
    #     return Task(
    #         config=self.tasks_config['review_task'],
    #         function=self.review_links,  # Ensure function correctly returns content
    #         output_file='final_reviewed.md'
    #     )

    # @retry(wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(5))
    # def review_links(self, content):
    #     """Validate links and replace broken ones with new examples."""
    #     print("Before review:", content)  # Debugging log
    #     updated_content = validate_and_replace_examples(content)
    #     print("After review:", updated_content)  # Debugging log
        
    #     return updated_content  # Ensure modified content is returned

    @task
    @retry(wait=wait_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
    def review_task(self) -> Task:
        """Reviews content, finds and replaces broken links with new relevant examples."""
        def review_and_validate(content):
            logger.info("Starting content review")
            if not validate_presentation_content(content):
                raise ValueError("Content validation failed before link review")
                
            updated_content = validate_and_replace_examples(content)
            
            if not validate_presentation_content(updated_content):
                raise ValueError("Content validation failed after link review")
                
            return updated_content

        return Task(
            config=self.tasks_config['review_task'],
            function=review_and_validate,
            output_file='final_reviewed.md'
        )

   
    @crew
    def crew(self) -> Crew:
        """Creates the Writers crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True
        )




