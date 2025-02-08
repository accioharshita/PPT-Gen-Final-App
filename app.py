__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os
import sys
import streamlit as st
from langtrace_python_sdk import langtrace
from src.ppt_flow.crews.researchers.researchers import Researchers
from src.ppt_flow.crews.writers.writers import Writers
from crewai.flow.flow import Flow, start, listen
import logging
from typing import Optional, Dict
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
import json
import re
import io
from tenacity import retry, stop_after_attempt, wait_exponential
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
SCOPES = [
    "https://www.googleapis.com/auth/presentations",
    "https://www.googleapis.com/auth/drive"
]
TEMPLATE_ID = "10muavbFdRofRMVp6D8RFLFIaQxdJIqoKaQzu7xKh_FU"

# Set page config must be the first Streamlit command
st.set_page_config(
    page_title="PPT Generator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS styles
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        margin-top: 1rem;
        background-color: #FF4B4B;
        color: white;
    }
    .stTextInput>div>div>input {
        background-color: #f0f2f6;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    h1 {
        color: #FF4B4B;
        font-size: 2.5rem !important;
    }
    h3 {
        color: #31333F;
    }
    .api-key-warning {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

def get_services():
    """Gets Google Slides and Drive services using service account credentials from Streamlit secrets."""
    try:
        # Get service account info from Streamlit secrets
        service_account_info = st.secrets["google_service_account"]
        
        # Create credentials
        credentials = service_account.Credentials.from_service_account_info(
            service_account_info,
            scopes=SCOPES
        )
        
        # Build services
        slides_service = build('slides', 'v1', credentials=credentials)
        drive_service = build('drive', 'v3', credentials=credentials)
        
        return slides_service, drive_service
    except Exception as e:
        logger.error(f"Error getting Google services: {e}")
        st.error("Failed to authenticate with Google services. Please check if the service account credentials are properly configured in Streamlit secrets.")
        raise

def copy_presentation(drive_service, template_id, new_title):
    """Creates a copy of the template presentation and returns its ID."""
    try:
        copy = drive_service.files().copy(
            fileId=template_id, body={'name': new_title}
        ).execute()
        return copy.get("id")
    except HttpError as error:
        logger.error(f"Error copying template: {error}")
        raise

def extract_links(text):
    """Extracts hyperlinks from markdown-style text and replaces them with display text."""
    pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    links = [(match.group(1), match.group(2)) for match in re.finditer(pattern, text)]
    text = re.sub(pattern, r'\1', text)
    return text, links

def parse_markdown(file_path):
    """Parses markdown slides and extracts structured data."""
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
    
    slides = re.split(r'### Slide (\d+): (.+)', content)[1:]
    slide_data = []
    
    for i in range(0, len(slides), 3):
        slide_number = int(slides[i].strip())
        title = slides[i + 1].strip()
        content, links = extract_links(slides[i + 2].strip())
        slide_data.append((slide_number, title, content, links))
    
    return sorted(slide_data, key=lambda x: x[0])

def create_slide(service, presentation_id, title, content, links):
    """Creates a slide with formatted text and adds links to both the slide and speaker notes."""
    
    # Create a new slide
    requests = [{'createSlide': {'slideLayoutReference': {'predefinedLayout': 'TITLE_AND_BODY'}}}]
    response = service.presentations().batchUpdate(presentationId=presentation_id, body={'requests': requests}).execute()
    slide_id = response['replies'][0]['createSlide']['objectId']

    # Get the slide
    slide = service.presentations().pages().get(presentationId=presentation_id, pageObjectId=slide_id).execute()
    title_id, body_id = None, None

    # Find the title and body placeholders
    for element in slide.get('pageElements', []):
        shape = element.get('shape', {})
        placeholder = shape.get('placeholder', {})
        if placeholder.get('type') == 'TITLE':
            title_id = element['objectId']
        elif placeholder.get('type') == 'BODY':
            body_id = element['objectId']

    requests = []

    # Add title and format it (24pt)
    if title_id:
        requests.append({
            'insertText': {
                'objectId': title_id,
                'text': title
            }
        })
        requests.append({
            'updateTextStyle': {
                'objectId': title_id,
                'style': {
                    'fontSize': {'magnitude': 24, 'unit': 'PT'}
                },
                'textRange': {
                    'type': 'ALL'
                },
                'fields': 'fontSize'
            }
        })

    # Process content and format text
    if body_id:
        formatted_text = ""
        for line in content.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'\*\*(.*?)\*\*', r'\1', line)
            
            if line.endswith(':'):
                formatted_text += f"{line}\n"
            else:
                formatted_text += f"• {line}\n"

        requests.append({
            'insertText': {
                'objectId': body_id,
                'text': formatted_text
            }
        })

        requests.append({
            'updateTextStyle': {
                'objectId': body_id,
                'style': {
                    'fontSize': {'magnitude': 14, 'unit': 'PT'}
                },
                'textRange': {
                    'type': 'ALL'
                },
                'fields': 'fontSize'
            }
        })

        paragraphs = formatted_text.split('\n')
        current_index = 0
        
        for paragraph in paragraphs:
            if paragraph.endswith(':'):
                requests.append({
                    'updateTextStyle': {
                        'objectId': body_id,
                        'style': {
                            'bold': True
                        },
                        'textRange': {
                            'type': 'FIXED_RANGE',
                            'startIndex': current_index,
                            'endIndex': current_index + len(paragraph)
                        },
                        'fields': 'bold'
                    }
                })
            current_index += len(paragraph) + 1

        if links:
            for link_text, link_url in links:
                start_index = formatted_text.find(link_text)
                if start_index != -1:
                    requests.append({
                        'updateTextStyle': {
                            'objectId': body_id,
                            'style': {
                                'link': {'url': link_url}
                            },
                            'textRange': {
                                'type': 'FIXED_RANGE',
                                'startIndex': start_index,
                                'endIndex': start_index + len(link_text)
                            },
                            'fields': 'link'
                        }
                    })

    if links:
        requests.append({
            'createShape': {
                'objectId': f"{slide_id}_notes",
                'shapeType': 'TEXT_BOX',
                'elementProperties': {
                    'pageObjectId': slide_id,
                    'size': {'width': {'magnitude': 400, 'unit': 'PT'}, 
                            'height': {'magnitude': 100, 'unit': 'PT'}},
                    'transform': {
                        'scaleX': 1,
                        'scaleY': 1,
                        'translateX': 50,
                        'translateY': 400,
                        'unit': 'PT'
                    }
                }
            }
        })
        
        speaker_notes_text = "Links:\n" + "\n".join([f"{text}: {url}" for text, url in links])
        requests.append({
            'insertText': {
                'objectId': f"{slide_id}_notes",
                'text': speaker_notes_text
            }
        })

    service.presentations().batchUpdate(presentationId=presentation_id, body={'requests': requests}).execute()

def export_presentation(drive_service, presentation_id, output_path):
    """Exports the Google Slides presentation as a PPTX file."""
    request = drive_service.files().export_media(
        fileId=presentation_id,
        mimeType='application/vnd.openxmlformats-officedocument.presentationml.presentation'
    )
    file = io.BytesIO()
    downloader = MediaIoBaseDownload(file, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    file.seek(0)
    with open(output_path, 'wb') as f:
        f.write(file.read())
    logger.info(f"Presentation saved to {output_path}")


class EduFlow(Flow):
    def __init__(self, input_variables: Optional[Dict] = None):
        super().__init__()
        self.input_variables = input_variables or {}
        self._validate_input()
        self.llm = get_llm(self.input_variables.get("model"))
        logger.info(f"Initialized EduFlow with variables: {self.input_variables}")

    def _validate_input(self):
        if not self.input_variables.get("topic"):
            raise ValueError("Topic is required in input_variables")

    @start()
    def generate_reseached_content(self):
        try:
            logger.info("Starting research phase")
            researchers = Researchers(model_name=self.input_variables.get('model'))
            research_output = researchers.crew().kickoff(self.input_variables)
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
            
            writers = Writers(model_name=self.input_variables.get('model'))
            writer_output = writers.crew().kickoff(combined_input)
            
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
            file_name = f"{topic}_presentation.md".replace(" ", "_").lower()
            output_path = os.path.join(output_dir, file_name)

            logger.info(f"Writing content to {output_path}")
            logger.debug(f"Content preview: {content[:100]}...")

            with open(output_path, "w", encoding='utf-8') as f:
                f.write(content)

            logger.info(f"Content saved successfully to {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"Save phase failed: {str(e)}", exc_info=True)
            raise

    def kickoff(self) -> str:
        result = super().kickoff()
        if isinstance(result, dict) and 'file_path' in result:
            return result['file_path']
        return result


# def create_presentation(md_file_path):
#     """Creates a PowerPoint presentation from the markdown file."""
#     try:
#         presentation_title = os.path.splitext(os.path.basename(md_file_path))[0].replace('_', ' ')
#         output_path = os.path.join(os.path.dirname(md_file_path), f"{presentation_title}.pptx")
        
#         slides_service, drive_service = get_services()
        
#         presentation_id = copy_presentation(drive_service, TEMPLATE_ID, presentation_title)
#         slide_data = parse_markdown(md_file_path)
        
#         for _, title, content, links in slide_data:
#             create_slide(slides_service, presentation_id, title, content, links)
        
#         export_presentation(drive_service, presentation_id, output_path)
#         return output_path
#     except Exception as e:
#         logger.error(f"Error creating presentation: {str(e)}")
#         raise

def handle_rate_limit(func):
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            time.sleep(1)  # Add a 1-second delay between API calls
            return result
        except HttpError as e:
            if e.resp.status == 429:  # Rate limit exceeded
                time.sleep(2)  # Wait longer if we hit the rate limit
                raise  # Let retry handle it
            raise
    return wrapper


def create_presentation(md_file_path):
    """Creates a PowerPoint presentation from the markdown file."""
    try:
        presentation_title = os.path.splitext(os.path.basename(md_file_path))[0].replace('_', ' ')
        output_path = os.path.join(os.path.dirname(md_file_path), f"{presentation_title}.pptx")
        
        slides_service, drive_service = get_services()
        
        presentation_id = copy_presentation(drive_service, TEMPLATE_ID, presentation_title)
        slide_data = parse_markdown(md_file_path)
        
        # Add progress bar
        progress_bar = st.progress(0)
        total_slides = len(slide_data)
        
        for idx, (_, title, content, links) in enumerate(slide_data):
            try:
                # Add rate limiting to the batch update
                @handle_rate_limit
                def execute_batch_update(requests):
                    return slides_service.presentations().batchUpdate(
                        presentationId=presentation_id, 
                        body={'requests': requests}
                    ).execute()
                
                # Create slide with rate-limited batch updates
                create_slide(slides_service, presentation_id, title, content, links)
                progress = (idx + 1) / total_slides
                progress_bar.progress(progress)
                time.sleep(1)  # Add delay between slides
                
            except Exception as e:
                st.warning(f"⚠️ Retrying slide {idx + 1} due to: {str(e)}")
                time.sleep(2)  # Wait before retry
                try:
                    create_slide(slides_service, presentation_id, title, content, links)
                except Exception as retry_error:
                    st.error(f"❌ Failed to create slide {idx + 1}: {str(retry_error)}")
                    continue
        
        progress_bar.progress(1.0)
        export_presentation(drive_service, presentation_id, output_path)
        return output_path
    except Exception as e:
        logger.error(f"Error creating presentation: {str(e)}")
        raise


# Sidebar configuration
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    PPT Generator is an AI-powered tool that helps you create professional presentations with ease. 
    Simply enter your topic, and our AI agents will:
    - Research your topic thoroughly
    - Generate well-structured content
    - Create a polished presentation
    
    Check this out to know more on API Keys-
    - [Open AI API Key](https://platform.openai.com/docs/quickstart)
    - [Serper API Key](https://docs.mindmac.app/how-to.../internet-browsing/get-serper-key)
    """)
    st.markdown("---")
    
    
    st.header("⚙️ Configuration")
    st.markdown("---")

    # Add model selection
    model_choice = st.selectbox(
        "🤖 Select GPT Model",
        options=["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        index=0,  # Default to first option
        key="model_choice"
    )

    openai_api_key = st.text_input("🔑 OpenAI API Key", type="password", key="openai_key")
    if openai_api_key:
        os.environ["OPENAI_API_KEY"] = openai_api_key

    serper_api_key = st.text_input("🔑 Serper API Key", type="password", key="serper_key")
    if serper_api_key:
        os.environ["SERPER_API_KEY"] = serper_api_key

    st.markdown("---")
    with st.expander("🔧 Advanced Settings"):
        user_api_key = st.text_input("LangTrace API Key (Optional)", type="password")
        api_key = user_api_key.strip() if user_api_key else os.getenv("LANGTRACE_API_KEY")
        if api_key:
            langtrace.init(api_key=api_key)

# Main content
st.title("📄 PPT Generator")
st.markdown("### Transform Your Ideas into Professional Presentations")

# # Initialize session state
# if 'markdown_path' not in st.session_state:
#     st.session_state.markdown_path = None
# if 'presentation_path' not in st.session_state:
#     st.session_state.presentation_path = None

# col1, col2 = st.columns([3, 1])

# with col1:
#     topic = st.text_input(
#         "What would you like to create a presentation about?",
#         placeholder="Enter your topic here...",
#     )

# with col2:
#     st.write("")  # Spacing
#     generate_button = st.button("🚀 Generate Content", use_container_width=True)

# # Handle content generation
# if generate_button:
#     if not topic.strip():
#         st.error("🎯 Please enter a topic to generate content.")
#     elif not openai_api_key or not serper_api_key:
#         st.error("🔑 Please enter both OpenAI and Serper API Keys in the sidebar.")
#     else:
#         with st.spinner("🎨 Generating presentation content..."):
#             try:
#                 input_variables = {"topic": topic}
#                 edu_flow = EduFlow(input_variables)
#                 st.session_state.markdown_path = edu_flow.kickoff()

#                 if st.session_state.markdown_path and os.path.exists(st.session_state.markdown_path):
#                     with open(st.session_state.markdown_path, "r", encoding="utf-8") as file:
#                         markdown_content = file.read()
                    
#                     with st.expander("📑 Generated Content", expanded=True):
#                         st.markdown(markdown_content, unsafe_allow_html=True)

#                         # Add markdown download button
#                         st.download_button(
#                             label="📥 Download Markdown",
#                             data=markdown_content,
#                             file_name=f"{topic}_presentation.md",
#                             mime="text/markdown")
                
#                 else:
#                     st.error("❌ Failed to generate markdown content. Please try again.")
#             except Exception as e:
#                 st.error(f"❌ An error occurred: {str(e)}")

# # Show create presentation button only if markdown content exists
# if st.session_state.markdown_path and os.path.exists(st.session_state.markdown_path):
#     create_ppt_button = st.button("🎯 Create PowerPoint Presentation", use_container_width=True)
    
#     if create_ppt_button:
#         with st.spinner("🎨 Creating PowerPoint presentation..."):
#             try:
#                 st.session_state.presentation_path = create_presentation(st.session_state.markdown_path)
#                 st.success(f"✅ Presentation created successfully! Saved to: {st.session_state.presentation_path}")
                
#                 # Create download button
#                 with open(st.session_state.presentation_path, "rb") as file:
#                     btn = st.download_button(
#                         label="📥 Download Presentation",
#                         data=file,
#                         file_name=os.path.basename(st.session_state.presentation_path),
#                         mime="application/vnd.openxmlformats-officedocument.presentationml.presentation"
#                     )
#             except Exception as e:
#                 st.error(f"❌ An error occurred while creating the presentation: {str(e)}")

if 'markdown_path' not in st.session_state:
    st.session_state.markdown_path = None
if 'presentation_path' not in st.session_state:
    st.session_state.presentation_path = None
if 'markdown_content' not in st.session_state:
    st.session_state.markdown_content = None

col1, col2 = st.columns([3, 1])

with col1:
    topic = st.text_input(
        "What would you like to create a presentation about?",
        placeholder="Enter your topic here...",
    )

with col2:
    st.write("")  # Spacing
    generate_button = st.button("🚀 Generate Content", use_container_width=True)

# Handle content generation
if generate_button:
    if not topic.strip():
        st.error("🎯 Please enter a topic to generate content.")
    elif not openai_api_key or not serper_api_key:
        st.error("🔑 Please enter both OpenAI and Serper API Keys in the sidebar.")
    else:
        with st.spinner("🎨 Generating presentation content..."):
            try:
                input_variables = {"topic": topic, "model": st.session_state.model_choice}
                edu_flow = EduFlow(input_variables)
                st.session_state.markdown_path = edu_flow.kickoff()

                if st.session_state.markdown_path and os.path.exists(st.session_state.markdown_path):
                    with open(st.session_state.markdown_path, "r", encoding="utf-8") as file:
                        st.session_state.markdown_content = file.read()
                    
                    with st.expander("📑 Generated Content", expanded=True):
                        st.markdown(st.session_state.markdown_content, unsafe_allow_html=True)
                else:
                    st.error("❌ Failed to generate markdown content. Please try again.")
            except Exception as e:
                st.error(f"❌ An error occurred: {str(e)}")

# Create a container for the download buttons
if st.session_state.markdown_content:
    download_col1, download_col2 = st.columns(2)
    
    with download_col1:
        # Markdown download button
        st.empty()
        st.download_button(
            label="📥 Download Markdown",
            data=st.session_state.markdown_content,
            file_name=f"{topic}_presentation.md",
            mime="text/markdown",
            use_container_width=True
        )
    
    with download_col2:
        # Create PowerPoint button
        create_ppt_button = st.button("🎯 Create PowerPoint Presentation", use_container_width=True)
        
        if create_ppt_button:
            with st.spinner("🎨 Creating PowerPoint presentation..."):
                try:
                    st.session_state.presentation_path = create_presentation(st.session_state.markdown_path)
                    st.success(f"✅ Presentation created successfully! Saved to: {st.session_state.presentation_path}")
                except Exception as e:
                    st.error(f"❌ An error occurred while creating the presentation: {str(e)}")

# PowerPoint download button (shows up after presentation is created)
if st.session_state.presentation_path and os.path.exists(st.session_state.presentation_path):
    with open(st.session_state.presentation_path, "rb") as file:
        st.download_button(
            label="📥 Download Presentation",
            data=file,
            file_name=os.path.basename(st.session_state.presentation_path),
            mime="application/vnd.openxmlformats-officedocument.presentationml.presentation",
            use_container_width=True
        )
