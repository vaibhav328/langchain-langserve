import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import Tool, DuckDuckGoSearchResults
from langchain.prompts import PromptTemplate
from langchain_cohere import ChatCohere
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, AgentType

load_dotenv()

ddg_search = DuckDuckGoSearchResults()

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:90.0) Gecko/20100101 Firefox/90.0'
}

def parse_html(content) -> str:
    soup = BeautifulSoup(content, 'html.parser')
    text_content_with_links = soup.get_text()
    return text_content_with_links

def fetch_web_page(url: str) -> str:
    response = requests.get(url, headers=HEADERS)
    return parse_html(response.content)

web_fetch_tool = Tool.from_function(
    func=fetch_web_page,
    name="WebFetcher",
    description="Fetches the content of a web page"
)

prompt_template = "summarize the following content: {content}"
llm = ChatCohere()

chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)

summarize_tool= Tool.from_function(
    func=chain.run,
    name="Summarizer",
    description="Summarizes a web page"
)

tools = [ddg_search, web_fetch_tool, summarize_tool]

agent = initialize_agent(
    tools=tools,
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    llm=llm,
    verbose=True
)

# prompt = "Research how to make best sandwich. Use your tools to search and summarize your research into steps so that a beginner can also make a sandwich"
prompt = "search for linked-in jobs go through the website and give me position and job location in json format give me around 10 values"
print(agent.run(prompt))

