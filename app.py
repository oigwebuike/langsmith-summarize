import os
# from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# LANGCHAIN_ENDPOINT="https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "Summarize"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

groq_api_key = os.getenv("GROQ_API_KEY")
promt = ChatPromptTemplate.from_messages(
    [
    ("system", "You are a virtual assistant, please respond to the user's question(s) based on the given context."),
    ("user", "Question:{question}\n Context:{context}")
])

# llm = ChatOpenAI(model="gpt-3.5-turbo")

llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0,
    groq_api_key=groq_api_key,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    # other params...
)

output_parser = StrOutputParser()
chain = promt | llm | output_parser

question = "Can you summarize the given text"
context = "God of courage, war, bloodshed, and violence. The son of Zeus and Hera, he was depicted as a beardless youth, either nude with a helmet and spear or sword, or as an armed warrior. Homer portrays him as moody and unreliable, and as being the most unpopular god on earth and Olympus (Iliad 5.890–1). He generally represents the chaos of war in contrast to Athena, a goddess of military strategy and skill. Ares is known for cuckolding his brother Hephaestus, conducting an affair with his wife Aphrodite. His sacred animals include vultures, venomous snakes, dogs, and boars. His Roman counterpart Mars by contrast was regarded as the dignified ancestor of the Roman people"


print(chain.invoke({"question":question, "context":context}))