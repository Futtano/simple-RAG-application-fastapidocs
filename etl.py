import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

########## Load and split FastAPI Learn pages ############
## Main content of the FastAPI Learn pages is inside <article></article> tags
def keep_article_elements_only(soup: BeautifulSoup) -> str:
    return str(soup.find('article').get_text())

# Regular expression to match all the URLs to extract information from
urls = (
    r'^https://fastapi\.tiangolo\.com/'
    r'(?:python-types|async|environment-variables|virtual-environments|'
    r'tutorial|advanced|fastapi-cli|deployment|how-to)/.*'
)

# Load the article text for all the websites
loader = SitemapLoader(
    web_path="https://fastapi.tiangolo.com/sitemap.xml",
    filter_urls=[urls],
    parsing_function=keep_article_elements_only,
)
docs = loader.load()

# Split documents in chuncks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

########## Embedding and vector store ##########
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
)

vector_store = PineconeVectorStore(
    pinecone_api_key=os.environ.get("PC_API_KEY"),
    index_name='fast-api-learn-pages',
    embedding=embeddings
)

# Add documents to the vector store
# Insert in batches and wait a little bit to avoid rate limits
for i in range(0, len(splits), 100):
    batch = splits[i:i+100]
    vector_store.add_documents(batch)
    os.system('timeout 5 > nul')  # To avoid rate limits