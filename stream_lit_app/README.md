## Adding commands how to run this on your local device

create a .env file inyour project structure and paste below code and ensure to replace with your API keys

PINECONE_API_KEY=your-pinecone-api-key
OPENAI_API_KEY=your-openai-api-key

To build
command: docker build -t <name-of-container> .
example: docker build -t qa-bot .

To execute:
command: docker run -p 8501:8501 --env-file .env <name-of-container>
example: docker run -p 8501:8501 --env-file .env qa-bot

After running to view on local browser:
command: docker run -p 8501:8501 --env-file .env qa-bot