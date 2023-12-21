# :notebook: Tutorial: How to Build an Internal Knowledge-Based Chatbot
<img width="412" alt="image" src="https://github.com/tarynduong/chat-bot-streamlit/assets/85856280/dd02a338-e453-4b8f-9096-7740cdceb65b">

**Why can't we just use ChatGPT or Google Bard to answer the question directly?**

Because:
- Commercial LLMs have been trained on a large corpus of data available on the internet. The data has more irrelevant context about your question than you might like.
- The data contained in your internal systems of interest might have not been used when training the LLM. It might be too recent and not available when the model was trained. Or it could be private and not available publicly on the internet. At the time of writing only data up until September 2021 is included in training of OpenAI GPT-3.5 and GPT-4 models.
- Currently available LLMs have been shown to produce hallucinations inventing data that is not true.

**What is an easy to implement alternative?**

Build your own chatbot to answer questions only using knowledge in your internal systems or of organization, in the form of PDF, PPT, DOCX documents.

Step 1: Split text corpus of the entire knowledge base into chunks - a chunk will represent a single piece of context available to be queried. Keep in mind that data of interest can be coming from multiple sources of different types, e.g. documentation in Confluence supplemented by PDF reports.

Step 2: Use the Embedding Model to transform each of the chunks into a vector embedding.

Step 3: Store all vector embeddings in a Vector Database.

Step 4: Save text that represents each of the embeddings separately together with the pointer to the embedding. Embed a question/query you want to ask using the same Embedding Model that was used to embed the knowledge base itself.

Step 5: Use the resulting Vector Embedding to run a query against the index in Vector Database. Choose how many vectors you want to retrieve from the Vector Database - it will equal the amount of context you will be retrieving and eventually using for answering the query question.

Step 6: Vector DB performs an Approximate Nearest Neighbour (ANN) search for the provided vector embedding against the index and returns previously chosen amount of context vectors. The procedure returns vectors that are most similar in a given Embedding/Latent space. 

Step 7: Map the returned Vector Embeddings to the text chunks that represent them.

Step 8: Pass a question together with the retrieved context text chunks to the LLM via prompt. Instruct the LLM to only use the provided context to answer the given question, if there is no data in the retrieved context that could be used, make sure that no made up answer is provided.

**:high_brightness: Features**
- Langchain: a framework for developing applications powered by language models
- Azure OpenAI API: text embedding, GPT model (gpt-35-turbo-16k) to generate answers to user's queries
- FAISS: leverage similarity search and clustering of dense vectors
- TruLens: evaluate and track performance of the chatbot
- UI built using Streamlit: where you can upload an audio file and download the summary exported to a doc file
