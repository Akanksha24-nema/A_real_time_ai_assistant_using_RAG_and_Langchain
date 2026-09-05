from langchain_community.llms import Ollama
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

#declaring model
llm = Ollama(model="llama3:8b")

#declaring search tool
search = DuckDuckGoSearchRun()

# defining chat prompt template"

prompt= ChatPromptTemplate.from_template(
    """you are an helpful assistant. You have to give answers "only" on the given search result.
      If you can't find the answer in the search result, say "I don't know".

Search_results:
{context}

Question:
{question}
"""
)

#chain for LLM
runnable = RunnablePassthrough.assign(
    context = lambda x: search.run(x["question"])
)

chain = runnable|prompt|llm

print("Hello I am your assistant. How can I help you today?")

while True:
    try:
        user_query= input("You: ")
        if user_query.upper() in ['EXIT','QUIT','BYE']:
            print('Assistant: Goodbye! have a nice day.')
            break

        response = chain.invoke({'question': user_query})
        print(f"Assistant:{response}")

    except Exception as e:
        print(f"Assisant: sorry, error: {e} occured. Please try again.")



    