import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.utilities import SQLDatabase
from langchain_community.tools import Tool
from langchain.agents import initialize_agent, AgentType
from sqlalchemy import create_engine, text


def init_env():
    load_dotenv()
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise ValueError("GROQ_API_KEY not found. Please set it as an environment variable.")
    return groq_key


def build_llm(groq_api_key: str) -> ChatOpenAI:
    """
    Create a LangChain ChatOpenAI object using Groq's OpenAI-compatible endpoint.
    """
    llm = ChatOpenAI(
        model="llama3-70b-8192",  # or "llama3-8b-8192"
        temperature=0,
        openai_api_key=groq_api_key,
        openai_api_base="https://api.groq.com/openai/v1",
    )
    return llm


def seed_duckdb(db_path: str = "stocks.duckdb") -> SQLDatabase:
    """
    Initialize a DuckDB database with example stock price data.
    """
    engine = create_engine(f'duckdb:///{db_path}')
    conn = engine.connect()

    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS stock_prices (
            company TEXT,
            closing_price DOUBLE,
            date DATE
        );
    """))

    conn.execute(text("DELETE FROM stock_prices;"))
    conn.execute(text("""
        INSERT INTO stock_prices VALUES
        ('Apple', 185.2, '2024-07-01'),
        ('Apple', 187.1, '2024-07-02'),
        ('Google', 135.6, '2024-07-01'),
        ('Google', 137.3, '2024-07-02'),
        ('NVIDIA', 120.0, '2024-07-01'),
        ('NVIDIA', 125.5, '2024-07-02')
    """))

    return SQLDatabase(engine)


def build_calculator_tool() -> Tool:
    """
    Basic unsafe calculator using eval (demo only).
    In production, use a secure parser.
    """
    def _calc(expr: str) -> str:
        return str(eval(expr, {"__builtins__": {}}, {}))

    return Tool(
        name="Calculator",
        func=_calc,
        description="Useful for making arithmetic calculations for investment decisions."
    )


def maybe_build_search_tool() -> Tool | None:
    """
    Create search tool if SERPAPI key is available.
    """
    serp_key = os.getenv("SERPAPI_API_KEY")
    if not serp_key:
        return None

    search = SerpAPIWrapper(serpapi_api_key=serp_key)
    return Tool(
        name="Search",
        func=search.run,
        description="Search for financial news, stock updates, and macroeconomic context."
    )


def build_agent(llm: ChatOpenAI, db: SQLDatabase):
    """
    Build LangChain agent using SQL, calculator, and optional search tools.
    """
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    sql_tools = toolkit.get_tools()

    tools = sql_tools + [build_calculator_tool()]
    search_tool = maybe_build_search_tool()
    if search_tool:
        tools.append(search_tool)

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    return agent


def demo_queries(agent):
    """
    Demonstration queries for testing the agent.
    """
    qs = [
        "What is the percentage change in Apple's closing price between 2024-07-01 and 2024-07-02?",
        "Compare the latest closing prices of Google and NVIDIA. Which one increased more?",
        "Did Apple stock rise in the last two days? If so, by what percentage?"
    ]
    for q in qs:
        print(f"\nQ: {q}")
        ans = agent.run(q)
        print(f"A: {ans}")


def main():
    groq_key = init_env()
    llm = build_llm(groq_key)
    db = seed_duckdb()
    agent = build_agent(llm, db)

    # Run a single query
    answer = agent.run(
        "Calculate the difference and percentage change in Apple's stock price between July 1st and July 2nd, 2024."
    )
    print("\n--- Single Query Result ---")
    print(answer)

    # Optional: run multiple demo queries
    demo_queries(agent)


if __name__ == "__main__":
    main()