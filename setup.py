from setuptools import setup, find_packages

setup(
    name="price_query_engine",
    version="0.1.0",
    description="Natural language interface for cryptocurrency pricing data using the Coin Metrics API",
    author="Price Query Engine Team",
    packages=find_packages(),
    install_requires=[
        "langchain>=0.0.267",
        "langchain-community>=0.0.1",
        "coinmetrics-api-client",
        "anthropic>=0.3.0",
        "pandas",
        "python-dateutil",
    ],
    entry_points={
        "console_scripts": [
            "price-query=price_query_engine.cli:main",
        ],
    },
    python_requires=">=3.7",
)
