"""Tools for intro.ipynb"""

import requests
import datetime

from tapeagents.tools.simple_browser import SimpleTextBrowser


def get_stock_ticker(company_name: str):
    """Get company stock ticker from its name."""
    yfinance = "https://query2.finance.yahoo.com/v1/finance/search"
    user_agent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/108.0.0.0 Safari/537.36"
    params = {"q": company_name, "quotes_count": 1, "country": "United States"}

    res = requests.get(url=yfinance, params=params, headers={"User-Agent": user_agent})
    data = res.json()

    company_code = data["quotes"][0]["symbol"]
    return company_code


def get_stock_data(symbol: str, start_date: str, end_date: str):
    """Get stock proces for a given symbol and date range.

    Args:
        symbol (str): Stock symbol.
        start_date (str): Start date in the format 'YYYY-MM-DD'.
        end_date (str): End date in the format 'YYYY-MM-DD'.

    Returns:
        List of tuples: Each tuple contains a 'YYYY-MM-DD' date and the stock price.
    """
    symbol = symbol.upper()
    # parse timestamps using datetime
    start_timestamp = int(datetime.datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?period1={start_timestamp}&period2={end_timestamp}&interval=1d"

    try:
        # make a request to Yahoo Finance API
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        data = response.json()

        timestamps = data["chart"]["result"][0]["timestamp"]
        prices = data["chart"]["result"][0]["indicators"]["quote"][0]["close"]
        while len(timestamps) > 100:
            timestamps = timestamps[::2]
            prices = prices[::2]

        return list(
            zip(
                [datetime.datetime.fromtimestamp(ts, datetime.timezone.utc).strftime("%Y-%m-%d") for ts in timestamps],
                prices,
            )
        )
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def search_web(query: str) -> list[dict]:
    return SimpleTextBrowser().get_search_results(query)


def read_page(url: str) -> str:
    return SimpleTextBrowser().get_whole_document(url)
