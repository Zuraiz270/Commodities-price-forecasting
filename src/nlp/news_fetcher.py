"""
News Fetching and Caching Module

Fetch commodity-related news from various sources for sentiment analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from pathlib import Path
import hashlib


@dataclass
class NewsArticle:
    """Represents a news article."""
    title: str
    description: str
    source: str
    published_at: datetime
    url: str
    content: Optional[str] = None
    

class NewsCache:
    """
    Cache for news articles to avoid repeated API calls.
    """
    
    def __init__(self, cache_dir: str = ".news_cache"):
        """
        Initialize cache.
        
        Args:
            cache_dir: Directory to store cached articles
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def _get_cache_key(self, query: str, start_date: str, end_date: str) -> str:
        """Generate cache key from query parameters."""
        key = f"{query}_{start_date}_{end_date}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(
        self,
        query: str,
        start_date: str,
        end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Get cached articles if available.
        
        Returns:
            DataFrame of articles or None if not cached
        """
        cache_key = self._get_cache_key(query, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.csv"
        
        if cache_file.exists():
            return pd.read_csv(cache_file, parse_dates=["published_at"])
        return None
    
    def save(
        self,
        articles: pd.DataFrame,
        query: str,
        start_date: str,
        end_date: str
    ):
        """Save articles to cache."""
        cache_key = self._get_cache_key(query, start_date, end_date)
        cache_file = self.cache_dir / f"{cache_key}.csv"
        articles.to_csv(cache_file, index=False)


class NewsFetcher:
    """
    Fetch commodity-related news from various APIs.
    
    Supports:
    - NewsAPI (newsapi.org)
    - GDELT (gdeltproject.org)
    - RSS feeds
    """
    
    # Default commodity keywords
    COMMODITY_KEYWORDS = {
        "aluminum": [
            "aluminum price", "aluminium market", "aluminum tariff",
            "bauxite", "alumina", "aluminum smelter", "LME aluminum",
            "aluminum supply", "aluminum demand"
        ],
        "copper": [
            "copper price", "copper market", "copper tariff",
            "copper mine", "copper supply", "LME copper",
            "copper demand", "copper stocks"
        ],
        "zinc": [
            "zinc price", "zinc market", "zinc mine",
            "LME zinc", "zinc supply", "zinc demand"
        ],
        "general_metals": [
            "base metals", "non-ferrous metals", "metal prices",
            "commodity prices", "industrial metals", "LME"
        ]
    }
    
    def __init__(
        self,
        api_key: str = None,
        cache: NewsCache = None,
        use_cache: bool = True
    ):
        """
        Initialize news fetcher.
        
        Args:
            api_key: NewsAPI key (optional, can use env var)
            cache: NewsCache instance
            use_cache: Whether to use caching
        """
        self.api_key = api_key
        self.cache = cache or NewsCache() if use_cache else None
    
    def fetch_newsapi(
        self,
        query: str,
        from_date: str,
        to_date: str,
        language: str = "en",
        sort_by: str = "relevancy"
    ) -> List[NewsArticle]:
        """
        Fetch news from NewsAPI.
        
        Args:
            query: Search query
            from_date: Start date (YYYY-MM-DD)
            to_date: End date (YYYY-MM-DD)
            language: Language code
            sort_by: Sort order
            
        Returns:
            List of NewsArticle objects
        """
        if not self.api_key:
            raise ValueError(
                "NewsAPI key required. Get one at https://newsapi.org"
            )
        
        try:
            from newsapi import NewsApiClient
            
            newsapi = NewsApiClient(api_key=self.api_key)
            
            response = newsapi.get_everything(
                q=query,
                from_param=from_date,
                to=to_date,
                language=language,
                sort_by=sort_by,
                page_size=100
            )
            
            articles = []
            for article in response.get("articles", []):
                articles.append(NewsArticle(
                    title=article.get("title", ""),
                    description=article.get("description", ""),
                    source=article.get("source", {}).get("name", ""),
                    published_at=datetime.fromisoformat(
                        article.get("publishedAt", "").replace("Z", "+00:00")
                    ),
                    url=article.get("url", ""),
                    content=article.get("content", "")
                ))
            
            return articles
            
        except ImportError:
            raise ImportError(
                "newsapi-python not installed. Install with: pip install newsapi-python"
            )
    
    def fetch_commodity_news(
        self,
        commodity: str,
        from_date: str,
        to_date: str,
        include_general: bool = True
    ) -> pd.DataFrame:
        """
        Fetch news for a specific commodity.
        
        Args:
            commodity: Commodity name ('aluminum', 'copper', 'zinc')
            from_date: Start date
            to_date: End date
            include_general: Include general metals news
            
        Returns:
            DataFrame with news articles
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(commodity, from_date, to_date)
            if cached is not None:
                print(f"Using cached news for {commodity}")
                return cached
        
        # Build query from keywords
        keywords = self.COMMODITY_KEYWORDS.get(commodity, [commodity])
        if include_general:
            keywords.extend(self.COMMODITY_KEYWORDS["general_metals"])
        
        query = " OR ".join(f'"{kw}"' for kw in keywords[:5])  # API limits
        
        articles = self.fetch_newsapi(query, from_date, to_date)
        
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "title": a.title,
                "description": a.description,
                "source": a.source,
                "published_at": a.published_at,
                "url": a.url,
                "content": a.content,
                "headline": f"{a.title}. {a.description}"
            }
            for a in articles
        ])
        
        # Cache results
        if self.cache and not df.empty:
            self.cache.save(df, commodity, from_date, to_date)
        
        return df
    
    def create_sample_news(
        self,
        commodity: str = "aluminum",
        n_samples: int = 100,
        start_date: str = "2024-01-01",
        end_date: str = "2024-12-31"
    ) -> pd.DataFrame:
        """
        Create sample news data for testing.
        
        Args:
            commodity: Target commodity
            n_samples: Number of sample articles
            start_date: Start date
            end_date: End date
            
        Returns:
            DataFrame with synthetic news
        """
        np.random.seed(42)
        
        # Sample headlines
        positive_headlines = [
            f"{commodity.title()} prices surge on strong demand outlook",
            f"Global {commodity} shortage expected to boost prices",
            f"Manufacturing boom drives {commodity} demand higher",
            f"Analysts upgrade {commodity} price forecasts",
            f"Infrastructure spending to boost {commodity} consumption"
        ]
        
        negative_headlines = [
            f"{commodity.title()} prices tumble amid oversupply concerns",
            f"Trade tensions weigh on {commodity} market",
            f"Weak demand sends {commodity} prices lower",
            f"New {commodity} mines to flood market with supply",
            f"Economic slowdown hits {commodity} demand"
        ]
        
        neutral_headlines = [
            f"{commodity.title()} prices stable amid mixed signals",
            f"Market awaits clarity on {commodity} outlook",
            f"{commodity.title()} trades in narrow range",
            f"Investors cautious on {commodity} positioning",
            f"{commodity.title()} inventory levels unchanged"
        ]
        
        # Generate random dates
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        dates = pd.date_range(start, end, freq="D")
        
        # Create samples
        data = []
        for _ in range(n_samples):
            sentiment_type = np.random.choice(
                ["positive", "negative", "neutral"],
                p=[0.3, 0.3, 0.4]
            )
            
            if sentiment_type == "positive":
                headline = np.random.choice(positive_headlines)
            elif sentiment_type == "negative":
                headline = np.random.choice(negative_headlines)
            else:
                headline = np.random.choice(neutral_headlines)
            
            data.append({
                "headline": headline,
                "title": headline.split(".")[0],
                "description": headline,
                "source": np.random.choice([
                    "Reuters", "Bloomberg", "Metal Bulletin",
                    "Mining.com", "Fastmarkets"
                ]),
                "published_at": np.random.choice(dates),
                "url": f"https://example.com/{np.random.randint(10000)}"
            })
        
        return pd.DataFrame(data).sort_values("published_at")


class GDELTFetcher:
    """
    Fetch news from GDELT (Global Database of Events, Language, and Tone).
    
    GDELT is free and provides global news coverage.
    """
    
    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    
    def fetch(
        self,
        query: str,
        mode: str = "artlist",
        max_records: int = 250,
        timespan: str = "7d"
    ) -> pd.DataFrame:
        """
        Fetch news from GDELT.
        
        Args:
            query: Search query
            mode: Output mode ('artlist', 'timelinevol')
            max_records: Maximum records to return
            timespan: Time span (e.g., '7d', '1m')
            
        Returns:
            DataFrame with articles
        """
        import requests
        
        params = {
            "query": query,
            "mode": mode,
            "maxrecords": max_records,
            "timespan": timespan,
            "format": "json"
        }
        
        response = requests.get(self.BASE_URL, params=params)
        
        if response.status_code != 200:
            raise Exception(f"GDELT API error: {response.status_code}")
        
        data = response.json()
        articles = data.get("articles", [])
        
        df = pd.DataFrame([
            {
                "title": a.get("title", ""),
                "headline": a.get("title", ""),
                "source": a.get("domain", ""),
                "published_at": pd.to_datetime(a.get("seendate", "")),
                "url": a.get("url", ""),
                "language": a.get("language", "en"),
                "tone": a.get("tone", 0)  # GDELT's tone score
            }
            for a in articles
        ])
        
        return df
