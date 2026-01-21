"""
Event Extraction Module

Extract structured market events from news text using LLMs.
Based on research: "Event-Based forecasting where LLMs extract 
structured events from news to aid numerical models."
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json


class EventType(Enum):
    """Types of commodity market events."""
    SUPPLY_DISRUPTION = "supply_disruption"
    DEMAND_CHANGE = "demand_change"
    TARIFF_TRADE = "tariff_trade"
    PRICE_MOVEMENT = "price_movement"
    PRODUCTION_CHANGE = "production_change"
    INVENTORY_CHANGE = "inventory_change"
    POLICY_REGULATION = "policy_regulation"
    ECONOMIC_INDICATOR = "economic_indicator"
    GEOPOLITICAL = "geopolitical"
    WEATHER_DISASTER = "weather_disaster"
    OTHER = "other"


@dataclass
class CommodityEvent:
    """Structured commodity market event."""
    event_type: EventType
    commodity: str
    entity: str  # Company, country, or organization
    sentiment: str  # positive, negative, neutral
    magnitude: float  # Impact intensity 0-1
    description: str
    date: str
    source_text: str
    confidence: float


class EventExtractor:
    """
    Extract structured events from commodity news using LLMs.
    
    Uses GPT-4 or other LLMs to convert unstructured news text
    into structured event representations for model input.
    """
    
    EXTRACTION_PROMPT = '''
You are an expert commodity market analyst. Extract structured events from the following news headline/text.

Text: {text}

Extract the following information in JSON format:
{{
    "event_type": one of [supply_disruption, demand_change, tariff_trade, price_movement, production_change, inventory_change, policy_regulation, economic_indicator, geopolitical, weather_disaster, other],
    "commodity": the commodity mentioned (aluminum, copper, zinc, or general if multiple),
    "entity": the company, country, or organization involved,
    "sentiment": one of [positive, negative, neutral] for price impact,
    "magnitude": float 0-1 indicating impact intensity,
    "description": brief description of the event
}}

Return ONLY valid JSON, no explanation.
'''
    
    def __init__(
        self,
        model: str = "gpt-4",
        api_key: str = None,
        use_local: bool = False
    ):
        """
        Initialize event extractor.
        
        Args:
            model: LLM model name
            api_key: OpenAI API key (or use env var)
            use_local: Use local LLM instead of OpenAI
        """
        self.model = model
        self.api_key = api_key
        self.use_local = use_local
        self.client = None
    
    def _init_client(self):
        """Initialize OpenAI client."""
        if self.use_local:
            return
        
        try:
            from openai import OpenAI
            import os
            
            api_key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError(
                "openai not installed. Install with: pip install openai"
            )
    
    def extract_event(self, text: str) -> Optional[CommodityEvent]:
        """
        Extract structured event from text.
        
        Args:
            text: News text to analyze
            
        Returns:
            CommodityEvent or None if extraction fails
        """
        if not self.use_local and self.client is None:
            self._init_client()
        
        try:
            if self.use_local:
                return self._extract_local(text)
            else:
                return self._extract_openai(text)
        except Exception as e:
            print(f"Event extraction failed: {e}")
            return None
    
    def _extract_openai(self, text: str) -> Optional[CommodityEvent]:
        """Extract using OpenAI API."""
        prompt = self.EXTRACTION_PROMPT.format(text=text)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial analyst."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=500
        )
        
        content = response.choices[0].message.content
        
        # Parse JSON response
        try:
            data = json.loads(content)
            return CommodityEvent(
                event_type=EventType(data.get("event_type", "other")),
                commodity=data.get("commodity", "unknown"),
                entity=data.get("entity", "unknown"),
                sentiment=data.get("sentiment", "neutral"),
                magnitude=float(data.get("magnitude", 0.5)),
                description=data.get("description", ""),
                date="",
                source_text=text,
                confidence=0.8
            )
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Failed to parse response: {content}")
            return None
    
    def _extract_local(self, text: str) -> CommodityEvent:
        """
        Extract using rule-based approach (fallback when no LLM).
        
        Simple keyword matching for basic event extraction.
        """
        text_lower = text.lower()
        
        # Detect event type
        event_type = EventType.OTHER
        if any(kw in text_lower for kw in ["strike", "shutdown", "disruption", "halt"]):
            event_type = EventType.SUPPLY_DISRUPTION
        elif any(kw in text_lower for kw in ["tariff", "trade war", "import", "export"]):
            event_type = EventType.TARIFF_TRADE
        elif any(kw in text_lower for kw in ["surge", "jump", "fall", "drop", "price"]):
            event_type = EventType.PRICE_MOVEMENT
        elif any(kw in text_lower for kw in ["demand", "consumption", "orders"]):
            event_type = EventType.DEMAND_CHANGE
        elif any(kw in text_lower for kw in ["production", "output", "capacity"]):
            event_type = EventType.PRODUCTION_CHANGE
        elif any(kw in text_lower for kw in ["inventory", "stocks", "warehouse"]):
            event_type = EventType.INVENTORY_CHANGE
        
        # Detect commodity
        commodity = "general"
        for com in ["aluminum", "aluminium", "copper", "zinc"]:
            if com in text_lower:
                commodity = "aluminum" if com == "aluminium" else com
                break
        
        # Detect sentiment
        positive_words = ["surge", "rise", "gain", "boost", "growth", "strong", "increase"]
        negative_words = ["fall", "drop", "decline", "weak", "slump", "tumble", "cut"]
        
        pos_count = sum(1 for w in positive_words if w in text_lower)
        neg_count = sum(1 for w in negative_words if w in text_lower)
        
        if pos_count > neg_count:
            sentiment = "positive"
        elif neg_count > pos_count:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        
        return CommodityEvent(
            event_type=event_type,
            commodity=commodity,
            entity="unknown",
            sentiment=sentiment,
            magnitude=0.5,
            description=text[:100],
            date="",
            source_text=text,
            confidence=0.5
        )
    
    def extract_batch(
        self,
        texts: List[str],
        dates: List[str] = None
    ) -> List[CommodityEvent]:
        """
        Extract events from multiple texts.
        
        Args:
            texts: List of news texts
            dates: Optional list of dates
            
        Returns:
            List of CommodityEvent objects
        """
        events = []
        dates = dates or [""] * len(texts)
        
        for text, date in zip(texts, dates):
            event = self.extract_event(text)
            if event:
                event.date = date
                events.append(event)
        
        return events
    
    def events_to_dataframe(
        self,
        events: List[CommodityEvent]
    ) -> pd.DataFrame:
        """Convert events to DataFrame."""
        return pd.DataFrame([asdict(e) for e in events])
    
    def create_event_features(
        self,
        events: List[CommodityEvent],
        date_col: str = "date"
    ) -> pd.DataFrame:
        """
        Create features from events for model input.
        
        Args:
            events: List of extracted events
            date_col: Column name for date
            
        Returns:
            DataFrame with event features
        """
        df = self.events_to_dataframe(events)
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert event_type to string
        df["event_type"] = df["event_type"].apply(lambda x: x.value if hasattr(x, 'value') else x)
        
        # One-hot encode event types
        event_dummies = pd.get_dummies(df["event_type"], prefix="event")
        
        # Sentiment encoding
        df["sentiment_score"] = df["sentiment"].map({
            "positive": 1, "neutral": 0, "negative": -1
        })
        
        # Combine features
        features = pd.concat([
            df[[date_col, "magnitude", "sentiment_score", "confidence"]],
            event_dummies
        ], axis=1)
        
        return features


class EventEmbedder:
    """
    Create embeddings from events for late fusion with price models.
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        use_pretrained: bool = False
    ):
        """
        Initialize embedder.
        
        Args:
            embedding_dim: Dimension of event embeddings
            use_pretrained: Use pretrained text embeddings
        """
        self.embedding_dim = embedding_dim
        self.use_pretrained = use_pretrained
        self.encoder = None
    
    def _init_encoder(self):
        """Initialize sentence transformer."""
        if not self.use_pretrained:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            self.encoder = SentenceTransformer("all-MiniLM-L6-v2")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
    
    def embed_events(
        self,
        events: List[CommodityEvent]
    ) -> np.ndarray:
        """
        Create embeddings for events.
        
        Args:
            events: List of events
            
        Returns:
            Array of embeddings [n_events, embedding_dim]
        """
        if self.use_pretrained:
            if self.encoder is None:
                self._init_encoder()
            
            texts = [e.description for e in events]
            embeddings = self.encoder.encode(texts)
            return embeddings
        
        else:
            # Simple feature-based embedding
            embeddings = []
            
            for event in events:
                # Encode event type (one-hot, 11 types)
                type_vec = np.zeros(11)
                type_idx = list(EventType).index(event.event_type)
                type_vec[type_idx] = 1
                
                # Encode sentiment
                sent_vec = np.zeros(3)
                sent_map = {"positive": 0, "negative": 1, "neutral": 2}
                sent_vec[sent_map.get(event.sentiment, 2)] = 1
                
                # Combine with magnitude and confidence
                embedding = np.concatenate([
                    type_vec,
                    sent_vec,
                    [event.magnitude],
                    [event.confidence]
                ])
                
                embeddings.append(embedding)
            
            return np.array(embeddings)
    
    def aggregate_daily(
        self,
        events: List[CommodityEvent],
        aggregation: str = "mean"
    ) -> Dict[str, np.ndarray]:
        """
        Aggregate event embeddings by date.
        
        Args:
            events: List of events
            aggregation: 'mean', 'max', or 'sum'
            
        Returns:
            Dict mapping date to aggregated embedding
        """
        embeddings = self.embed_events(events)
        
        # Group by date
        date_embeddings = {}
        for event, emb in zip(events, embeddings):
            date = event.date
            if date not in date_embeddings:
                date_embeddings[date] = []
            date_embeddings[date].append(emb)
        
        # Aggregate
        aggregated = {}
        for date, embs in date_embeddings.items():
            stacked = np.stack(embs)
            if aggregation == "mean":
                aggregated[date] = stacked.mean(axis=0)
            elif aggregation == "max":
                aggregated[date] = stacked.max(axis=0)
            elif aggregation == "sum":
                aggregated[date] = stacked.sum(axis=0)
        
        return aggregated
