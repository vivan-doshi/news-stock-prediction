"""
News Analysis and Brainstorming for Better Event Selection
Analyzes current news types and proposes improved selection strategies
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class NewsAnalyzer:
    """Analyzes news data and proposes better event selection strategies"""
    
    def __init__(self, data_dir="../01-data"):
        self.data_dir = Path(data_dir)
        
    def analyze_news_types(self, ticker):
        """Analyze what types of news we have"""
        print(f"\n{'='*60}")
        print(f"NEWS TYPE ANALYSIS FOR {ticker}")
        print(f"{'='*60}")
        
        # Load full news dataset
        news_file = self.data_dir / f"{ticker}_eodhd_news.csv"
        df = pd.read_csv(news_file)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"\n📊 DATASET OVERVIEW:")
        print(f"  Total news articles: {len(df):,}")
        print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        print(f"  Unique dates: {df['date'].dt.date.nunique():,}")
        print(f"  Average articles per day: {len(df) / df['date'].dt.date.nunique():.1f}")
        
        # Analyze tags
        print(f"\n🏷️  TAGS ANALYSIS:")
        all_tags = []
        for tags_str in df['tags'].dropna():
            if isinstance(tags_str, str) and tags_str.strip():
                try:
                    tags = eval(tags_str) if tags_str.startswith('[') else [tags_str]
                    all_tags.extend(tags)
                except:
                    all_tags.append(tags_str)
        
        if all_tags:
            tag_counts = Counter(all_tags)
            print(f"  Most common tags:")
            for tag, count in tag_counts.most_common(10):
                print(f"    {tag}: {count}")
        
        # Analyze sentiment distribution
        print(f"\n😊 SENTIMENT ANALYSIS:")
        sentiment_stats = df['sentiment_polarity'].describe()
        print(f"  Mean sentiment: {sentiment_stats['mean']:.3f}")
        print(f"  Sentiment std: {sentiment_stats['std']:.3f}")
        print(f"  Positive sentiment (>0.5): {(df['sentiment_polarity'] > 0.5).sum():,}")
        print(f"  Negative sentiment (<-0.5): {(df['sentiment_polarity'] < -0.5).sum():,}")
        print(f"  Neutral sentiment (-0.5 to 0.5): {((df['sentiment_polarity'] >= -0.5) & (df['sentiment_polarity'] <= 0.5)).sum():,}")
        
        # Analyze content length
        df['content_length'] = df['content'].str.len()
        print(f"\n📝 CONTENT ANALYSIS:")
        print(f"  Mean content length: {df['content_length'].mean():.0f} characters")
        print(f"  Median content length: {df['content_length'].median():.0f} characters")
        print(f"  Short articles (<500 chars): {(df['content_length'] < 500).sum():,}")
        print(f"  Long articles (>2000 chars): {(df['content_length'] > 2000).sum():,}")
        
        return df
    
    def categorize_news_by_keywords(self, df, ticker):
        """Categorize news by keywords to understand types"""
        print(f"\n🔍 NEWS CATEGORIZATION BY KEYWORDS:")
        
        categories = {
            'Earnings': [
                'earnings', 'quarterly', 'financial results', 'revenue', 'profit',
                'q1', 'q2', 'q3', 'q4', 'eps', 'guidance', 'forecast'
            ],
            'Product Launches': [
                'launch', 'release', 'unveil', 'announce', 'introduce', 'new product',
                'iphone', 'ipad', 'mac', 'watch', 'airpods' if ticker == 'AAPL' else
                'model', 'cybertruck', 'semi', 'roadster', 'gigafactory'
            ],
            'Executive Changes': [
                'ceo', 'executive', 'leadership', 'management', 'board', 'director',
                'appoint', 'resign', 'retire', 'successor', 'interim'
            ],
            'Financial News': [
                'stock split', 'dividend', 'buyback', 'share repurchase', 'ipo',
                'acquisition', 'merger', 'deal', 'investment', 'funding'
            ],
            'Legal/Regulatory': [
                'lawsuit', 'settlement', 'court', 'regulatory', 'sec', 'fda',
                'investigation', 'fine', 'penalty', 'compliance', 'antitrust'
            ],
            'Market Analysis': [
                'analyst', 'rating', 'upgrade', 'downgrade', 'target price',
                'recommendation', 'outlook', 'forecast', 'prediction'
            ],
            'Technology': [
                'technology', 'innovation', 'patent', 'research', 'development',
                'software', 'hardware', 'ai', 'artificial intelligence', 'machine learning'
            ],
            'Competition': [
                'competitor', 'competition', 'market share', 'versus', 'vs',
                'google', 'microsoft', 'amazon', 'samsung' if ticker == 'AAPL' else
                'ford', 'gm', 'toyota', 'bmw', 'mercedes'
            ]
        }
        
        category_counts = {}
        for category, keywords in categories.items():
            # Create pattern for matching
            pattern = '|'.join(keywords)
            mask = df['title'].str.contains(pattern, case=False, na=False) | \
                   df['content'].str.contains(pattern, case=False, na=False)
            category_counts[category] = mask.sum()
            print(f"  {category}: {mask.sum():,} articles")
        
        return category_counts
    
    def analyze_current_selection(self, ticker):
        """Analyze our current event selection"""
        print(f"\n🎯 CURRENT EVENT SELECTION ANALYSIS:")
        
        # Load current major events
        major_events_file = self.data_dir / f"{ticker}_major_events.csv"
        if major_events_file.exists():
            major_events = pd.read_csv(major_events_file)
            major_events['date'] = pd.to_datetime(major_events['date'])
            
            print(f"  Current major events: {len(major_events)}")
            print(f"  Date range: {major_events['date'].min().date()} to {major_events['date'].max().date()}")
            
            # Analyze sentiment of selected events
            print(f"\n📊 SELECTED EVENTS SENTIMENT:")
            sentiment_stats = major_events['sentiment_polarity'].describe()
            print(f"  Mean sentiment: {sentiment_stats['mean']:.3f}")
            print(f"  Positive events: {(major_events['sentiment_polarity'] > 0).sum()}")
            print(f"  Negative events: {(major_events['sentiment_polarity'] < 0).sum()}")
            
            # Show sample titles
            print(f"\n📰 SAMPLE SELECTED EVENTS:")
            for i, (_, event) in enumerate(major_events.head(5).iterrows()):
                print(f"  {i+1}. {event['title'][:80]}...")
                print(f"     Sentiment: {event['sentiment_polarity']:.3f}")
        
        return major_events_file.exists()
    
    def brainstorm_improved_selection(self, df, ticker):
        """Brainstorm improved event selection strategies"""
        print(f"\n💡 BRAINSTORMING IMPROVED EVENT SELECTION:")
        
        strategies = {}
        
        # Strategy 1: Sentiment-based selection
        print(f"\n1. 📊 SENTIMENT-BASED SELECTION:")
        extreme_positive = df[df['sentiment_polarity'] > 0.8]
        extreme_negative = df[df['sentiment_polarity'] < -0.8]
        print(f"   Extreme positive (>0.8): {len(extreme_positive)} events")
        print(f"   Extreme negative (<-0.8): {len(extreme_negative)} events")
        strategies['sentiment'] = pd.concat([extreme_positive, extreme_negative])
        
        # Strategy 2: Content quality-based
        print(f"\n2. 📝 CONTENT QUALITY SELECTION:")
        quality_threshold = df['content_length'].quantile(0.8)  # Top 20% by length
        high_quality = df[df['content_length'] >= quality_threshold]
        print(f"   High-quality articles (>{quality_threshold:.0f} chars): {len(high_quality)} events")
        strategies['quality'] = high_quality
        
        # Strategy 3: Specific event types
        print(f"\n3. 🎯 SPECIFIC EVENT TYPES:")
        
        # Earnings announcements
        earnings_pattern = r'(q[1-4]|quarterly|earnings).*(report|results|announcement)'
        earnings = df[df['title'].str.contains(earnings_pattern, case=False, na=False, regex=True)]
        print(f"   Earnings announcements: {len(earnings)} events")
        strategies['earnings'] = earnings
        
        # Product launches (more specific)
        if ticker == 'AAPL':
            product_pattern = r'(iphone|ipad|mac|watch|airpods).*(launch|release|unveil|announce)'
        else:  # TSLA
            product_pattern = r'(model|cybertruck|semi|roadster|gigafactory).*(launch|release|unveil|announce)'
        
        products = df[df['title'].str.contains(product_pattern, case=False, na=False, regex=True)]
        print(f"   Product launches: {len(products)} events")
        strategies['products'] = products
        
        # CEO/Executive changes
        executive_pattern = r'(ceo|executive|leadership).*(appoint|resign|retire|change|successor)'
        executives = df[df['title'].str.contains(executive_pattern, case=False, na=False, regex=True)]
        print(f"   Executive changes: {len(executives)} events")
        strategies['executives'] = executives
        
        # Strategy 4: Combined criteria
        print(f"\n4. 🔄 COMBINED CRITERIA:")
        
        # High sentiment + High quality
        combined = df[
            ((df['sentiment_polarity'] > 0.7) | (df['sentiment_polarity'] < -0.7)) &
            (df['content_length'] > df['content_length'].quantile(0.7))
        ]
        print(f"   High sentiment + High quality: {len(combined)} events")
        strategies['combined'] = combined
        
        # Strategy 5: Volume-based (if we had trading volume data)
        print(f"\n5. 📈 VOLUME-BASED (CONCEPTUAL):")
        print(f"   Could combine with trading volume spikes")
        print(f"   Would require stock price/volume data alignment")
        
        return strategies
    
    def propose_new_selection_strategies(self, ticker):
        """Propose new selection strategies based on analysis"""
        print(f"\n🚀 PROPOSED NEW SELECTION STRATEGIES:")
        
        strategies = {
            'Strategy A: Rare High-Impact Events': {
                'description': 'Focus on truly rare events with high potential impact',
                'criteria': [
                    'Earnings announcements (Q1-Q4 only)',
                    'Major product launches with "unveil" or "announce"',
                    'CEO/executive changes with "appoint" or "resign"',
                    'Legal settlements with "settle" or "lawsuit"',
                    'Sentiment > 0.8 or < -0.8',
                    'Content length > 1000 characters'
                ],
                'expected_events': '10-20 per year',
                'rationale': 'These events typically move markets significantly'
            },
            
            'Strategy B: News Quality + Sentiment': {
                'description': 'Combine content quality with sentiment strength',
                'criteria': [
                    'Sentiment > 0.7 or < -0.7 (strong sentiment)',
                    'Content length > 75th percentile',
                    'Title contains action words: "announce", "launch", "unveil", "reveal"',
                    'Exclude market analysis and opinion pieces'
                ],
                'expected_events': '30-50 per year',
                'rationale': 'Quality content with strong sentiment signals importance'
            },
            
            'Strategy C: Time-Based + Impact': {
                'description': 'Focus on specific time periods with known high impact',
                'criteria': [
                    'Earnings season (Jan, Apr, Jul, Oct)',
                    'Product launch seasons (Sep for AAPL, various for TSLA)',
                    'Annual shareholder meetings',
                    'Major industry conferences (WWDC, CES, etc.)',
                    'Sentiment > 0.6 or < -0.6'
                ],
                'expected_events': '15-25 per year',
                'rationale': 'Timing matters for market impact'
            },
            
            'Strategy D: Multi-Factor Scoring': {
                'description': 'Score events on multiple dimensions',
                'criteria': [
                    'Sentiment score (0-3 points)',
                    'Content length score (0-2 points)',
                    'Event type score (0-3 points)',
                    'Keyword importance score (0-2 points)',
                    'Total score >= 6 out of 10'
                ],
                'expected_events': '20-40 per year',
                'rationale': 'Balanced approach considering multiple factors'
            }
        }
        
        for strategy_name, details in strategies.items():
            print(f"\n{strategy_name}:")
            print(f"  Description: {details['description']}")
            print(f"  Criteria:")
            for criterion in details['criteria']:
                print(f"    • {criterion}")
            print(f"  Expected Events: {details['expected_events']}")
            print(f"  Rationale: {details['rationale']}")
        
        return strategies
    
    def create_selection_recommendations(self, ticker):
        """Create specific recommendations for event selection"""
        print(f"\n💡 RECOMMENDATIONS FOR {ticker}:")
        
        recommendations = {
            'Immediate Actions': [
                'Implement Strategy A (Rare High-Impact Events) for better signal-to-noise ratio',
                'Focus on earnings announcements with specific quarterly keywords',
                'Include only major product launches with "unveil" or "announce" keywords',
                'Add executive changes with "appoint" or "resign" keywords',
                'Require minimum content length of 1000 characters'
            ],
            
            'Medium-term Improvements': [
                'Develop sentiment scoring based on financial impact keywords',
                'Create industry-specific keyword lists for better categorization',
                'Implement time-based filtering (earnings seasons, product cycles)',
                'Add volume analysis when trading data is available',
                'Create multi-factor scoring system'
            ],
            
            'Long-term Enhancements': [
                'Integrate with stock price data for immediate impact validation',
                'Develop machine learning models for event importance prediction',
                'Create sector-specific event categories',
                'Implement real-time event monitoring',
                'Add international news sources for global impact'
            ]
        }
        
        for category, items in recommendations.items():
            print(f"\n{category}:")
            for item in items:
                print(f"  • {item}")
        
        return recommendations
    
    def run_complete_analysis(self, tickers=['AAPL', 'TSLA']):
        """Run complete news analysis and brainstorming"""
        print(f"\n{'='*80}")
        print(f"COMPREHENSIVE NEWS ANALYSIS & BRAINSTORMING")
        print(f"{'='*80}")
        
        results = {}
        
        for ticker in tickers:
            print(f"\n{'🍎' if ticker == 'AAPL' else '⚡'} ANALYZING {ticker}")
            
            # Analyze news types
            df = self.analyze_news_types(ticker)
            
            # Categorize by keywords
            categories = self.categorize_news_by_keywords(df, ticker)
            
            # Analyze current selection
            has_current = self.analyze_current_selection(ticker)
            
            # Brainstorm improvements
            strategies = self.brainstorm_improved_selection(df, ticker)
            
            # Propose new strategies
            new_strategies = self.propose_new_selection_strategies(ticker)
            
            # Create recommendations
            recommendations = self.create_selection_recommendations(ticker)
            
            results[ticker] = {
                'total_news': len(df),
                'categories': categories,
                'strategies': strategies,
                'recommendations': recommendations
            }
        
        # Summary recommendations
        print(f"\n{'='*80}")
        print(f"SUMMARY RECOMMENDATIONS")
        print(f"{'='*80}")
        
        print(f"\n🎯 TOP PRIORITY STRATEGIES:")
        print(f"  1. Strategy A: Rare High-Impact Events (10-20 events/year)")
        print(f"  2. Focus on earnings, product launches, and executive changes")
        print(f"  3. Require strong sentiment (>0.7 or <-0.7) and quality content")
        
        print(f"\n📊 EXPECTED IMPROVEMENTS:")
        print(f"  • Better signal-to-noise ratio")
        print(f"  • Higher economic impact per event")
        print(f"  • More statistically significant results")
        print(f"  • Clearer business implications")
        
        return results


def main():
    """Run the complete news analysis and brainstorming"""
    analyzer = NewsAnalyzer()
    results = analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
