"""
Improved Event Selection Based on Analysis
Implements Strategy A: Rare High-Impact Events
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class ImprovedEventSelector:
    """Improved event selection based on comprehensive analysis"""
    
    def __init__(self, data_dir="../01-data", output_dir="../03-output"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
    def select_high_impact_events(self, ticker, target_events=50):
        """
        Select high-impact events using Strategy A criteria
        
        Parameters:
        -----------
        ticker : str
            Stock ticker
        target_events : int
            Target number of events to select
        """
        print(f"\n{'='*60}")
        print(f"IMPROVED EVENT SELECTION FOR {ticker}")
        print(f"Target: {target_events} high-impact events")
        print(f"{'='*60}")
        
        # Load full news dataset
        news_file = self.data_dir / f"{ticker}_eodhd_news.csv"
        df = pd.read_csv(news_file)
        df['date'] = pd.to_datetime(df['date'])
        
        print(f"📊 Starting with {len(df):,} total news articles")
        
        # Step 1: Filter by content quality
        print(f"\n1️⃣ CONTENT QUALITY FILTERING:")
        df['content_length'] = df['content'].str.len()
        
        # Remove very short articles (likely low quality)
        quality_threshold = 800  # Minimum content length
        df_quality = df[df['content_length'] >= quality_threshold].copy()
        print(f"   After quality filter (>={quality_threshold} chars): {len(df_quality):,} articles")
        
        # Step 2: Filter by sentiment strength
        print(f"\n2️⃣ SENTIMENT STRENGTH FILTERING:")
        sentiment_threshold = 0.7  # Strong sentiment threshold
        df_sentiment = df_quality[
            (df_quality['sentiment_polarity'] > sentiment_threshold) |
            (df_quality['sentiment_polarity'] < -sentiment_threshold)
        ].copy()
        print(f"   After sentiment filter (|sentiment| > {sentiment_threshold}): {len(df_sentiment):,} articles")
        
        # Step 3: Category-specific filtering
        print(f"\n3️⃣ CATEGORY-SPECIFIC FILTERING:")
        
        selected_events = []
        
        # 3a. Earnings announcements (highest priority)
        earnings_pattern = r'(q[1-4]|quarterly|earnings).*(report|results|announcement|beat|miss)'
        earnings = df_sentiment[
            df_sentiment['title'].str.contains(earnings_pattern, case=False, na=False, regex=True)
        ].copy()
        
        if len(earnings) > 0:
            # Take the most recent earnings (last 16 quarters = 4 years)
            earnings = earnings.sort_values('date').tail(16)
            selected_events.append(earnings)
            print(f"   Earnings announcements: {len(earnings)} events")
        
        # 3b. Major product launches
        if ticker == 'AAPL':
            product_pattern = r'(iphone|ipad|mac|watch|airpods).*(launch|release|unveil|announce|reveal)'
        else:  # TSLA
            product_pattern = r'(model|cybertruck|semi|roadster|gigafactory).*(launch|release|unveil|announce|reveal)'
        
        products = df_sentiment[
            df_sentiment['title'].str.contains(product_pattern, case=False, na=False, regex=True)
        ].copy()
        
        if len(products) > 0:
            # Take major product announcements
            products = products.sort_values('date').tail(20)
            selected_events.append(products)
            print(f"   Product launches: {len(products)} events")
        
        # 3c. Executive changes
        executive_pattern = r'(ceo|executive|leadership|management|board).*(appoint|resign|retire|change|successor|interim|promote)'
        executives = df_sentiment[
            df_sentiment['title'].str.contains(executive_pattern, case=False, na=False, regex=True)
        ].copy()
        
        if len(executives) > 0:
            # Take major executive changes
            executives = executives.sort_values('date').tail(10)
            selected_events.append(executives)
            print(f"   Executive changes: {len(executives)} events")
        
        # 3d. Legal/Regulatory events
        legal_pattern = r'(lawsuit|settlement|court|regulatory|sec|fda|investigation|fine|penalty|antitrust).*(settle|resolve|announce|charge)'
        legal = df_sentiment[
            df_sentiment['title'].str.contains(legal_pattern, case=False, na=False, regex=True)
        ].copy()
        
        if len(legal) > 0:
            # Take major legal events
            legal = legal.sort_values('date').tail(8)
            selected_events.append(legal)
            print(f"   Legal/Regulatory events: {len(legal)} events")
        
        # 3e. Financial events (stock splits, dividends, major deals)
        financial_pattern = r'(stock split|dividend|buyback|share repurchase|ipo|acquisition|merger|deal|investment).*(announce|approve|declare)'
        financial = df_sentiment[
            df_sentiment['title'].str.contains(financial_pattern, case=False, na=False, regex=True)
        ].copy()
        
        if len(financial) > 0:
            # Take major financial events
            financial = financial.sort_values('date').tail(8)
            selected_events.append(financial)
            print(f"   Financial events: {len(financial)} events")
        
        # Step 4: Combine and deduplicate
        print(f"\n4️⃣ COMBINING AND DEDUPLICATING:")
        if selected_events:
            all_selected = pd.concat(selected_events, ignore_index=True)
            # Remove duplicates based on title and date
            all_selected = all_selected.drop_duplicates(subset=['title', 'date'])
            print(f"   Combined events: {len(all_selected)}")
        else:
            print("   ❌ No events found with current criteria!")
            return None
        
        # Step 5: Final selection and ranking
        print(f"\n5️⃣ FINAL SELECTION AND RANKING:")
        
        # Sort by date and select most recent
        all_selected = all_selected.sort_values('date')
        
        # If we have more than target, select the most recent
        if len(all_selected) > target_events:
            all_selected = all_selected.tail(target_events)
        
        print(f"   Final selected events: {len(all_selected)}")
        print(f"   Date range: {all_selected['date'].min().date()} to {all_selected['date'].max().date()}")
        
        # Step 6: Analyze selected events
        print(f"\n6️⃣ SELECTED EVENTS ANALYSIS:")
        print(f"   Positive sentiment events: {(all_selected['sentiment_polarity'] > 0).sum()}")
        print(f"   Negative sentiment events: {(all_selected['sentiment_polarity'] < 0).sum()}")
        print(f"   Mean sentiment: {all_selected['sentiment_polarity'].mean():.3f}")
        print(f"   Mean content length: {all_selected['content_length'].mean():.0f} characters")
        
        # Step 7: Save results
        print(f"\n7️⃣ SAVING RESULTS:")
        
        # Save detailed events
        events_file = f"{ticker}_improved_events.csv"
        all_selected.to_csv(self.data_dir / events_file, index=False)
        
        # Create event dates file for analysis
        event_dates = pd.DataFrame({'Date': all_selected['date'].dt.date.unique()})
        event_dates = event_dates.sort_values('Date')
        dates_file = f"{ticker}_improved_event_dates.csv"
        event_dates.to_csv(self.data_dir / dates_file, index=False)
        
        print(f"   ✓ Saved {len(all_selected)} events to {events_file}")
        print(f"   ✓ Created {dates_file} with {len(event_dates)} unique event dates")
        
        # Show sample events
        print(f"\n📰 SAMPLE SELECTED EVENTS:")
        for i, (_, event) in enumerate(all_selected.tail(5).iterrows()):
            print(f"   {i+1}. {event['title'][:60]}...")
            print(f"      Date: {event['date'].date()}")
            print(f"      Sentiment: {event['sentiment_polarity']:.3f}")
            print(f"      Length: {event['content_length']} chars")
        
        return all_selected
    
    def run_improved_selection(self, tickers=['AAPL', 'TSLA'], target_events=50):
        """Run improved event selection for all tickers"""
        print(f"\n{'='*80}")
        print(f"IMPROVED EVENT SELECTION PROCESS")
        print(f"Strategy A: Rare High-Impact Events")
        print(f"Target: {target_events} events per ticker")
        print(f"{'='*80}")
        
        results = {}
        
        for ticker in tickers:
            print(f"\n{'🍎' if ticker == 'AAPL' else '⚡'} PROCESSING {ticker}")
            selected_events = self.select_high_impact_events(ticker, target_events)
            results[ticker] = selected_events
        
        # Summary
        print(f"\n{'='*80}")
        print(f"IMPROVED SELECTION SUMMARY")
        print(f"{'='*80}")
        
        for ticker, events in results.items():
            if events is not None:
                print(f"\n{ticker}:")
                print(f"  Selected Events: {len(events)}")
                print(f"  Date Range: {events['date'].min().date()} to {events['date'].max().date()}")
                print(f"  Event Frequency: {len(events['date'].dt.date.unique())} unique days")
                print(f"  Positive Events: {(events['sentiment_polarity'] > 0).sum()}")
                print(f"  Negative Events: {(events['sentiment_polarity'] < 0).sum()}")
                print(f"  Mean Sentiment: {events['sentiment_polarity'].mean():.3f}")
            else:
                print(f"\n{ticker}: ❌ Selection failed")
        
        print(f"\n🎯 NEXT STEPS:")
        print(f"  1. Run event study with improved event selection")
        print(f"  2. Compare results with previous analysis")
        print(f"  3. Validate statistical significance")
        print(f"  4. Create updated visualizations")
        
        return results


def main():
    """Run improved event selection"""
    selector = ImprovedEventSelector()
    results = selector.run_improved_selection(target_events=50)


if __name__ == "__main__":
    main()
