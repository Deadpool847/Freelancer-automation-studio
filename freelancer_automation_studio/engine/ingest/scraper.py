import httpx
from selectolax.parser import HTMLParser
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
import time
from typing import List, Dict, Optional, Callable
from loguru import logger
import asyncio
from pathlib import Path

class WebScraper:
    """Compliant web scraper with robots.txt respect and rate limiting"""
    
    def __init__(self, respect_robots: bool = True, rate_limit: float = 1.0, max_pages: int = 100):
        self.respect_robots = respect_robots
        self.rate_limit = rate_limit
        self.max_pages = max_pages
        self.visited_urls = set()
        self.robots_parsers = {}
        self.last_request_time = {}
        
    def _get_robots_parser(self, base_url: str) -> RobotFileParser:
        """Get or create robots.txt parser for domain"""
        domain = urlparse(base_url).netloc
        
        if domain not in self.robots_parsers:
            rp = RobotFileParser()
            robots_url = f"{urlparse(base_url).scheme}://{domain}/robots.txt"
            
            try:
                rp.set_url(robots_url)
                rp.read()
                self.robots_parsers[domain] = rp
                logger.info(f"Loaded robots.txt from {robots_url}")
            except Exception as e:
                logger.warning(f"Could not load robots.txt: {e}")
                # Create permissive parser
                rp = RobotFileParser()
                self.robots_parsers[domain] = rp
        
        return self.robots_parsers[domain]
    
    def _can_fetch(self, url: str) -> bool:
        """Check if URL can be fetched according to robots.txt"""
        if not self.respect_robots:
            return True
        
        parser = self._get_robots_parser(url)
        return parser.can_fetch("*", url)
    
    def _rate_limit_wait(self, domain: str):
        """Apply rate limiting per domain"""
        if domain in self.last_request_time:
            elapsed = time.time() - self.last_request_time[domain]
            wait_time = (1.0 / self.rate_limit) - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
        
        self.last_request_time[domain] = time.time()
    
    def scrape(self, start_url: str, progress_callback: Optional[Callable] = None) -> List[Dict]:
        """Scrape pages starting from start_url"""
        results = []
        to_visit = [start_url]
        domain = urlparse(start_url).netloc
        
        logger.info(f"Starting scrape of {start_url}")
        
        while to_visit and len(results) < self.max_pages:
            url = to_visit.pop(0)
            
            if url in self.visited_urls:
                continue
            
            if not self._can_fetch(url):
                logger.info(f"Skipping {url} (blocked by robots.txt)")
                continue
            
            # Rate limiting
            self._rate_limit_wait(domain)
            
            try:
                with httpx.Client(timeout=30.0, follow_redirects=True) as client:
                    response = client.get(url, headers={"User-Agent": "FreelancerAutomationStudio/1.0"})
                    
                if response.status_code != 200:
                    logger.warning(f"Status {response.status_code} for {url}")
                    continue
                
                # Parse HTML
                tree = HTMLParser(response.text)
                
                # Extract data
                page_data = {
                    'url': url,
                    'title': tree.css_first('title').text() if tree.css_first('title') else '',
                    'text': ' '.join([node.text() for node in tree.css('p')]),
                    'headers': [h.text() for h in tree.css('h1, h2, h3')],
                    'links': [urljoin(url, a.attributes.get('href', '')) for a in tree.css('a[href]')],
                    'scraped_at': time.time()
                }
                
                results.append(page_data)
                self.visited_urls.add(url)
                
                logger.info(f"Scraped {url} ({len(results)}/{self.max_pages})")
                
                # Find new URLs to visit (same domain only)
                for link in page_data['links']:
                    if urlparse(link).netloc == domain and link not in self.visited_urls:
                        to_visit.append(link)
                
                # Progress callback
                if progress_callback:
                    progress_callback(len(results) / self.max_pages)
                
            except Exception as e:
                logger.error(f"Error scraping {url}: {e}")
                continue
        
        logger.info(f"Scraping complete. Collected {len(results)} pages.")
        return results