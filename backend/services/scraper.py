from playwright.sync_api import sync_playwright
import json
import random
import os
import time
from datetime import datetime

#========= CONFIG =========
COOKIES_FILE = "cookies.json"

def save_cookies(context):
	"""Save cookies to file after manual login"""
	cookies = context.cookies()
	with open(COOKIES_FILE, "w") as f:
		json.dump(cookies, f)
	print(f"Cookies saved to {COOKIES_FILE}")

def load_cookies(context):
	"""Load cookies from file if available"""
	if os.path.exists(COOKIES_FILE):
		with open(COOKIES_FILE, "r") as f:
			cookies = json.load(f)
		context.add_cookies(cookies)
		print(f"Cookies loaded from {COOKIES_FILE}")
		return True
	return False

def scrape_tweets(topic: str, limit: int = 20):
	print(f"Scraping tweets for {topic}")

	with sync_playwright() as p:
		# 1. Launch browser
		browser = p.chromium.launch(headless=True)
		# browser = p.chromium.launch(headless=False)

		# 2. Context
		context = browser.new_context(
			viewport={'width': 1280, 'height': 800},
			user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
		)
		
		# Load cookies if available
		cookies_loaded = load_cookies(context)
		
		page = context.new_page()

		# 3. Go to the search page
		print("Navigating to Twitter...")
		page.goto(f"https://x.com/search?q={topic}", timeout=60000)

		# 4. Wait for the content
		print("Waiting for content...")
		try:
			page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
		except Exception as e:
			print("Error: Twitter requires login.")
			print(f"Details: {e}")
			
			if not cookies_loaded:
				print("\n" + "="*50)
				print("No cookies found. Please log in manually.")
				print("After logging in, the cookies will be saved automatically.")
				print("="*50 + "\n")
				
				# Wait for user to log in manually
				page.goto("https://x.com/login", timeout=60000)
				
				# Wait for successful login (home timeline appears)
				print("Waiting for you to complete login...")
				try:
					page.wait_for_url("**/home", timeout=0)  # No timeout - wait as long as needed for captcha
					print("Login detected! Saving cookies...")
					save_cookies(context)
					
					# Now try the search again
					print("Retrying search...")
					page.goto(f"https://x.com/search?q={topic}", timeout=60000)
					page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
				except Exception as login_error:
					print(f"Login timeout or error: {login_error}")
					page.screenshot(path="error_screenshot.png")
					browser.close()
					return
			else:
				print("Cookies expired or invalid. Deleting old cookies...")
				os.remove(COOKIES_FILE)
				page.screenshot(path="error_screenshot.png")
				browser.close()
				print("Please run the script again to log in.")
				return

		# 5. The Scroll
		# We need to trigger lazy loading
		tweets_data = []
		scroll_attempts = 0
		max_scrolls = max(10, limit // 3)

		while len(tweets_data) < limit and scroll_attempts < max_scrolls:
			print(f"Scrolling... (Collected {len(tweets_data)}/{limit})")

			# Extract tweets visible on screen
			articles = page.query_selector_all('article[data-testid="tweet"]')

			for article in articles:
				try:
					# Get text
					text_element = article.query_selector('div[data-testid="tweetText"]')
					text = text_element.inner_text() if text_element else "No Text"

					# Get User Handle
					user_element = article.query_selector('div[data-testid="User-Name"]')
					user_text = user_element.inner_text() if user_element else "Unknown"
					
					handle = "@" + user_text.split('@')[-1].strip() if '@' in user_text else "Unknown"

					# Avoid duplicates
					if text not in [t['text'] for t in tweets_data]:
						tweets_data.append({
							"handle": handle,
							"text": text,
							"timestamp": datetime.now().isoformat()
						})
				except Exception as e:
					continue  # skip broken tweets

			# Scroll down by a random amount
			page.evaluate("window.scrollBy(0,1000)")
			time.sleep(random.uniform(1.5, 3))  # Random sleep = Human Behaviour
			scroll_attempts += 1

		# 6. Save Data
		print(f"Scrape Complete. Found {len(tweets_data)} tweets.")

		# Ensure data directory exists (in project root)
		project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
		data_dir = os.path.join(project_root, "data")
		os.makedirs(data_dir, exist_ok=True)

		# Sanitize topic for filename (replace spaces with underscores)
		safe_topic = topic.replace(" ", "_")
		filename = os.path.join(data_dir, f"tweets_{safe_topic}.json")
		with open(filename, "w", encoding="utf-8") as f:
			json.dump(tweets_data, f, indent=4, ensure_ascii=False)

		print(f"Saved to {filename}")

		browser.close()

	return tweets_data

if __name__ == "__main__":
	import sys
	if len(sys.argv) >= 2:
		topic = sys.argv[1]
		limit = int(sys.argv[2]) if len(sys.argv) >= 3 else 20
		scrape_tweets(topic, limit)