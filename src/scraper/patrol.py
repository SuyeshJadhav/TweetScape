import asyncio
from playwright.async_api import async_playwright
import json
import random
import os
from datetime import datetime

#========= CONFIG =========
TOPIC = "ChatGPT"
# Number of tweets to scrape
LIMIT = 20
COOKIES_FILE = "cookies.json"

async def save_cookies(context):
	"""Save cookies to file after manual login"""
	cookies = await context.cookies()
	with open(COOKIES_FILE, "w") as f:
		json.dump(cookies, f)
	print(f"Cookies saved to {COOKIES_FILE}")

async def load_cookies(context):
	"""Load cookies from file if available"""
	if os.path.exists(COOKIES_FILE):
		with open(COOKIES_FILE, "r") as f:
			cookies = json.load(f)
		await context.add_cookies(cookies)
		print(f"Cookies loaded from {COOKIES_FILE}")
		return True
	return False

async def scrape_tweets():
	print(f"Scraping tweets for {TOPIC}")

	async with async_playwright() as p:
		# 1. Launch browser
		# browser = await p.chromium.launch(headless=True)
		browser = await p.chromium.launch(headless=False)

		# 2. Context
		context = await browser.new_context(
			viewport={'width': 1280, 'height': 800},
			user_agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
		)
		
		# Load cookies if available
		cookies_loaded = await load_cookies(context)
		
		page = await context.new_page()

		# 3. Go to the search page
		print("Navigating to Twitter...")
		await page.goto(f"https://x.com/search?q={TOPIC}", timeout=60000)

		# 4. Wait for the content
		print("Waiting for content...")
		try:
			await page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
		except Exception as e:
			print("Error: Twitter requires login.")
			print(f"Details: {e}")
			
			if not cookies_loaded:
				print("\n" + "="*50)
				print("No cookies found. Please log in manually.")
				print("After logging in, the cookies will be saved automatically.")
				print("="*50 + "\n")
				
				# Wait for user to log in manually
				await page.goto("https://x.com/login", timeout=60000)
				
				# Wait for successful login (home timeline appears)
				print("Waiting for you to complete login...")
				try:
					await page.wait_for_url("**/home", timeout=0)  # No timeout - wait as long as needed for captcha
					print("Login detected! Saving cookies...")
					await save_cookies(context)
					
					# Now try the search again
					print("Retrying search...")
					await page.goto(f"https://x.com/search?q={TOPIC}", timeout=60000)
					await page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
				except Exception as login_error:
					print(f"Login timeout or error: {login_error}")
					await page.screenshot(path="error_screenshot.png")
					await browser.close()
					return
			else:
				print("Cookies expired or invalid. Deleting old cookies...")
				os.remove(COOKIES_FILE)
				await page.screenshot(path="error_screenshot.png")
				await browser.close()
				print("Please run the script again to log in.")
				return

		# 5. The Scroll
		# We need to trigger lazy loading
		tweets_data = []
		scroll_attempts = 0

		while len(tweets_data) < LIMIT and scroll_attempts < 10:
			print(f"Scrolling... (Collected {len(tweets_data)}/{LIMIT})")

			# Extract tweets visible on screen
			articles = await page.query_selector_all('article[data-testid="tweet"]')

			for article in articles:
				try:
					# Get text
					text_element = await article.query_selector('div[data-testid="tweetText"]')
					text = await text_element.inner_text() if text_element else "No Text"

					# Get User Handle
					user_element = await article.query_selector('div[data-testid="User-Name"]')
					user_text = await user_element.inner_text() if user_element else "Unknown"
					
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
			await page.evaluate("window.scrollBy(0,1000)")
			await asyncio.sleep(random.uniform(1.5, 3))  # Random sleep = Human Behaviour
			scroll_attempts += 1

		# 6. Save Data
		print(f"Scrape Complete. Found {len(tweets_data)} tweets.")

		# Ensure data directory exists
		os.makedirs("data", exist_ok=True)

		filename = f"data/tweets_{TOPIC}.json"
		with open(filename, "w", encoding="utf-8") as f:
			json.dump(tweets_data, f, indent=4, ensure_ascii=False)

		print(f"Saved to {filename}")

		await browser.close()

if __name__ == "__main__":
	asyncio.run(scrape_tweets())