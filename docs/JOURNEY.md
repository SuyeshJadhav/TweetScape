# The Builder's Diary: Building Tweetscape 🚀

## Chapter 1: The Spark (How it Started) 🌱

I’ve spent way too much time scrolling through Twitter, trying to figure out what the world actually thinks. Are vaccines actually good? Is Gemini better than ChatGPT?

I’d read hundreds of tweets, trying to sort them in my head: "Okay, this person is a tech enthusiast, this one is skeptical, this one is just trolling..."

But reading hundreds of tweets is exhausting, and it's impossible to keep track of the bigger picture in your head.

I wanted to see the meaningful patterns. I didn't just want to read individual tweets; I wanted a tool that could read thousands of them and show me the landscape—who is saying what, and why. I wanted to turn a stream of noise into something I could actually understand.

## Chapter 2: The Pivots (When Things Didn't Work) 🔄

Building this tool was a constant battle against how messy human language is.

### The "Twitter Language" Problem 🗣️
I started with `TextBlob` because it's the standard for sentiment analysis. Big mistake.
It turns out, Twitter doesn't speak proper English; it speaks "Internet."
TextBlob would look at unstructured grammar, misspelled words, or slang and fail completely. It flagged "sick beat" as negative. It couldn't handle the reality of how people tweet.
I had to switch to **RoBERTa**, a model actually trained on tweets. Suddenly, the app understood that a misspelled, slang-filled rant could actually be "Joy," not "Neutral."

### The "AI Debate" Paradox (t-SNE vs. UMAP) 🤖
My biggest technical challenge happened when I tried to visualize the debate around **Open Source vs. Closed Source AI**. I used **t-SNE** to cluster the data, expecting to see two clear sides: The "Open Weight" supporters vs. the "Safety/Closed" camp.

Instead, I got a single, messy blob.

**The Failure:** t-SNE focuses on local similarities. Both sides were using the exact same keywords: "Impact," "Future," "Risk," "Control." Because the vocabulary was the same, t-SNE lumped everyone into one big "AI Discussion Cluster."

**The Pivot:** I switched to **UMAP** because it preserves the global structure. UMAP understood that while the words were identical, the context was polar opposite.
**The Result:** UMAP physically pulled the clusters apart—Open Source advocates on one side, Safety/Closed Source on the other. It transformed a generic blob of tech talk into a clear visual "Tug of War."

## Chapter 3: The Real Struggles (Engineering Reality) 💥

### The "Spam Click" Disaster 🖱️
During one of my first tests, I got impatient. The scraping felt slow, so I spam-clicked the "Search" button.
*Click. Click. Click.*
Behind the scenes, my **backend** spawned a new `Playwright` browser instance for every single click.
My RAM spiked. My PC froze completely. And to top it off, Twitter saw 20 simultaneous login attempts from my IP and instantly blocked me.
**Lesson Learned:** Debounce your buttons, and strictly manage your browser instances (`scraper.py` now handles errors much better). And maybe don't spam click.

### The RAM Panic 😱
I got greedy with models. I wanted **RoBERTa** for sentiment, **DistilRoBERTa** for emotion, **KeyBERT** for topics...
Initially, I was initializing these models inside my API routes. Big mistake. Every request tried to re-load gigabytes of weights, crashing the server.
**The Fix:** I realized I needed a dedicated "Model Loader." I created `backend/services/models.py` to centralized all model initialization. Now, everything loads *once* at startup. It takes a few seconds to boot, but once it's up, it's fast.

### The Infinite Scroll Trap 🔄
My scraper initially just scrolled down to get more tweets. Simple, right? But Twitter's page is "virtualized"—tweets disappear from the DOM as you scroll past them.
My scraper would "see" 20 tweets, scroll down, and... see the *same* 20 tweets because the page hadn't refreshed yet, but the DOM had changed. It was stuck in an infinite loop of reading the same div.
**The Fix:** I had to implement smart waiting logic in Playwright, checking not just for element presence, but for *new* data, before triggering the next scroll. It turned a flaky script into a reliable data miner.

## Chapter 4: The Agent (Automating Curiosity) 🕵️‍♂️

Even with the map, I found myself doing too much work. I'd search "Gemini", look at the map for 5 minutes, then search "Gemini vs ChatGPT", then "Gemini pricing"...

I got tired of being the search engine.

I integrated **Gemma 3** (via Ollama) to build the Investigator Agent.
Why Gemma 3? I tested a few models (like Llama 3), but Gemma 3 hit the sweet spot. Despite being a larger model, it ran surprisingly fast on my machine and, more importantly, its summaries were just better. It captured the nuance of the clusters in a way smaller models missed.

Now, instead of just searching, the Agent realizes that if I ask about "DeepSeek," it should also check "DeepSeek privacy" and "DeepSeek performance" automatically. It treats the search bar like a conversation, not a database query.

## Chapter 5: The Full Stack Symphony (Building the "Tug of War") 🎻

The backend improved, but JSON data is boring. I needed to *show* the conflict.

### React + D3.js: A Dangerous Dance
I love React, and I respect D3.js, but making them work together is tricky. React wants to control the DOM; D3 wants to control the DOM.
I tried using wrapper libraries like `react-force-graph`, but they were too limiting. I couldn't get that specific "Tug of War" physics where improved sentiment pulls you right, and anger pulls you left.
**The Solution:** I went raw. I built a custom D3 force simulation *inside* a React `useEffect` hook in `ClusterMap.jsx`.
- **Force X**: Mapped directly to the RoBERTa sentiment score (-1 to +1).
- **Force Y**: A gentle gravity to keep things centered but floating.
- **Force Collide**: To stop the bubbles from overlapping (these are tweets, not interactions!).

Now, when you search a polarizing topic like "Pineapple on Pizza," you physically see the bubbles rip apart to opposite sides of the screen. It’s not just a chart; it’s a battlefield.

### Refactoring for Sanity (FastAPI) ⚡
My initial backend was a mess of scripts. As the frontend grew complex, I needed a real API.
Migrating to **FastAPI** made a huge difference. Automatic documentation (Swagger UI), Pydantic validation for my data models, and async support for the scraper.
It turned a "science project" into a deployable **TweetScape** product interactable via a clean REST API.

---