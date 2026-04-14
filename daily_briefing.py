import os
import requests
import feedparser
import pytz
import yfinance as yf
import google.generativeai as genai
from datetime import datetime, timedelta

# ─── 환경변수 (GitHub Actions Secrets에서 가져옴) ─────────────────────
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
GEMINI_API_KEY      = os.getenv("GOOGLE_API_KEY")
CITY                = os.getenv("CITY_NAME", "Seoul,KR")
DISCORD_WEBHOOK     = os.getenv("DISCORD_WEBHOOK_URL")

# ─── RSS 피드 설정 ────────────────────────────────────────────────
NEWS_RSS_URLS = [
    "https://feeds.bbci.co.uk/news/business/rss.xml",
    "https://feeds.bbci.co.uk/news/science_and_environment/rss.xml",
    "https://feeds.bbci.co.uk/news/technology/rss.xml",
]

ECONOMY_RSS_URLS = [
    # 한국 경제/주식/투자 뉴스
    "https://www.hankyung.com/feed/economy",        # 한국경제 - 경제
    "https://www.hankyung.com/feed/finance",         # 한국경제 - 금융/증권
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=^KS11&region=KR&lang=ko-KR",  # Yahoo Finance KOSPI
    "https://www.mk.co.kr/rss/30100041/",            # 매일경제 - 증권
    "https://www.mk.co.kr/rss/30000001/",            # 매일경제 - 경제
    "https://rss.donga.com/economy.xml",             # 동아일보 - 경제
    # 글로벌 경제/투자 뉴스
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",  # CNBC Economy
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=15839069",  # CNBC Investing
    "https://www.investing.com/rss/news.rss",  # Investing.com 뉴스
]

TZ = pytz.timezone("Asia/Seoul")

# ─── API 키 유효성 검사 및 모델 초기화 ────────────────────────────
if not GEMINI_API_KEY:
    raise ValueError("GOOGLE_API_KEY secret is not set.")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY secret is not set.")
if not DISCORD_WEBHOOK:
    raise ValueError("DISCORD_WEBHOOK_URL secret is not set.")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")


# ──────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────

# 1) 날씨 관련 함수
def fetch_weather():
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": CITY, "appid": OPENWEATHER_API_KEY, "units": "metric", "lang": "kr"}
    r = requests.get(url, params=params)
    r.raise_for_status()
    current = r.json()

    forecast_url = "https://api.openweathermap.org/data/2.5/forecast"
    r = requests.get(forecast_url, params=params)
    r.raise_for_status()
    forecast = r.json()

    hourly_temps = []
    now = datetime.now(TZ)
    for item in forecast["list"]:
        dt = datetime.fromtimestamp(item["dt"], TZ)
        if dt <= now + timedelta(hours=24):
            hourly_temps.append({
                "time": dt.strftime("%H:%M"),
                "temp": item["main"]["temp"],
                "icon": item["weather"][0]["icon"],
            })
    return {
        "current": {
            "desc": current["weather"][0]["description"].capitalize(),
            "temp": current["main"]["temp"],
            "feels": current["main"]["feels_like"],
            "humidity": current["main"]["humidity"],
            "icon": current["weather"][0]["icon"],
        },
        "hourly": hourly_temps,
    }


def create_temperature_graph(hourly_temps):
    graph_width = min(len(hourly_temps), 8)
    step = max(1, len(hourly_temps) // graph_width)
    points = hourly_temps[::step][:graph_width]

    temps = [pt["temp"] for pt in points]
    min_temp, max_temp = min(temps), max(temps)
    temp_range = max_temp - min_temp or 1
    max_bar = 20

    lines = []
    for pt in points:
        length = int((pt["temp"] - min_temp) / temp_range * max_bar)
        bars = "█" * length
        lines.append(f"{pt['time']:>5} | {bars:<{max_bar}} {pt['temp']:.1f}°C")
    return "\n".join(lines)


def build_weather_embed(data):
    icon_url = f"https://openweathermap.org/img/wn/{data['current']['icon']}@2x.png"
    title = f"🏙️ {CITY} 오늘의 날씨 ({datetime.now(TZ).strftime('%Y-%m-%d')})"
    graph = create_temperature_graph(data["hourly"])
    hourly_text = f"```\n{graph}\n```"
    return {
        "title": title,
        "description": data["current"]["desc"],
        "color": 0x3498DB,
        "thumbnail": {"url": icon_url},
        "fields": [
            {"name": "🌡️ 현재 온도", "value": f"{data['current']['temp']}°C", "inline": True},
            {"name": "🤗 체감 온도", "value": f"{data['current']['feels']}°C", "inline": True},
            {"name": "💧 습도", "value": f"{data['current']['humidity']}%", "inline": True},
            {"name": "📊 시간별 기온 그래프", "value": hourly_text, "inline": False},
        ],
        "footer": {"text": "Powered by OpenWeatherMap"},
    }


# 2) 뉴스 관련 함수
def fetch_recent_entries(rss_urls):
    now = datetime.now(TZ)
    start = now - timedelta(hours=24)
    entries = []
    for rss_url in rss_urls:
        try:
            feed = feedparser.parse(rss_url)
            category = (
                feed.feed.title
                if hasattr(feed.feed, "title")
                else rss_url.split("/")[-2].replace("_", " ").title()
            )
            for e in feed.entries:
                try:
                    pub_time_struct = e.get("published_parsed") or e.get("updated_parsed")
                    if pub_time_struct:
                        pub = datetime(*pub_time_struct[:6], tzinfo=pytz.utc).astimezone(TZ)
                        if pub >= start and hasattr(e, "title") and hasattr(e, "link"):
                            entries.append(f"- [{category}] {e.title.strip()} ({e.link.strip()})")
                except Exception:
                    continue
        except Exception as feed_error:
            print(f"Error fetching RSS feed {rss_url}: {feed_error}")
            continue
    return entries


def summarize_news_with_gemini(entries, prompt_template):
    if not entries:
        return "최근 24시간 이내 새로운 뉴스가 없습니다."
    prompt = prompt_template + "\n".join(entries)
    try:
        res = model.generate_content(prompt)
        return res.text
    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return "뉴스 요약 생성 중 오류가 발생했습니다."


def build_news_embed(summary):
    return {
        "title": f"📰 세계 뉴스 요약 ({datetime.now(TZ).strftime('%Y-%m-%d')})",
        "description": summary,
        "color": 0x2ECC71,
        "footer": {"text": "Powered by Google Gemini & BBC RSS"},
    }


def build_economy_news_embed(summary):
    return {
        "title": f"💰 경제·주식 뉴스 요약 ({datetime.now(TZ).strftime('%Y-%m-%d')})",
        "description": summary,
        "color": 0xF39C12,
        "footer": {"text": "Powered by Google Gemini & 한경/매경/CNBC RSS"},
    }


def build_market_analysis_embed(analysis):
    return {
        "title": f"📊 시장 트렌드 분석 ({datetime.now(TZ).strftime('%Y-%m-%d')})",
        "description": analysis,
        "color": 0xE74C3C,
        "footer": {"text": "Powered by Google Gemini & 한경/매경/CNBC RSS"},
    }


# 3) 주요 지수 관련 함수
MARKET_INDICES = {
    "^KS11": "KOSPI",
    "^KQ11": "KOSDAQ",
    "^GSPC": "S&P 500",
    "^IXIC": "NASDAQ",
    "^N225": "Nikkei 225",
}


def fetch_market_indices():
    results = []
    for symbol, name in MARKET_INDICES.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev, curr = hist["Close"].iloc[-2], hist["Close"].iloc[-1]
                change_pct = ((curr - prev) / prev) * 100
                arrow = "🔺" if change_pct > 0 else "🔻" if change_pct < 0 else "➖"
                results.append({
                    "name": name, "price": curr,
                    "change_pct": change_pct, "arrow": arrow,
                })
            elif len(hist) == 1:
                results.append({
                    "name": name, "price": hist["Close"].iloc[-1],
                    "change_pct": 0, "arrow": "➖",
                })
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    return results


def build_indices_embed(indices):
    lines = []
    for idx in indices:
        lines.append(
            f"{idx['arrow']} **{idx['name']}** : {idx['price']:,.2f}  ({idx['change_pct']:+.2f}%)"
        )
    description = "\n".join(lines) if lines else "지수 데이터를 가져올 수 없습니다."
    return {
        "title": f"📈 주요 지수 현황 ({datetime.now(TZ).strftime('%Y-%m-%d')})",
        "description": description,
        "color": 0x1ABC9C,
        "footer": {"text": "Powered by Yahoo Finance"},
    }


# 4) 환율 관련 함수
EXCHANGE_RATES = {
    "USDKRW=X": "🇺🇸 USD/KRW",
    "EURKRW=X": "🇪🇺 EUR/KRW",
    "JPYKRW=X": "🇯🇵 JPY/KRW (100엔)",
}


def fetch_exchange_rates():
    results = []
    for symbol, name in EXCHANGE_RATES.items():
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2d")
            if len(hist) >= 2:
                prev, curr = hist["Close"].iloc[-2], hist["Close"].iloc[-1]
                change_pct = ((curr - prev) / prev) * 100
                display_rate = curr * 100 if "JPY" in symbol else curr
                results.append({
                    "name": name, "rate": display_rate,
                    "change_pct": change_pct,
                })
            elif len(hist) == 1:
                curr = hist["Close"].iloc[-1]
                display_rate = curr * 100 if "JPY" in symbol else curr
                results.append({"name": name, "rate": display_rate, "change_pct": 0})
        except Exception as e:
            print(f"Error fetching {name}: {e}")
    return results


def build_exchange_embed(rates):
    lines = []
    for r in rates:
        arrow = "🔺" if r["change_pct"] > 0 else "🔻" if r["change_pct"] < 0 else "➖"
        lines.append(
            f"{r['name']} : **{r['rate']:,.2f}원** {arrow} ({r['change_pct']:+.2f}%)"
        )
    description = "\n".join(lines) if lines else "환율 데이터를 가져올 수 없습니다."
    return {
        "title": f"💱 환율 현황 ({datetime.now(TZ).strftime('%Y-%m-%d')})",
        "description": description,
        "color": 0x9B59B6,
        "footer": {"text": "Powered by Yahoo Finance"},
    }


# 5) 공포·탐욕 지수 함수
def fetch_fear_greed_index():
    r = requests.get("https://api.alternative.me/fng/?limit=1", timeout=10)
    r.raise_for_status()
    data = r.json()["data"][0]
    return {"score": int(data["value"]), "label": data["value_classification"]}


def build_fear_greed_embed(fg):
    score = fg["score"]
    label = fg["label"]

    # 점수에 따른 색상 및 바 그래프
    if score <= 25:
        color, emoji = 0xE74C3C, "😱"
        label_kr = "극도의 공포"
    elif score <= 45:
        color, emoji = 0xE67E22, "😨"
        label_kr = "공포"
    elif score <= 55:
        color, emoji = 0xF1C40F, "😐"
        label_kr = "중립"
    elif score <= 75:
        color, emoji = 0x2ECC71, "😊"
        label_kr = "탐욕"
    else:
        color, emoji = 0x27AE60, "🤑"
        label_kr = "극도의 탐욕"

    bar_length = 20
    filled = int(score / 100 * bar_length)
    bar = "█" * filled + "░" * (bar_length - filled)

    description = (
        f"{emoji} **{score}/100 — {label_kr}** ({label})\n"
        f"```\n공포 [{bar}] 탐욕\n```\n"
        f"*0 = 극도의 공포 | 100 = 극도의 탐욕*"
    )
    return {
        "title": f"😱 공포·탐욕 지수 ({datetime.now(TZ).strftime('%Y-%m-%d')})",
        "description": description,
        "color": color,
        "footer": {"text": "Powered by Alternative.me Fear & Greed Index"},
    }


# 6) 디스코드 전송 함수
def send_to_discord(embeds):
    for embed in embeds:
        payload = {"embeds": [embed]}
        try:
            r = requests.post(DISCORD_WEBHOOK, json=payload)
            r.raise_for_status()
        except Exception as e:
            print(f"Error sending embed to Discord: {e}")
            continue


# ──────────────────────────────────────────────────────────────────
# MAIN EXECUTION
# ──────────────────────────────────────────────────────────────────

def run_daily_briefing():
    print("✅ Daily briefing script started...")
    all_embeds = []

    # 날씨
    print("🌦️ Fetching weather...")
    try:
        wdata = fetch_weather()
        wembed = build_weather_embed(wdata)
        all_embeds.append(wembed)
        print("👍 Weather info processed!")
    except Exception as e:
        print(f"❌ Error in weather processing: {e}")

    # 일반 뉴스
    print("📰 Fetching general news...")
    try:
        entries = fetch_recent_entries(NEWS_RSS_URLS)
        news_prompt = (
            "아래 뉴스 목록을 보고, 중요한 이슈 3~5개를 중요도 순으로 요약해줘. "
            "각 항목은 '[분야] 제목' 형식으로 시작하고, 핵심 내용과 원문 링크를 포함해줘.\n\n"
            "뉴스 목록:\n"
        )
        summary = summarize_news_with_gemini(entries, news_prompt)
        nembed = build_news_embed(summary)
        all_embeds.append(nembed)
        print("👍 General news processed!")
    except Exception as e:
        print(f"❌ Error in general news processing: {e}")

    # 경제·주식·투자 뉴스
    print("💰 Fetching economy & investment news...")
    try:
        economy_entries = fetch_recent_entries(ECONOMY_RSS_URLS)
        if economy_entries:
            economy_summary_prompt = (
                "아래 경제·주식·투자 뉴스 목록을 보고, 투자자에게 중요한 핵심 소식 3~5개를 요약해줘. "
                "각 항목은 '[출처] 제목'으로 시작하고, 시장에 미치는 영향과 원문 링크를 포함해줘. "
                "한국 시장과 글로벌 시장 소식을 균형있게 다뤄줘.\n\n"
                "뉴스 목록:\n"
            )
            economy_summary = summarize_news_with_gemini(economy_entries, economy_summary_prompt)
            eembed = build_economy_news_embed(economy_summary)
            all_embeds.append(eembed)

            market_prompt = (
                "아래 경제·주식·투자 뉴스 목록을 분석하여 다음 항목을 정리해줘:\n"
                "1. **시장 심리**: 현재 시장의 전반적 분위기 (낙관/중립/비관)\n"
                "2. **핵심 키워드**: 오늘의 주요 경제 키워드 3~5개\n"
                "3. **주목 섹터**: 주목할 만한 산업/섹터와 그 이유\n"
                "4. **리스크 요인**: 투자자가 주의해야 할 리스크\n"
                "5. **투자 인사이트**: 단기적 투자 관점에서의 시사점\n\n"
                "뉴스 목록:\n"
            )
            market_analysis = summarize_news_with_gemini(economy_entries, market_prompt)
            membed = build_market_analysis_embed(market_analysis)
            all_embeds.append(membed)
            print("👍 Economy & investment news processed!")
        else:
            print("ℹ️ No new economy news.")
    except Exception as e:
        print(f"❌ Error in economy news processing: {e}")

    # 주요 지수 현황
    print("📈 Fetching market indices...")
    try:
        indices = fetch_market_indices()
        if indices:
            all_embeds.append(build_indices_embed(indices))
            print("👍 Market indices processed!")
        else:
            print("ℹ️ No index data available.")
    except Exception as e:
        print(f"❌ Error in market indices processing: {e}")

    # 환율 현황
    print("💱 Fetching exchange rates...")
    try:
        rates = fetch_exchange_rates()
        if rates:
            all_embeds.append(build_exchange_embed(rates))
            print("👍 Exchange rates processed!")
        else:
            print("ℹ️ No exchange rate data available.")
    except Exception as e:
        print(f"❌ Error in exchange rates processing: {e}")

    # 공포·탐욕 지수
    print("😱 Fetching Fear & Greed Index...")
    try:
        fg = fetch_fear_greed_index()
        all_embeds.append(build_fear_greed_embed(fg))
        print("👍 Fear & Greed Index processed!")
    except Exception as e:
        print(f"❌ Error in Fear & Greed Index processing: {e}")

    # 수집된 모든 임베드 전송
    if all_embeds:
        print(f"🚀 Sending {len(all_embeds)} embeds to Discord...")
        send_to_discord(all_embeds)
        print("🎉 All tasks completed!")
    else:
        print("🤔 No content to send.")


if __name__ == "__main__":
    run_daily_briefing()
