import json
import os
import requests
import feedparser
import datetime
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# --- 配置区 ---
DEEPSEEK_API_KEY = os.environ.get('DEEPSEEK_API_KEY')
NEWS_API_KEY = os.environ.get('NEWS_API_KEY')

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
NEWS_API_URL = "https://newsapi.org/v2/everything"
INDICATORS_FILE = "indicators.json"
SCORES_FILE = "scores-v3.json"

DECAY_FACTOR = 0.75
WEIGHT_FLOOR = 1

# --- 1. 网络请求基础 ---

def create_retry_session():
    session = requests.Session()
    retry_strategy = Retry(
        total=3, status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"], backoff_factor=1
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    return session

# --- 2. 数据获取模块 (返回结构化数据) ---

# A. 国际/商业新闻 (NewsAPI)
def fetch_newsapi_data(query, api_key, session):
    print(f"🌐 正在调用 NewsAPI 获取: {query}...")
    headers = {"X-Api-Key": api_key}
    params = {
        "q": query, "language": "zh", "pageSize": 10,
        "sortBy": "publishedAt", 
        "searchIn": "title,description"
    }
    
    result = {"text": "", "articles": []}
    
    try:
        response = session.get(NEWS_API_URL, headers=headers, params=params, timeout=10)
        if response.status_code != 200: return result
        data = response.json()
        if data.get('totalResults', 0) == 0: return result
        
        summary = ""
        for article in data['articles'][:5]:
            title = article['title']
            date_str = article['publishedAt'][:10]
            source = article['source']['name']
            url = article['url']
            
            summary += f"- [NewsAPI] {title} ({date_str})\n"
            result["articles"].append({
                "title": title,
                "source": f"NewsAPI / {source}",
                "date": date_str,
                "url": url
            })
            
        result["text"] = summary
        return result
    except Exception as e:
        print(f"⚠️ NewsAPI 调用部分失败: {e}")
        return result

# B. 中国官方信源 (Google News RSS)
def fetch_official_sources():
    print("🇨🇳 正在监控中国官方信源 (通过 Google RSS)...")
    
    targets = [
        { "name": "外交部/国防部", "query": "site:mfa.gov.cn OR site:mod.gov.cn" },
        { "name": "解放军报/军网", "query": "site:81.cn OR site:chinamil.com.cn" },
        { "name": "海事局", "query": "site:msa.gov.cn AND (禁航 OR 演习 OR 实弹)" }
    ]
    
    result = {"text": "", "articles": []}
    all_text = ""
    
    for target in targets:
        encoded_query = requests.utils.quote(target['query'] + " when:2d")
        rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=zh-CN&gl=CN&ceid=CN:zh-CN"
        
        try:
            feed = feedparser.parse(rss_url)
            if not feed.entries: continue
                
            all_text += f"\n【{target['name']}】:\n"
            for entry in feed.entries[:3]:
                title = entry.title
                published = entry.published if 'published' in entry else "近期"
                link = entry.link if 'link' in entry else "#"
                
                # 简单的日期格式化
                try:
                    dt = datetime.strptime(published, "%a, %d %b %Y %H:%M:%S %Z")
                    date_str = dt.strftime("%Y-%m-%d")
                except:
                    date_str = published[:16]

                all_text += f"- {title} ({date_str})\n"
                result["articles"].append({
                    "title": title,
                    "source": f"官方信源 / {target['name']}",
                    "date": date_str,
                    "url": link
                })
                
        except Exception as e:
            print(f"⚠️ RSS 获取失败 ({target['name']}): {e}")
            
    result["text"] = all_text
    return result

# --- 3. 综合情报获取 ---

def get_combined_intelligence(category, news_api_query, news_api_key, session):
    final_text = ""
    all_articles = []
    
    # 1. NewsAPI
    news_res = fetch_newsapi_data(news_api_query, news_api_key, session)
    if news_res["text"]:
        final_text += "=== 国际与商业媒体 ===\n" + news_res["text"] + "\n"
        all_articles.extend(news_res["articles"])
    
    # 2. 官方信源 (仅军事/政治)
    if category in ["军事后勤", "政治舆论"]:
        off_res = fetch_official_sources()
        if off_res["text"]:
            final_text += "=== 官方信源 ===\n" + off_res["text"] + "\n"
            all_articles.extend(off_res["articles"])
            
    if not final_text: final_text = "未获取到相关新闻。"
        
    return {"text": final_text, "articles": all_articles}

# --- 4. LLM 分析 (不变) ---

def get_triggered_indicators(category, news_text, indicators_list, api_key):
    category_indicators = [ind for ind in indicators_list if ind['category'] == category]
    if not category_indicators: return {"triggered_ids": [], "reasoning": "无指标。"}

    system_prompt = f"""
    你是一名敏锐的情报分析师。请根据提供的【混合情报源】判断是否**明确触发**了预警指标。
    
    **关键判断准则：**
    1. **官方信源权重极高：** 即使是“例行记者会”，如果发言人使用了“性质恶劣”、“严重后果”、“明确交代”、“反制”等强硬词汇，应视为触发“外交强硬声明”类指标 (如 POL-2)。
    2. **区分烈度：** - 一般抗议 -> 不触发
       - 强硬警告/严正交涉 -> 触发中低权重指标 (POL-2)
       - 战争威胁/最后通牒 -> 触发高权重指标 (POL-1)
    3. **宁可误报不可漏报：** 对于官方的异常表态，保持较高的敏感度。
    
    请返回 JSON: {{ "triggered_ids": ["ID1"], "reasoning": "简短分析..." }}
    """
    user_prompt = f"""
    **【预警指标 ({category})】**
    {json.dumps(category_indicators, indent=2, ensure_ascii=False)}
    **【混合情报源】**
    "{news_text}"
    """
    headers = { "Content-Type": "application/json", "Authorization": f"Bearer {api_key}" }
    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(DEEPSEEK_API_URL, headers=headers, data=json.dumps(payload), timeout=45)
        return json.loads(response.json()['choices'][0]['message']['content'])
    except Exception as e:
        print(f"❌ LLM 分析失败: {e}")
        return {"triggered_ids": [], "reasoning": f"分析出错: {e}"}

# --- 5. 主程序 ---

def main():
    if not DEEPSEEK_API_KEY or not NEWS_API_KEY:
        print("❌ 错误: 缺少 API 密钥。")
        exit(1)

    try:
        with open(INDICATORS_FILE, 'r', encoding='utf-8') as f:
            all_indicators_master = {ind['id']: ind for ind in json.load(f)}
    except:
        print(f"❌ 无法加载 {INDICATORS_FILE}")
        exit(1)

    try:
        with open(SCORES_FILE, 'r', encoding='utf-8') as f:
            yesterday_state = json.load(f).get('active_indicators', {})
    except:
        yesterday_state = {}

    session = create_retry_session()
    queries = {
        "经济金融": '(台湾 OR 中国) AND (经济 OR 贸易 OR 制裁 OR 供应链 OR 芯片)',
        "军事后勤": '(台湾 OR 中国) AND (军事 OR 演习 OR 解放军 OR 航母 OR 禁航)',
        "政治舆论": '(台湾 OR 中国) AND (外交 OR 政治 OR 警告 OR 撤侨)',
        "在地体感(厦门)": '厦门 AND (防空 OR 演习 OR 交通管制)'
    }

    print("--- 开始多源情报采集与分析 (V5.1) ---")
    
    results = {}
    news_sources = {} # 用于保存新闻源
    
    for category in ["经济金融", "军事后勤", "政治舆论", "在地体感(厦门)"]:
        key_map = {"经济金融": "econ", "军事后勤": "mil", "政治舆论": "pol", "在地体感(厦门)": "local"}
        key = key_map[category]
        
        # 获取情报 (包含文本和源列表)
        if category == "在地体感(厦门)":
            intel = {"text": "厦门本地居民反馈：本周防空警报测试是年度例行测试，超市物资供应充足，未见抢购，社会秩序正常。", "articles": []}
        else:
            intel = get_combined_intelligence(category, queries[category], NEWS_API_KEY, session)
            
        # 保存新闻源
        news_sources[key] = intel["articles"]
        
        # LLM 分析
        results[key] = get_triggered_indicators(category, intel["text"], list(all_indicators_master.values()), DEEPSEEK_API_KEY)

    # --- 状态计算 ---
    today_triggered_ids = set()
    for res in results.values():
        today_triggered_ids.update(res.get('triggered_ids', []))
    
    today_state = {}
    today_str = str(datetime.now(timezone.utc).date())

    for ind_id, data in yesterday_state.items():
        if ind_id not in all_indicators_master: continue
        base_weight = all_indicators_master[ind_id]['weight']
        if ind_id in today_triggered_ids:
            today_state[ind_id] = { "base_weight": base_weight, "current_weight": base_weight, "triggered_on": today_str }
        else:
            new_weight = data['current_weight'] * DECAY_FACTOR
            if new_weight >= WEIGHT_FLOOR:
                today_state[ind_id] = { "base_weight": base_weight, "current_weight": new_weight, "triggered_on": data['triggered_on'] }

    for ind_id in today_triggered_ids:
        if ind_id not in today_state and ind_id in all_indicators_master:
            base_weight = all_indicators_master[ind_id]['weight']
            today_state[ind_id] = { "base_weight": base_weight, "current_weight": base_weight, "triggered_on": today_str }

    total_possible = sum(i['weight'] for i in all_indicators_master.values())
    current_total = sum(i['current_weight'] for i in today_state.values())
    score = (current_total / total_possible) * 100 if total_possible > 0 else 0

    final_data = {
        "score": round(score),
        "total_indicators_possible": len(all_indicators_master),
        "active_indicators_count": len(today_state),
        "active_indicators": today_state,
        "category_reasoning": { k: v['reasoning'] for k, v in results.items() },
        "news_sources": news_sources, # <-- 新增字段
        "last_updated": datetime.now(timezone.utc).isoformat()
    }
    
    with open(SCORES_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_data, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 分析完成。总分: {round(score)}")

if __name__ == "__main__":
    main()

