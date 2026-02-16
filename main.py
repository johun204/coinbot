import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import os
import time
import math
from datetime import datetime, timedelta

# ==========================================
# [설정] Configuration
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

FILES = {
	"wallet": os.path.join(DATA_DIR, 'wallet.json'),
	"history": os.path.join(DATA_DIR, 'trade_history.json'),
	"analysis": os.path.join(DATA_DIR, 'analysis_result.json'),
	"learning": os.path.join(DATA_DIR, 'learning_db.json'),
	"market_data": os.path.join(DATA_DIR, 'market_data.json')
}

# [Parameters] 
SCAN_CANDIDATES = 40	
TOP_N = 5			   
MIN_ORDER_KRW = 6000	
FEE = 0.0005			
SLIPPAGE_WEIGHT = 0.5   

# [Survival & Sniper Constants]
STOP_LOSS_PCT = -4.0	   # 하드 손절 (시드 보호)
TRAILING_START_ROI = 2.0   # 2% 이상 수익 발생 시 트레일링 시작
TRAILING_GAP_PCT = 1.0	 # 고점 대비 1% 꺾이면 즉각 익절
MIN_HOLDING_MINUTES = 45   # 매수 후 최소 45분은 버팀 (휩소/노이즈 방어)
MIN_REBALANCE_GAP = 0.10   # 비중 차이가 10% 이상 나야만 물타기/불타기 (잦은 매매 방지)

MAX_CONCURRENT_REQUESTS = 8

if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)

class NpEncoder(json.JSONEncoder):
	def default(self, obj):
		if isinstance(obj, np.integer): return int(obj)
		elif isinstance(obj, np.floating): return float(obj)
		elif isinstance(obj, np.ndarray): return obj.tolist()
		return super(NpEncoder, self).default(obj)

def truncate(number, decimals=0):
	factor = 10.0 ** decimals
	return math.floor(number * factor) / factor

# ==========================================
# [Logger]
# ==========================================
class SystemLogger:
	def __init__(self): 
		self.logs = []
		self.log_file = os.path.join(DATA_DIR, 'system.log')

	def reset(self): self.logs = []
	
	def log(self, level, message):
		ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		log_msg = f"[{ts}] [{level}] {message}"
		print(log_msg)
		self.logs.append(log_msg)
		try:
			with open(self.log_file, 'a', encoding='utf-8') as f:
				f.write(log_msg + "\n")
		except: pass

logger = SystemLogger()

# ==========================================
# [Class 1] DataManager
# ==========================================
class DataManager:
	def __init__(self):
		self.file_path = FILES['market_data']
		self.cache = self.load_cache()
		self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

	def load_cache(self):
		if os.path.exists(self.file_path):
			try:
				with open(self.file_path, 'r') as f: return json.load(f)
			except: pass
		return {}

	def save_to_disk(self):
		with open(self.file_path, 'w', encoding='utf-8') as f:
			json.dump(self.cache, f, indent=4, cls=NpEncoder)

	async def _request(self, session, url, params=None, retries=3):
		async with self.semaphore:
			for i in range(retries):
				try:
					async with session.get(url, params=params, timeout=5) as response:
						if response.status == 200: return await response.json()
						elif response.status == 429: await asyncio.sleep(0.5)
						else: await asyncio.sleep(0.1)
				except: await asyncio.sleep(0.2)
			return None

	def is_valid_tick_size(self, price):
		if price >= 2000000: tick = 1000
		elif price >= 1000000: tick = 500
		elif price >= 500000: tick = 100
		elif price >= 100000: tick = 50
		elif price >= 10000: tick = 10
		elif price >= 1000: tick = 1
		elif price >= 100: tick = 0.1
		elif price >= 10: tick = 0.01
		else: tick = 0.001
		return (tick / price) * 100 <= 0.5

	async def get_market_universe(self, session):
		data = await self._request(session, "https://api.upbit.com/v1/market/all", {"isDetails": "true"})
		if not data: return [], []
		valid, risks = [], []
		for item in data:
			if not item['market'].startswith("KRW-"): continue
			ev = item.get('market_event', {})
			is_risk = ev.get('warning', False) or (isinstance(ev.get('caution'), dict) and any(ev.get('caution').values()))
			if is_risk: risks.append(item['market'])
			else: valid.append(item['market'])
		return valid, risks

	async def get_top_candidates(self, session, valid_markets, limit=SCAN_CANDIDATES):
		candidates = []
		chunk_size = 40
		tasks = [self._request(session, "https://api.upbit.com/v1/ticker", {"markets": ",".join(valid_markets[i:i+chunk_size])}) 
				 for i in range(0, len(valid_markets), chunk_size)]
		results = await asyncio.gather(*tasks)
		for res in results:
			if res: candidates.extend(res)
		if not candidates: return []
		filtered = [c for c in candidates if self.is_valid_tick_size(c['trade_price'])]
		filtered.sort(key=lambda x: x['acc_trade_price_24h'], reverse=True)
		return [x['market'] for x in filtered[:limit]]

	async def get_current_prices_batch(self, session, markets):
		"""[신규] 가상 학습 채점을 위해 여러 마켓의 현재가를 한 번에 가져옵니다."""
		if not markets: return {}
		prices = {}
		chunk_size = 40
		tasks = [self._request(session, "https://api.upbit.com/v1/ticker", {"markets": ",".join(markets[i:i+chunk_size])}) 
				 for i in range(0, len(markets), chunk_size)]
		results = await asyncio.gather(*tasks)
		for res in results:
			if res:
				for ticker in res:
					prices[ticker['market']] = ticker['trade_price']
		return prices

	async def get_orderbook(self, session, market):
		res = await self._request(session, "https://api.upbit.com/v1/orderbook", {"markets": market})
		return res[0]['orderbook_units'] if res else None

	async def get_smart_candles(self, session, market, unit='minutes/15', target_count=500):
		cached_data = self.cache.get(market, [])
		final_data = []

		if cached_data:
			new_data = await self._request(session, f"https://api.upbit.com/v1/candles/{unit}", {"market": market, "count": 200})
			if not new_data: 
				final_data = cached_data
			else:
				df_old = pd.DataFrame(cached_data)
				df_new = pd.DataFrame(new_data)
				if not df_old.empty and not df_new.empty:
					df_combined = pd.concat([df_old, df_new]).drop_duplicates(subset=['candle_date_time_kst'])
					df_combined = df_combined.sort_values('candle_date_time_kst').reset_index(drop=True)
					final_data = df_combined.iloc[-target_count:].to_dict('records')
				else:
					final_data = new_data
		else:
			to_date = None
			req_loop = math.ceil(target_count / 200)
			for _ in range(req_loop):
				params = {"market": market, "count": 200}
				if to_date: params['to'] = to_date
				data = await self._request(session, f"https://api.upbit.com/v1/candles/{unit}", params)
				if not data: break
				final_data.extend(data)
				to_date = data[-1]['candle_date_time_utc']
				await asyncio.sleep(0.1)
			final_data.sort(key=lambda x: x['candle_date_time_kst'])
			final_data = final_data[-target_count:]

		if len(final_data) < 200: return pd.DataFrame()

		self.cache[market] = final_data
		df = pd.DataFrame(final_data)
		if df.empty: return df
		df = df[['candle_date_time_kst', 'opening_price', 'high_price', 'low_price', 'trade_price', 'candle_acc_trade_volume']]
		df.columns = ['time', 'open', 'high', 'low', 'close', 'volume']
		return df

# ==========================================
# [Class 2] FeedbackLearner (완벽한 가상 진화 엔진)
# ==========================================
class FeedbackLearner:
	def __init__(self):
		self.db_path = FILES['learning']
		self.data = self.load_db()

	def load_db(self):
		if os.path.exists(self.db_path):
			try: 
				with open(self.db_path, 'r') as f: return json.load(f)
			except: pass
		return {"markets": {}}

	def save_db(self):
		with open(self.db_path, 'w', encoding='utf-8') as f: json.dump(self.data, f, indent=4, cls=NpEncoder)

	def init_market(self, market):
		if market not in self.data['markets']:
			self.data['markets'][market] = {
				"weights": {"trend": 0.25, "mom": 0.25, "vol": 0.25, "accel": 0.25},
				"predictions": [], 
				"recent_scores": [],
				"cumulative_roi": 0.0
			}

	def get_market_weights(self, market):
		self.init_market(market)
		return self.data['markets'][market]['weights']

	def update_and_get_smoothed_score(self, market, raw_score):
		self.init_market(market)
		scores = self.data['markets'][market]['recent_scores']
		scores.append(raw_score)
		if len(scores) > 3: scores.pop(0)
		return sum(scores) / len(scores)

	def record_prediction(self, market, price, conviction, components):
		self.init_market(market)
		# 메모리 폭발 방지: 예측 기록은 최대 10개만 유지
		if len(self.data['markets'][market]['predictions']) > 10:
			self.data['markets'][market]['predictions'].pop(0)
			
		self.data['markets'][market]['predictions'].append({
			"ts": time.time(),
			"price": price,
			"conviction": conviction,
			"components": components
		})
		self.save_db()

	async def evaluate_virtual_predictions(self, session, dm):
		"""[핵심 수정] 예측 기록이 있는 모든 코인의 가격을 일괄 조회하여 100% 채점합니다."""
		now = time.time()
		markets_to_fetch = []
		
		# 1. 채점해야 할(15분 지난) 코인 리스트 수집
		for mkt, m_data in self.data['markets'].items():
			preds = m_data.get('predictions', [])
			for p in preds:
				if now - p['ts'] >= 900: # 15분 경과
					markets_to_fetch.append(mkt)
					break # 한 종목당 하나라도 있으면 Fetch 대상
					
		if not markets_to_fetch: return

		# 2. 현재가 일괄 가져오기 (API 호출 1~2번으로 끝냄)
		current_prices = await dm.get_current_prices_batch(session, markets_to_fetch)
		
		learned = 0
		for mkt in markets_to_fetch:
			if mkt not in current_prices: continue
			
			m_data = self.data['markets'][mkt]
			preds = m_data.get('predictions', [])
			new_preds = []
			actual_p = current_prices[mkt]
			
			for p in preds:
				if now - p['ts'] >= 900:
					roi = ((actual_p - p['price']) / p['price'] * 100) - (FEE * 2 * 100)
					weights = m_data['weights']
					
					# 예측 적중 여부 판단 (Conviction > 0.5 이면 롱 포지션으로 간주)
					predicted_up = p['conviction'] > 0.5
					actual_up = roi > 0
					
					lr = 0.05 # 학습률
					
					if predicted_up == actual_up:
						# [보상] 예측 성공: 당시 높았던 지표 강화
						for ind, score in p['components'].items():
							weights[ind] = max(0.01, weights[ind] + (lr * (score/100.0)))
					else:
						# [처벌] 예측 실패: 당시 뻥카 친 지표 삭감
						for ind, score in p['components'].items():
							weights[ind] = max(0.01, weights[ind] - (lr * (score/100.0)))
					
					# 정규화
					tot = sum(weights.values())
					for ind in weights: weights[ind] /= tot
					
					learned += 1
				else:
					# 아직 15분 안 지난 건 남겨둠
					new_preds.append(p)
					
			m_data['predictions'] = new_preds
			
		if learned > 0:
			logger.log("EVOLVE", f"Virtual Engine: Evaluated & Updated DNA for {learned} points.")
			self.save_db()

	def update_real_trade(self, market, roi):
		self.init_market(market)
		m_data = self.data['markets'][market]
		# 너무 큰 손실이어도 영원한 마비가 오지 않도록 0.9 곱해줌
		m_data['cumulative_roi'] = (m_data.get('cumulative_roi', 0.0) * 0.9) + roi
		self.save_db()

	def get_market_multiplier(self, market):
		self.init_market(market)
		roi = self.data['markets'][market].get('cumulative_roi', 0.0)
		# 멀티플라이어 범위를 0.5 ~ 1.5로 제한하여 봇 마비(식물인간) 방지
		return 0.5 + (1.0 / (1 + np.exp(-roi * 0.3)))

# ==========================================
# [Class 3] QuantAnalyzer (현실적 스케일링)
# ==========================================
class QuantAnalyzer:
	def calculate_indicators(self, df):
		if df.empty or len(df) < 50: return df
		close = df['close']; high = df['high']; low = df['low']; volume = df['volume']

		df['MA20'] = close.rolling(20).mean()
		df['MA60'] = close.rolling(60).mean()
		df['MA200'] = close.rolling(200).mean()
		df['MACD'] = close.ewm(span=12).mean() - close.ewm(span=26).mean()
		
		delta = close.diff()
		gain = delta.where(delta > 0, 0).rolling(14).mean()
		loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
		df['RSI'] = 100 - (100 / (1 + gain/loss))
		
		tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
		df['ATR'] = tr.rolling(14).mean()
		df['Vol_MA20'] = volume.rolling(20).mean()

		def calc_slope(y):
			if len(y) < 5: return 0
			return np.polyfit(np.arange(len(y)), y, 1)[0]
		
		df['Slope'] = close.rolling(5).apply(calc_slope, raw=True)
		df['Slope_Pct'] = (df['Slope'] / close) * 100 
		df['Accel'] = df['Slope_Pct'].diff()

		is_up = close > df['open']
		df['Up_Vol_Ratio'] = volume.where(is_up, 0).rolling(5).sum() / volume.rolling(5).sum().replace(0, 1)

		return df

	def sigmoid(self, x, center=0, scale=1):
		return 1 / (1 + np.exp(-(x - center) / scale))

# ==========================================
# [Class 4] Strategy Core (조울증 치료 및 매크로 방패)
# ==========================================
class StrategyCore:
	def __init__(self, dm, learner):
		self.dm = dm
		self.analyzer = QuantAnalyzer()
		self.learner = learner

	async def get_btc_macro_filter(self, session):
		"""[핵심] 완벽한 하락장 필터"""
		df = await self.dm.get_smart_candles(session, "KRW-BTC", "minutes/60", 200)
		if df.empty: return 1.0
		df = self.analyzer.calculate_indicators(df)
		if 'MA60' not in df.columns: return 1.0

		curr = df.iloc[-1]
		
		# 데드크로스 및 이평선 하회 시 완벽한 하락장으로 판단
		is_downtrend = curr['close'] < curr['MA60'] and curr['MACD'] < 0
		if is_downtrend:
			return 0.1 # 알트코인 매수 의지를 90% 꺾어버림 (절대 방어)
		return 1.0

	async def analyze_market(self, session, market, btc_filter):
		df = await self.dm.get_smart_candles(session, market, 'minutes/15', 500)
		if df.empty or len(df) < 200: return None
		df = self.analyzer.calculate_indicators(df)
		if 'MA200' not in df.columns: return None
		curr = df.iloc[-1]

		w = self.learner.get_market_weights(market)
		
		# [조울증 치료] 각 지표의 스케일(Scale)을 대폭 넓혀 현실적으로 셋팅
		# 1. Trend: MA60과의 이격도. (10% 이상 벌어져야 만점)
		trend_val = ((curr['close'] - curr['MA60']) / curr['MA60']) * 100
		trend_score = self.analyzer.sigmoid(trend_val, center=0, scale=10) * 100

		# 2. Mom: RSI가 50 이상일 때 점수 상승 (Scale 15로 완만하게)
		mom_score = self.analyzer.sigmoid(curr['RSI'], center=50, scale=15) * 100

		# 3. Vol: 거래량이 20봉 평균 대비 3배 터질 때 만점
		vol_ratio = curr['volume'] / curr.get('Vol_MA20', 1)
		up_vol = curr.get('Up_Vol_Ratio', 0.5)
		vol_score = self.analyzer.sigmoid(vol_ratio, center=2.5, scale=1.0) * up_vol * 100

		# 4. Accel: 기울기 변화량. -1.0 ~ 1.0 사이 (Scale 0.5)
		accel_score = self.analyzer.sigmoid(curr['Accel'], center=0.0, scale=0.5) * 100

		# 가중 합산
		raw_score = (trend_score * w['trend']) + (mom_score * w['mom']) + (vol_score * w['vol']) + (accel_score * w['accel'])
		
		# Hysteresis (최근 3회 평균)
		smoothed_score = self.learner.update_and_get_smoothed_score(market, raw_score)
		
		# Macro & Micro Multiplier
		final_score = smoothed_score * btc_filter * self.learner.get_market_multiplier(market)

		# 확신도 매핑 (50점 기준, 스케일 15로 아주 부드럽고 안정적으로 비중 계산)
		conviction = self.analyzer.sigmoid(final_score, 50, 15) 
		
		opinion = "Buy" if conviction > 0.65 else ("Sell" if conviction < 0.35 else "Wait")

		comps = {"trend": trend_score, "mom": mom_score, "vol": vol_score, "accel": accel_score}
		self.learner.record_prediction(market, float(curr['close']), conviction, comps)

		if conviction > 0.5:
			logger.log("DEBUG", f"{market} | Conv:{conviction:.2f} | Score:{final_score:.1f}")

		return {
			"market": market, 
			"score": round(final_score, 2), 
			"conviction": round(conviction, 4), 
			"opinion": opinion,
			"current_price": float(curr['close']), 
			"snapshot": {
				"component_scores": comps,
				"rsi": float(curr['RSI']),
				"slope": float(curr['Slope_Pct']),
				"dna": w 
			}
		}

	async def execute_scan(self, session, held_coins):
		# 1. 100% 가상 학습 평가 실행
		await self.learner.evaluate_virtual_predictions(session, self.dm)

		# 2. 강력한 매크로 방패
		btc_filter = await self.get_btc_macro_filter(session)
		if btc_filter < 1.0: logger.log("WARN", f"BTC MACRO DOWNTREND. Filter Level: {btc_filter}")

		# 3. 마켓 스캔
		valid, risks = await self.dm.get_market_universe(session)
		candidates = await self.dm.get_top_candidates(session, valid)
		
		targets = list(set(list(held_coins.keys()) + candidates))
		results = []
		
		for i in range(0, len(targets), 5):
			batch = targets[i:i+5]
			tasks = []
			for mkt in batch:
				if mkt in risks and mkt in held_coins:
					results.append({"market": mkt, "score": 0, "conviction": 0, "opinion": "Force Sell"})
					continue
				if mkt in risks: continue
				tasks.append(self.analyze_market(session, mkt, btc_filter))
			
			for res in await asyncio.gather(*tasks):
				if res: results.append(res)
			await asyncio.sleep(0.1)

		results.sort(key=lambda x: x.get('conviction', 0), reverse=True)
		return results

# ==========================================
# [Class 5] Asset Manager (뚝심과 스나이핑)
# ==========================================
class AssetManager:
	def __init__(self, dm, learner):
		self.dm = dm
		self.learner = learner
		self.load_data()
		self.blacklists = {} 

	def load_data(self):
		if os.path.exists(FILES['wallet']):
			try: 
				with open(FILES['wallet'], 'r') as f: self.wallet = json.load(f)
			except: self.wallet = {"krw": 1000000, "coins": {}}
		else: self.wallet = {"krw": 1000000, "coins": {}}
		
		if os.path.exists(FILES['history']):
			try: 
				with open(FILES['history'], 'r') as f: self.history = json.load(f)
			except: self.history = []
		else: self.history = []

	def save_data(self, leaderboard, current_prices):
		with open(FILES['wallet'], 'w', encoding='utf-8') as f: json.dump(self.wallet, f, indent=4, cls=NpEncoder)
		
		cutoff = datetime.now() - timedelta(days=30)
		self.history = [h for h in self.history if datetime.strptime(h['time'], "%Y-%m-%d %H:%M:%S") > cutoff]
		with open(FILES['history'], 'w', encoding='utf-8') as f: json.dump(self.history, f, indent=4, cls=NpEncoder)
		
		total_equity = self.wallet['krw']
		pf_display = {"krw": self.wallet['krw'], "coins": {}}
		
		for mkt, info in self.wallet['coins'].items():
			curr_p = current_prices.get(mkt, info['avg_price'])
			net_val = info['qty'] * curr_p * (1 - FEE)
			total_equity += net_val
			pf_display['coins'][mkt] = {
				"qty": info['qty'], "avg_price": info['avg_price'],
				"current_price": curr_p, "net_value_krw": net_val, "roi": info['roi'],
				"target_weight": info.get('target_weight', 0),
				"holding_mins": (time.time() - info.get('entry_time', time.time())) / 60
			}
		
		with open(FILES['analysis'], 'w', encoding='utf-8') as f:
			json.dump({
				"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				"total_equity": total_equity,
				"leaderboard": leaderboard, 
				"portfolio": pf_display, 
				"logs": logger.logs
			}, f, indent=4, cls=NpEncoder)

	async def calc_trade_info(self, session, market, amount):
		ob = await self.dm.get_orderbook(session, market)
		if not ob: return 0, 0
		remain = amount; acquired = 0; spent = 0
		for unit in ob:
			p = unit['ask_price']; s = unit['ask_size']; cost = p*s
			if remain <= cost: 
				acquired += remain/p; spent += remain; remain = 0; break
			else: 
				acquired += s; spent += cost; remain -= cost 
		avg = spent / acquired if acquired > 0 else 0
		return avg, truncate(acquired, 8)

	async def execute_trades(self, session, leaderboard):
		total_equity = self.wallet['krw']
		current_prices = {}
		for mkt, info in self.wallet['coins'].items():
			ob = await self.dm.get_orderbook(session, mkt)
			p = ob[0]['bid_price'] if ob else info['avg_price']
			current_prices[mkt] = p
			total_equity += info['qty'] * p * (1 - FEE)
			info['roi'] = ((p - info['avg_price']) / info['avg_price']) * 100
			if 'entry_time' not in info: info['entry_time'] = time.time()

		targets = {}
		# 확신도 0.65 이상일 때만 매수 편입 (안전마진 확보)
		valid_cands = [c for c in leaderboard[:TOP_N] if c.get('conviction', 0) > 0.65] 
		total_conv = sum(c['conviction'] for c in valid_cands)
		
		for cand in leaderboard[:TOP_N]:
			if cand in valid_cands:
				weight = (cand['conviction'] / total_conv) if total_conv > 0 else 0
				targets[cand['market']] = min(weight, 0.3)
			else:
				targets[cand['market']] = 0.0

		# === 1. SELL (수익 방어 및 뚝심) ===
		for mkt in list(self.wallet['coins'].keys()):
			info = self.wallet['coins'][mkt]
			curr_val = info['qty'] * current_prices[mkt]
			curr_weight = curr_val / total_equity if total_equity > 0 else 0
			target_w = targets.get(mkt, 0.0)
			
			holding_mins = (time.time() - info['entry_time']) / 60
			
			if info['roi'] > TRAILING_START_ROI: 
				info['peak_roi'] = max(info.get('peak_roi', 0), info['roi'])
			
			should_hard_sell = False
			# 1. 하드 손절
			if info['roi'] <= STOP_LOSS_PCT: 
				should_hard_sell = True
			# 2. 트레일링 익절
			elif info.get('peak_roi', 0) >= TRAILING_START_ROI and info['roi'] <= info['peak_roi'] - TRAILING_GAP_PCT: 
				should_hard_sell = True
			# 3. 지표 완전 박살 (단, 최소 45분은 버텼을 때만)
			else:
				cand = next((c for c in leaderboard if c['market'] == mkt), None)
				if cand and cand.get('conviction', 0) < 0.35 and holding_mins > MIN_HOLDING_MINUTES:
					should_hard_sell = True

			if should_hard_sell: target_w = 0.0
			
			diff = curr_weight - target_w
			
			# [Fix] 잔파도에 안 흔들리도록 비중 차이 10% 이상 날 때만 리밸런싱
			if diff > MIN_REBALANCE_GAP or target_w == 0.0: 
				sell_amt = diff * total_equity
				qty_to_sell = sell_amt / current_prices[mkt]
				ob = await self.dm.get_orderbook(session, mkt)
				bid_p = ob[0]['bid_price'] if ob else current_prices[mkt]
				
				if bid_p > 0:
					revenue = qty_to_sell * bid_p * (1 - FEE)
					self.wallet['krw'] += revenue
					info['qty'] -= qty_to_sell
					
					self.history.append({
						"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
						"type": "SELL", "market": mkt, "price": bid_p,
						"amount": revenue, "roi": info['roi']
					})

					if info['qty'] * bid_p < 1000:
						self.learner.update_real_trade(mkt, info['roi'])
						del self.wallet['coins'][mkt]
						self.blacklists[mkt] = time.time() + (60 * 60) # 1시간 쿨타임
						logger.log("INFO", f"SELL ALL {mkt} | ROI: {info['roi']:.2f}%")

		# === 2. BUY (확실한 기회 포착) ===
		for mkt, target_w in targets.items():
			if target_w <= 0: continue
			if mkt in self.blacklists and time.time() < self.blacklists[mkt]: continue
			
			info = self.wallet['coins'].get(mkt)
			curr_val = (info['qty'] * current_prices.get(mkt, 0)) if info else 0
			curr_weight = curr_val / total_equity if total_equity > 0 else 0
			
			diff = target_w - curr_weight
			
			if diff > MIN_REBALANCE_GAP:
				buy_amt = diff * total_equity
				if buy_amt < MIN_ORDER_KRW: continue
				if self.wallet['krw'] < buy_amt: buy_amt = self.wallet['krw'] * 0.99
				
				avg_p, qty = await self.calc_trade_info(session, mkt, buy_amt)
				if qty > 0:
					cand = next((x for x in leaderboard if x['market'] == mkt), None)
					if cand and avg_p > cand['current_price'] * (1 + SLIPPAGE_WEIGHT/100): continue

					cost = qty * avg_p * (1 + FEE)
					self.wallet['krw'] -= cost
					
					if mkt not in self.wallet['coins']:
						self.wallet['coins'][mkt] = {
							'qty': 0, 'avg_price': 0, 'roi': 0,
							'entry_time': time.time() # 매수 시간 기록
						}
					
					info = self.wallet['coins'][mkt]
					old_qty = info['qty']; old_avg = info['avg_price']
					info['qty'] += qty
					info['avg_price'] = ((old_qty * old_avg) + (qty * avg_p)) / info['qty']
					info['target_weight'] = target_w
					
					self.history.append({
						"time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
						"type": "BUY", "market": mkt, "price": avg_p,
						"amount": cost, "conviction": cand['conviction'] if cand else 0
					})
					logger.log("INFO", f"BUY {mkt} | Target: {target_w*100:.1f}%")

		return current_prices

async def main():
	logger.reset()
	logger.log("INFO", "=== Alpha-Pro V37.0 ===")
	async with aiohttp.ClientSession() as session:
		dm = DataManager()
		learner = FeedbackLearner()
		core = StrategyCore(dm, learner)
		am = AssetManager(dm, learner)
		
		try:
			leaderboard = await core.execute_scan(session, am.wallet.get('coins', {}))
			current_prices = await am.execute_trades(session, leaderboard)
			am.save_data(leaderboard, current_prices)
			dm.save_to_disk() 
		except Exception as e:
			logger.log("ERROR", f"Error: {e}")

if __name__ == "__main__":
	if os.name == 'nt':
		asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
	asyncio.run(main())
