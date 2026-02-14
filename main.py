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

# [Hit & Run Constants]
STOP_LOSS_PCT = -2.5	   # 타이트한 손절
TRAILING_START_ROI = 1.0   # 1% 수익부터 스탑로스 본절/익절로 끌어올림
TRAILING_GAP_PCT = 0.5	 # 고점 대비 0.5% 하락시 칼익절
MIN_REBALANCE_GAP = 0.02   # 비중이 2%만 어긋나도 리밸런싱
MIN_HOLDING_MINUTES = 30   # 매도 후 해당 종목 30분 매수 금지 (블랙리스트)

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
# [Class 1] DataManager (Tick Size Filter 추가)
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
		"""호가 단위(Tick Size)가 너무 큰 동전주 제외 (1틱 = 0.5% 초과 시 스캘핑 불가)"""
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
		
		# [Filter] Tick Size가 너무 큰 코인 필터링
		filtered = [c for c in candidates if self.is_valid_tick_size(c['trade_price'])]
		filtered.sort(key=lambda x: x['acc_trade_price_24h'], reverse=True)
		return [x['market'] for x in filtered[:limit]]

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
# [Class 2] FeedbackLearner (무한 가상 학습 엔진)
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
		"""[핵심] 각 코인별로 100% 독립된 DNA와 예측 기록 저장소를 생성합니다."""
		if market not in self.data['markets']:
			self.data['markets'][market] = {
				"weights": {"trend": 0.25, "mom": 0.25, "vol": 0.25, "accel": 0.25},
				"predictions": [], # 가상 학습을 위한 섀도우 복싱 기록
				"recent_scores": [] # Hysteresis (노이즈 제거용 최근 3회 점수)
			}

	def get_market_weights(self, market):
		self.init_market(market)
		return self.data['markets'][market]['weights']

	def update_and_get_smoothed_score(self, market, raw_score):
		"""[Hysteresis] 순간적인 휩소(노이즈)를 방지하기 위해 최근 3회 점수의 평균을 사용합니다."""
		self.init_market(market)
		scores = self.data['markets'][market]['recent_scores']
		scores.append(raw_score)
		if len(scores) > 3: scores.pop(0)
		return sum(scores) / len(scores)

	def record_prediction(self, market, price, conviction, components):
		"""현재 타점의 예측 데이터를 저장하여 미래에 채점받을 준비를 합니다."""
		self.init_market(market)
		self.data['markets'][market]['predictions'].append({
			"ts": time.time(),
			"price": price,
			"conviction": conviction,
			"components": components
		})
		self.save_db()

	def evaluate_virtual_predictions(self, current_prices):
		"""[자기 진화 핵심] 실제 돈을 쓰지 않고도, 과거 15분 전의 예측을 채점하여 뇌(가중치)를 재조립합니다."""
		now = time.time()
		learned = 0
		
		for mkt, m_data in self.data['markets'].items():
			preds = m_data.get('predictions', [])
			new_preds = []
			
			for p in preds:
				# 15분(900초)이 경과한 예측 건에 대해 채점
				if now - p['ts'] >= 900:
					if mkt in current_prices:
						actual_p = current_prices[mkt]
						# 수수료 왕복(0.1%)을 뺀 순수익률
						roi = ((actual_p - p['price']) / p['price'] * 100) - (FEE * 2 * 100)
						
						weights = m_data['weights']
						lr = 0.05 # 학습률 (5%)
						
						for ind, score in p['components'].items():
							norm_score = score / 100.0
							# [음성 강화] 오를 줄 알았는데 떨어졌으면 가중치 삭감, 오르면 보상
							adjustment = lr * roi * norm_score
							weights[ind] = max(0.01, weights[ind] + adjustment)
						
						# 가중치 정규화 (합이 1이 되도록)
						tot = sum(weights.values())
						for ind in weights: weights[ind] /= tot
						
						learned += 1
				else:
					new_preds.append(p)
			
			m_data['predictions'] = new_preds
			
		if learned > 0:
			logger.log("EVOLVE", f"Virtual Learning: {learned} predictions evaluated and DNA updated.")
			self.save_db()

# ==========================================
# [Class 3] QuantAnalyzer (10가지 이상의 지표 복합 연산)
# ==========================================
class QuantAnalyzer:
	def calculate_indicators(self, df):
		if df.empty or len(df) < 200: return df
		close = df['close']; high = df['high']; low = df['low']; volume = df['volume']

		# 1~3. 추세 (MA20, 60, 200)
		df['MA20'] = close.rolling(20).mean()
		df['MA60'] = close.rolling(60).mean()
		df['MA200'] = close.rolling(200).mean()
		
		# 4~5. MACD & Signal
		ema12 = close.ewm(span=12).mean()
		ema26 = close.ewm(span=26).mean()
		df['MACD'] = ema12 - ema26
		
		# 6. RSI
		delta = close.diff()
		gain = delta.where(delta > 0, 0).rolling(14).mean()
		loss = (-delta.where(delta < 0, 0)).rolling(14).mean().replace(0, 0.001)
		df['RSI'] = 100 - (100 / (1 + gain/loss))
		
		# 7. Stochastic (K)
		low14 = low.rolling(14).min()
		high14 = high.rolling(14).max()
		df['Stoch_K'] = 100 * ((close - low14) / (high14 - low14).replace(0, 0.001))
		
		# 8. ATR (변동성)
		tr = pd.concat([high-low, (high-close.shift(1)).abs(), (low-close.shift(1)).abs()], axis=1).max(axis=1)
		df['ATR'] = tr.rolling(14).mean()
		
		# 9~10. Bollinger Bands Width (스퀴즈 판별용)
		std = close.rolling(20).std()
		df['BB_Up'] = df['MA20'] + 2*std
		df['BB_Low'] = df['MA20'] - 2*std
		df['BB_Width'] = (df['BB_Up'] - df['BB_Low']) / df['MA20']

		# 11. OBV (매집량)
		df['OBV'] = (np.sign(delta) * volume).fillna(0).cumsum()
		
		# 12. Volume SMA
		df['Vol_MA20'] = volume.rolling(20).mean()

		# 13~14. Slope & Accel (선행성)
		def calc_slope(y):
			if len(y) < 5: return 0
			return np.polyfit(np.arange(len(y)), y, 1)[0]
		
		df['Slope'] = close.rolling(5).apply(calc_slope, raw=True)
		df['Slope_Pct'] = (df['Slope'] / close) * 100 
		df['Accel'] = df['Slope_Pct'].diff()

		# 15. Up_Volume Ratio (상승 거래량 비율)
		is_up = close > df['open']
		df['Up_Vol_Ratio'] = volume.where(is_up, 0).rolling(5).sum() / volume.rolling(5).sum().replace(0, 1)

		return df

	def sigmoid(self, x, center=0, scale=1):
		"""Hard Threshold를 파괴하는 부드러운 확률 전환 함수"""
		return 1 / (1 + np.exp(-(x - center) / scale))

# ==========================================
# [Class 4] Strategy Core (Continuous Multi-Factor)
# ==========================================
class StrategyCore:
	def __init__(self, dm, learner):
		self.dm = dm
		self.analyzer = QuantAnalyzer()
		self.learner = learner

	async def get_btc_macro_filter(self, session):
		df = await self.dm.get_smart_candles(session, "KRW-BTC", "minutes/60", 200)
		if df.empty: return 1.0
		df = self.analyzer.calculate_indicators(df)
		if 'MA60' not in df.columns: return 1.0

		curr = df.iloc[-1]
		# BTC 대세 하락장 판별 (이평선 역배열 + 하락 기울기)
		if curr['close'] < curr['MA60'] and curr['Slope_Pct'] < -0.1:
			return 0.3 # 모든 알트코인 매수 신호를 70% 삭감 (Veto)
		return 1.0

	async def analyze_market(self, session, market, btc_filter):
		df = await self.dm.get_smart_candles(session, market, 'minutes/15', 500)
		if df.empty: return None
		df = self.analyzer.calculate_indicators(df)
		if 'MA200' not in df.columns: return None
		curr = df.iloc[-1]

		# 종목별 독립 DNA 
		w = self.learner.get_market_weights(market)
		
		# --- [Continuous Component Scoring 0~100] ---
		# 1. Trend: 정배열 및 MACD 강도
		trend_val = ((curr['close'] - curr['MA60']) / curr['MA60'] * 100) + curr['MACD']
		trend_score = self.analyzer.sigmoid(trend_val, center=0, scale=2) * 100

		# 2. Mom: RSI와 스토캐스틱 최적 타점 (너무 높으면 과매수 페널티)
		mom_val = (curr['RSI'] - 50) + (curr['Stoch_K'] - 50) * 0.5
		mom_score = self.analyzer.sigmoid(mom_val, center=0, scale=20) * 100

		# 3. Volatility: 상승 거래량 분출 & Squeeze Breakout
		vol_ratio = curr['volume'] / curr.get('Vol_MA20', 1)
		up_vol = curr.get('Up_Vol_Ratio', 0.5)
		vol_score = self.analyzer.sigmoid(vol_ratio, center=2.0, scale=0.5) * up_vol * 100
		
		# [Squeeze Logic] 밴드가 좁은데(수축) 거래량이 안터지면 가점 박탈
		if curr['BB_Width'] < 0.05 and vol_ratio < 2.0:
			vol_score *= 0.1 

		# 4. Prediction: 가속도 기반 선제 진입
		accel_val = curr['Slope_Pct'] + curr['Accel']
		accel_score = self.analyzer.sigmoid(accel_val, center=0.1, scale=0.2) * 100

		# 가중 합산
		raw_score = (trend_score * w['trend']) + (mom_score * w['mom']) + (vol_score * w['vol']) + (accel_score * w['accel'])
		
		# 노이즈 제거 (Hysteresis) 및 BTC 거시 필터 적용
		smoothed_score = self.learner.update_and_get_smoothed_score(market, raw_score)
		final_score = smoothed_score * btc_filter

		# [진입 장벽 완화] 중심점 60 (잦고 공격적인 매매 유도)
		conviction = self.analyzer.sigmoid(final_score, 60, 10) 
		opinion = "Buy" if conviction > 0.6 else ("Sell" if conviction < 0.4 else "Wait")

		comps = {"trend": trend_score, "mom": mom_score, "vol": vol_score, "accel": accel_score}
		
		# 이번 타점을 기록하여 15분 뒤에 채점받도록 함 (Virtual Learning)
		self.learner.record_prediction(market, float(curr['close']), conviction, comps)

		if conviction > 0.55:
			logger.log("DEBUG", f"{market} | Conv:{conviction:.2f} | Score:{final_score:.1f} (T{trend_score:.0f}/M{mom_score:.0f}/V{vol_score:.0f}/A{accel_score:.0f})")

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
				"bb_width": float(curr['BB_Width']),
				"dna": w 
			}
		}

	async def execute_scan(self, session, held_coins):
		# 1. 가상 학습 평가 (과거 예측 채점)
		valid, risks = await self.dm.get_market_universe(session)
		candidates = await self.dm.get_top_candidates(session, valid)
		
		current_candidate_prices = {}
		for c in candidates[:TOP_N*2]: 
			ob = await self.dm.get_orderbook(session, c)
			if ob: current_candidate_prices[c] = ob[0]['bid_price']
		self.learner.evaluate_virtual_predictions(current_candidate_prices)

		# 2. 거시 필터 확인
		btc_filter = await self.get_btc_macro_filter(session)
		if btc_filter < 1.0: logger.log("WARN", "BTC Macro Downtrend Detected. Restricting Altcoin Buys.")

		# 3. 개별 종목 스캔
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
# [Class 5] Asset Manager (Hit & Run)
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
		
		# 로그 최적화 (1달 치만 보관)
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
				"target_weight": info.get('target_weight', 0)
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

		targets = {}
		# 확신도 0.55 이상이면 비중 편입 시작 (공격적)
		valid_cands = [c for c in leaderboard[:TOP_N] if c.get('conviction', 0) > 0.55] 
		total_conv = sum(c['conviction'] for c in valid_cands)
		
		for cand in leaderboard[:TOP_N]:
			if cand in valid_cands:
				weight = (cand['conviction'] / total_conv) if total_conv > 0 else 0
				targets[cand['market']] = min(weight, 0.3) # 최대 30% 몰빵 가능
			else:
				targets[cand['market']] = 0.0

		# === 1. SELL (방어 & 익절) ===
		for mkt in list(self.wallet['coins'].keys()):
			info = self.wallet['coins'][mkt]
			curr_val = info['qty'] * current_prices[mkt]
			curr_weight = curr_val / total_equity if total_equity > 0 else 0
			
			target_w = targets.get(mkt, 0.0)
			
			# [Hit & Run] 다이내믹 트레일링 스탑
			if info['roi'] > TRAILING_START_ROI: 
				info['peak_roi'] = max(info.get('peak_roi', 0), info['roi'])
			
			should_hard_sell = False
			if info['roi'] < STOP_LOSS_PCT: 
				should_hard_sell = True # 하드 손절
			elif info.get('peak_roi', 0) > TRAILING_START_ROI and info['roi'] < info['peak_roi'] - TRAILING_GAP_PCT: 
				should_hard_sell = True # 익절 (고점 대비 하락)
			else:
				# [Hysteresis] AI 점수가 낮아져서 팔아야 할 때
				cand = next((c for c in leaderboard if c['market'] == mkt), None)
				if cand and cand.get('conviction', 0) < 0.4:
					should_hard_sell = True

			if should_hard_sell: target_w = 0.0
			
			diff = curr_weight - target_w
			
			if diff > MIN_REBALANCE_GAP: 
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
						del self.wallet['coins'][mkt]
						# 매도 후 30분 매수 금지 (블랙리스트)
						self.blacklists[mkt] = time.time() + (MIN_HOLDING_MINUTES * 60)
						logger.log("INFO", f"SELL ALL {mkt} | ROI: {info['roi']:.2f}% | Hit & Run")

		# === 2. BUY (적극적 편입) ===
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
						cand = next((c for c in leaderboard if c['market'] == mkt), None)
						self.wallet['coins'][mkt] = {
							'qty': 0, 'avg_price': 0, 'roi': 0
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
	logger.log("INFO", "=== Alpha-Pro V36.0 Virtual AI Evolution ===")
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
