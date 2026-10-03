# ============================================================
# TMF Directional Strategy V1
# Long-Only Backtest
# Uses same logic as TMF Directional Dashboard V2
# ============================================================

declare upper;

# ---------------- INPUTS ----------------
input fastLength = 9;
input slowLength = 21;
input rsiLength = 14;
input momentumLength = 5;
input volumeLength = 20;

input bullishRSI = 55;
input bearishRSI = 45;

input strongSignalThreshold = 4;
input tradeSize = 100;

# ---------------- INDICATORS ----------------
def fastEMAValue = ExpAverage(close, fastLength);
def slowEMAValue = ExpAverage(close, slowLength);

def vwapValue = VWAP();
def rsiValue = RSI(length = rsiLength);

def momentumValue =
    close - close[momentumLength];

def avgVolume =
    Average(volume, volumeLength);

def highVolume =
    volume > avgVolume;

def positiveBar =
    close > close[1];

def negativeBar =
    close < close[1];

# ============================================================
# COMPONENT SCORES
# ============================================================

# EMA TREND
def trendScore =
    if fastEMAValue > slowEMAValue then 1
    else if fastEMAValue < slowEMAValue then -1
    else 0;

# VWAP
def vwapScore =
    if close > vwapValue then 1
    else if close < vwapValue then -1
    else 0;

# RSI
def rsiScore =
    if rsiValue > bullishRSI then 1
    else if rsiValue < bearishRSI then -1
    else 0;

# MOMENTUM
def momentumScore =
    if momentumValue > 0 then 1
    else if momentumValue < 0 then -1
    else 0;

# DIRECTIONAL VOLUME
def volumeScore =
    if highVolume and positiveBar then 1
    else if highVolume and negativeBar then -1
    else 0;

# ============================================================
# TOTAL SCORE
# ============================================================

def totalScore =
    trendScore +
    vwapScore +
    rsiScore +
    momentumScore +
    volumeScore;

# ============================================================
# REGIMES
# ============================================================

def strongBullish =
    totalScore >= strongSignalThreshold;

def strongBearish =
    totalScore <= -strongSignalThreshold;

# Only trigger when entering regime
def bullishEntry =
    strongBullish and !strongBullish[1];

def bearishExit =
    strongBearish and !strongBearish[1];

# ============================================================
# STRATEGY ORDERS
# ============================================================

AddOrder(
    OrderType.BUY_TO_OPEN,
    bullishEntry,
    open[-1],
    tradeSize,
    Color.GREEN,
    Color.GREEN,
    "LONG +4/+5"
);

AddOrder(
    OrderType.SELL_TO_CLOSE,
    bearishExit,
    open[-1],
    tradeSize,
    Color.RED,
    Color.RED,
    "EXIT -4/-5"
);

# ============================================================
# OPTIONAL VISUALS
# ============================================================

plot FastEMA = fastEMAValue;
FastEMA.SetDefaultColor(Color.CYAN);
FastEMA.SetLineWeight(2);

plot SlowEMA = slowEMAValue;
SlowEMA.SetDefaultColor(Color.YELLOW);
SlowEMA.SetLineWeight(2);

plot VWAPLine = vwapValue;
VWAPLine.SetDefaultColor(Color.WHITE);

# ============================================================
# SCORE LABEL
# ============================================================

AddLabel(
    yes,
    "TMF STRATEGY SCORE: " + totalScore,
    if totalScore >= 4 then Color.GREEN
    else if totalScore <= -4 then Color.RED
    else Color.GRAY
);

