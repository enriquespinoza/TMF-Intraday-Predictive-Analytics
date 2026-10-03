# ============================================================
# TMF Directional Signal Dashboard V2
# Score Range: -5 to +5
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

input showSignals = yes;
input showLabels = yes;
input showVWAP = yes;

# ---------------- MOVING AVERAGES ----------------
def fastEMAValue = ExpAverage(close, fastLength);
def slowEMAValue = ExpAverage(close, slowLength);

plot FastEMA = fastEMAValue;
FastEMA.SetDefaultColor(Color.CYAN);
FastEMA.SetLineWeight(2);

plot SlowEMA = slowEMAValue;
SlowEMA.SetDefaultColor(Color.YELLOW);
SlowEMA.SetLineWeight(2);

# ---------------- VWAP ----------------
def vwapValue = VWAP();

plot VWAPLine =
    if showVWAP then vwapValue
    else Double.NaN;

VWAPLine.SetDefaultColor(Color.WHITE);
VWAPLine.SetLineWeight(2);

# ---------------- RSI ----------------
def rsiValue = RSI(length = rsiLength);

# ---------------- MOMENTUM ----------------
def momentumValue = close - close[momentumLength];

# ---------------- VOLUME ----------------
def avgVolume = Average(volume, volumeLength);
def highVolume = volume > avgVolume;

# Direction of current price bar
def positiveBar = close > close[1];
def negativeBar = close < close[1];

# ============================================================
# COMPONENT SCORES
# ============================================================

# EMA TREND: +1 / -1
def trendScore =
    if fastEMAValue > slowEMAValue then 1
    else if fastEMAValue < slowEMAValue then -1
    else 0;

# VWAP: +1 / -1
def vwapScore =
    if close > vwapValue then 1
    else if close < vwapValue then -1
    else 0;

# RSI: +1 / 0 / -1
def rsiScore =
    if rsiValue > bullishRSI then 1
    else if rsiValue < bearishRSI then -1
    else 0;

# MOMENTUM: +1 / -1
def momentumScore =
    if momentumValue > 0 then 1
    else if momentumValue < 0 then -1
    else 0;

# DIRECTIONAL VOLUME
# High volume up bar   = +1
# High volume down bar = -1
# Normal volume        = 0
def volumeScore =
    if highVolume and positiveBar then 1
    else if highVolume and negativeBar then -1
    else 0;

# ============================================================
# COMPOSITE SCORE
# ============================================================

def totalScore =
    trendScore +
    vwapScore +
    rsiScore +
    momentumScore +
    volumeScore;

# ---------------- REGIMES ----------------
def strongBullish = totalScore >= strongSignalThreshold;
def bullish = totalScore >= 2 and totalScore < strongSignalThreshold;

def strongBearish = totalScore <= -strongSignalThreshold;
def bearish = totalScore <= -2 and totalScore > -strongSignalThreshold;

def neutral = totalScore > -2 and totalScore < 2;

# ============================================================
# SIGNALS
# Only trigger when ENTERING a strong regime
# ============================================================

def newStrongBullish =
    strongBullish and !strongBullish[1];

def newStrongBearish =
    strongBearish and !strongBearish[1];

plot BuySignal =
    if showSignals and newStrongBullish
    then low
    else Double.NaN;

BuySignal.SetPaintingStrategy(PaintingStrategy.ARROW_UP);
BuySignal.SetDefaultColor(Color.GREEN);
BuySignal.SetLineWeight(4);

plot SellSignal =
    if showSignals and newStrongBearish
    then high
    else Double.NaN;

SellSignal.SetPaintingStrategy(PaintingStrategy.ARROW_DOWN);
SellSignal.SetDefaultColor(Color.RED);
SellSignal.SetLineWeight(4);

# ============================================================
# DASHBOARD
# ============================================================

AddLabel(
    showLabels,
    "TMF SCORE: " + totalScore,
    if strongBullish then Color.GREEN
    else if bullish then Color.LIGHT_GREEN
    else if strongBearish then Color.RED
    else if bearish then Color.LIGHT_RED
    else Color.GRAY
);

AddLabel(
    showLabels,
    if strongBullish then "STRONG BULLISH"
    else if bullish then "BULLISH"
    else if strongBearish then "STRONG BEARISH"
    else if bearish then "BEARISH"
    else "NEUTRAL",
    if strongBullish then Color.GREEN
    else if bullish then Color.LIGHT_GREEN
    else if strongBearish then Color.RED
    else if bearish then Color.LIGHT_RED
    else Color.GRAY
);

# ---------------- COMPONENT LABELS ----------------

AddLabel(
    showLabels,
    "TREND: " +
    if trendScore == 1 then "+1"
    else if trendScore == -1 then "-1"
    else "0",
    if trendScore == 1 then Color.GREEN
    else if trendScore == -1 then Color.RED
    else Color.GRAY
);

AddLabel(
    showLabels,
    "VWAP: " +
    if vwapScore == 1 then "+1"
    else if vwapScore == -1 then "-1"
    else "0",
    if vwapScore == 1 then Color.GREEN
    else if vwapScore == -1 then Color.RED
    else Color.GRAY
);

AddLabel(
    showLabels,
    "RSI: " + Round(rsiValue, 1) + " (" +
    if rsiScore == 1 then "+1)"
    else if rsiScore == -1 then "-1)"
    else "0)",
    if rsiScore == 1 then Color.GREEN
    else if rsiScore == -1 then Color.RED
    else Color.GRAY
);

AddLabel(
    showLabels,
    "MOM: " +
    if momentumScore == 1 then "+1"
    else if momentumScore == -1 then "-1"
    else "0",
    if momentumScore == 1 then Color.GREEN
    else if momentumScore == -1 then Color.RED
    else Color.GRAY
);

AddLabel(
    showLabels,
    "VOL: " +
    if volumeScore == 1 then "+1"
    else if volumeScore == -1 then "-1"
    else "0",
    if volumeScore == 1 then Color.GREEN
    else if volumeScore == -1 then Color.RED
    else Color.GRAY
);
