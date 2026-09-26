import streamlit as st
import json
import pandas as pd
from azure.storage.blob import BlobServiceClient
import os
import plotly.graph_objects as go

st.set_page_config(page_title="Crypto Sentiment Signals", layout="wide")

# Keys come from environment variables, NOT hardcoded
STORAGE_ACCOUNT = os.environ.get("STORAGE_ACCOUNT_NAME", "cryptosentimentdl")
STORAGE_KEY = os.environ.get("STORAGE_ACCOUNT_KEY")

connection_string = f"DefaultEndpointsProtocol=https;AccountName={STORAGE_ACCOUNT};AccountKey={STORAGE_KEY};EndpointSuffix=core.windows.net"
blob_service_client = BlobServiceClient.from_connection_string(connection_string)
gold_container = blob_service_client.get_container_client("gold")

@st.cache_data(ttl=300)  # refresh every 5 min
def load_signals():
    records = []
    for blob in gold_container.list_blobs(name_starts_with="signals/"):
        blob_client = gold_container.get_blob_client(blob.name)
        data = json.loads(blob_client.download_blob().readall())
        # blob contains a JSON array — extend, not append
        if isinstance(data, list):
            records.extend(data)
        else:
            records.append(data)
    if not records:
        return pd.DataFrame(columns=["timestamp", "symbol", "signal", "p_score",
                                     "news_score", "ta_score", "current_price",
                                     "confidence_label", "rationale"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")
    return df

st.title("🪙 Crypto Sentiment Trading Signals")

df = load_signals()

if df.empty:
    st.warning("No signals found yet.")
else:
    latest = df.iloc[-1]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Signal", latest["signal"])
    col2.metric("P-Score", f"{latest['p_score']:.4f}")
    col3.metric("BTC Price", f"${latest['current_price']:,.2f}")
    col4.metric("Confidence", latest["confidence_label"])

    st.subheader("Sentiment & Signal Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["p_score"], name="P-Score", line=dict(color="cyan")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["news_score"], name="News Score", line=dict(color="orange")))
    fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ta_score"], name="TA Score", line=dict(color="green")))
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Rationale")
    st.info(latest["rationale"])

    st.subheader("Signal History")
    st.dataframe(df[["timestamp", "symbol", "signal", "p_score", "news_score", "ta_score", "current_price", "confidence_label"]], use_container_width=True)