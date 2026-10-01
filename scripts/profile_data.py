import pandas as pd

df = pd.read_parquet("data/raw/tickets.parquet")
tag_cols = [c for c in df.columns if c.startswith("tag_")]

print("Rows:", len(df))
print("\nLanguage:\n", df["language"].value_counts(dropna=False))
print("\nType:\n", df["type"].value_counts(dropna=False))
print("\nPriority:\n", df["priority"].value_counts(dropna=False))
print("\nTop queues:\n", df["queue"].value_counts().head(10))

# Rough "network-related" filter using tags and subject
NETWORK = {"network", "connectivity", "outage", "disruption", "vpn", "bluetooth"}
tags_lower = df[tag_cols].apply(lambda col: col.fillna("").str.lower())
has_network_tag = tags_lower.isin(NETWORK).any(axis=1)
subj = df["subject"].fillna("").str.lower()
has_network_subject = subj.str.contains("network|connect|wlan|wifi|wi-fi|internet|vpn|router")
net = df[has_network_tag | has_network_subject]
print("\nNetwork-related rows:", len(net))
print(net["language"].value_counts())

# Rough proxy: answers that mostly ask for more info instead of giving steps
ans = net["answer"].fillna("").str.lower()
asks = ans.str.contains("please provide|please specify|could you|kindly provide|please share|bitte")
print("\nAnswers that look like info requests:", int(asks.sum()), "of", len(net))
print("Median answer length (chars):", int(ans.str.len().median()))