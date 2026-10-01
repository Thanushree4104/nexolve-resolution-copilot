import textwrap

import pandas as pd

df = pd.read_parquet("data/raw/tickets.parquet")
tag_cols = [c for c in df.columns if c.startswith("tag_")]

NETWORK = {"network", "connectivity", "outage", "disruption", "vpn", "bluetooth"}
tags_lower = df[tag_cols].apply(lambda col: col.fillna("").str.lower())
has_tag = tags_lower.isin(NETWORK).any(axis=1)
subj = df["subject"].fillna("").str.lower()
has_subject = subj.str.contains("network|connect|wlan|wifi|wi-fi|internet|vpn|router")
net = df[(has_tag | has_subject) & (df["language"] == "en")].copy()

ans = net["answer"].fillna("").str.lower()
asks = ans.str.contains("please provide|please specify|could you|kindly provide|please share|bitte")
net["info_request"] = asks

for label, group in [("NOT FLAGGED AS INFO REQUEST", net[~net.info_request]),
                     ("FLAGGED AS INFO REQUEST", net[net.info_request])]:
    print("=" * 70, "\n", label, "\n", "=" * 70)
    for _, row in group.sample(6, random_state=1).iterrows():
        print("\nSUBJECT:", row["subject"])
        print("BODY:", textwrap.shorten(str(row["body"]), 300))
        print("ANSWER:", textwrap.fill(str(row["answer"]), 100))