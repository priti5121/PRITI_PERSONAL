# mitigation_planner.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
import networkx as nx
import numpy as np
import json

# === Model & Embedder Load ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')
cve_impact_model = RandomForestClassifier()

class MitigationPlanner:

    def __init__(self):
        self.cve_model_trained = False
        self.cve_df = pd.DataFrame()
        self.threat_actors = []
        self.asset_config = {}

    # === Threat Actor Risk Scoring ===
    def score_threat_actor(self, actor):
        score = 0
        if actor.get("motivation") == "Espionage":
            score += 3
        score += len(actor.get("techniques", [])) * 2
        score += len(actor.get("tools", []))
        score += sum(self.score_cve(cve) for cve in actor.get("cves", [])) / 10
        return min(score, 10)

    # === CVE Risk Classification ===
    def train_cve_model(self, cve_df):
        features = cve_df[["cvss_score", "exploitability", "age_days"]]
        labels = cve_df["exploitation_status"]
        self.cve_impact_model.fit(features, labels)
        self.cve_model_trained = True
        self.cve_df = cve_df

    def score_cve(self, cve):
        if not self.cve_model_trained:
            return 5  # default mid risk
        feat = [[cve.get("cvss_score", 5), cve.get("exploitability", 0.5), cve.get("age_days", 365)]]
        return self.cve_impact_model.predict_proba(feat)[0][1] * 10

    # === TTP-to-Mitigation Mapping (NLP) ===
    def recommend_mitigation(self, ttp_description):
        mitigations = []
        keywords = ["patch", "update", "monitor", "block", "yara", "sigma"]
        for k in keywords:
            if k in ttp_description.lower():
                mitigations.append(f"Consider: {k} strategy for mitigation.")
        return mitigations or ["Refer to MITRE ATT&CK or CISA for context-specific advice."]

    # === Threat Similarity Matching ===
    def find_similar_threat_actor(self, new_actor_text, existing_actors_text):
        new_emb = embedder.encode(new_actor_text, convert_to_tensor=True)
        existing_embs = embedder.encode(existing_actors_text, convert_to_tensor=True)
        scores = util.pytorch_cos_sim(new_emb, existing_embs)[0]
        most_similar = torch.argmax(scores).item()
        return existing_actors_text[most_similar], scores[most_similar].item()

    # === Configuration Risk Evaluation ===
    def configuration_risk(self, vulnerable_software):
        risky = [conf for conf in vulnerable_software if self.asset_config.get(conf, False)]
        return len(risky), risky

# === Example Usage ===
if __name__ == "__main__":
    planner = MitigationPlanner()

    # Train CVE model
    cve_df = pd.DataFrame([
        {"cvss_score": 8.5, "exploitability": 0.9, "age_days": 100, "exploitation_status": 1},
        {"cvss_score": 5.0, "exploitability": 0.3, "age_days": 600, "exploitation_status": 0}
    ])
    planner.train_cve_model(cve_df)

    # Simulated threat actor
    threat_actor = {
        "name": "Lazarus Group",
        "motivation": "Espionage",
        "techniques": ["T1071", "T1566.001"],
        "tools": ["Custom RAT"],
        "cves": [{"cvss_score": 9.8, "exploitability": 0.95, "age_days": 50}]
    }

    risk_score = planner.score_threat_actor(threat_actor)
    print("Threat Risk Score:", risk_score)

    # Recommend mitigation
    ttp_text = "The actor uses T1071 for C2, commonly seen in ransomware. Patch and block DNS tunneling."
    mitigations = planner.recommend_mitigation(ttp_text)
    print("Mitigation Recommendations:", mitigations)

    # Threat similarity
    match, sim_score = planner.find_similar_threat_actor(
        "North Korean group using T1071 and custom backdoor",
        ["APT28 using phishing", "Lazarus using T1071 and RATs", "FancyBear using exploits"]
    )
    print("Most Similar Actor:", match, "| Similarity Score:", sim_score)

    # Configuration risk
    planner.asset_config = {"Apache 2.4.49": True, "OpenSSL 1.0.2": False}
    config_risk_count, risky_items = planner.configuration_risk(["Apache 2.4.49", "OpenSSL 1.0.2"])
    print("Configuration Risks:", config_risk_count, "Items:", risky_items)
