import json
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from neo_model import ThreatActor  # Your neomodel structure

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')

def generate_embedding(actor):
    combined_text = (
        f"Name: {actor.name or ''}\n"
        f"Description: {actor.description or ''}\n"
        f"Aliases: {', '.join(actor.aliases) if actor.aliases else ''}\n"
        f"Sophistication: {actor.sophistication or ''}\n"
        f"Resource Level: {actor.resource_level or ''}\n"
        f"Motivation: {actor.primary_motivation or ''}\n"
        f"Country: {actor.country or ''}\n"
        f"Victim Countries: {', '.join(actor.victim_countries) if actor.victim_countries else ''}\n"
        f"First Seen: {actor.first_seen_source or ''}\n"
        f"Created At: {actor.created_at or ''}\n"
        f"Last Seen: {actor.last_seen or ''}"
    )
    return combined_text, embedding_model.encode(combined_text)

def plot_similarity(name1, name2, similarity):
    plt.figure(figsize=(8, 4))
    plt.plot([0, 1], [similarity, similarity], marker='o', linestyle='-', color='blue')
    plt.title("Threat Actor Cosine Similarity")
    plt.xticks([0, 1], [name1, name2])
    plt.ylim(0, 1)
    plt.ylabel("Cosine Similarity")
    plt.grid(True)

    # Automatically annotate score
    plt.text(0.5, similarity + 0.02, f"{similarity:.4f}", ha='center', fontsize=12, color='darkgreen')

    plt.tight_layout()
    plt.savefig("2_threat_actor_pair_similarity.png")
    plt.show()

def main():
    uuid = "c4e22d1135a74e3a8b9541a689a390a1"

    # Fetch ThreatActor nodes with same UUID
    actors = ThreatActor.nodes.filter(uuid=uuid)

    if len(actors) < 2:
        print(f"❌ Only {len(actors)} actor(s) found with UUID '{uuid}'. Need at least 2.")
        return

    print(f"✅ Found {len(actors)} ThreatActor entries with same UUID.\n")

    vectors, names, summaries = [], [], []

    # Generate vectors and log details
    for actor in actors:
        summary, vector = generate_embedding(actor)
        vectors.append(vector)
        summaries.append(summary)
        names.append(actor.name or actor.uid)
        print(f"📌 Actor: {actor.name or actor.uid}")
        print(summary)
        print('-' * 60)

    # Compute cosine similarity
    similarity = util.cos_sim(vectors[0], vectors[1]).item()
    print(f"\n✅ Cosine Similarity between '{names[0]}' and '{names[1]}': {round(similarity, 4)}")

    # Save vectors
    vector_data = [
        {
            "uid": actor.uid,
            "name": actor.name,
            "uuid": actor.uuid,
            "vector": vec.tolist(),
            "description": actor.description,
        }
        for actor, vec in zip(actors, vectors)
    ]

    with open("2_threat_actors_vectors.json", "w") as f:
        json.dump(vector_data, f, indent=4)

    with open("2_threat_actor_pair_similarity.json", "w") as f:
        json.dump({
            "actor_1": names[0],
            "actor_2": names[1],
            "similarity_score": round(similarity, 4)
        }, f, indent=4)

    # Visualize
    plot_similarity(names[0], names[1], similarity)

if __name__ == "__main__":
    main()
