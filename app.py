# Jupyter "magic" that writes everything in this cell to a file named app.py.
# This lets me keep working in a notebook, but still produce a runnable Python script.
import argparse  # For parsing command-line arguments (e.g., --k 10 --topn 5 --out ...)
import json      # For writing the final recommendations out to a JSON file
import numpy as np  # NumPy for fast numerical operations (arrays, argpartition, etc.)
import pandas as pd  # Pandas for reading CSVs and reshaping data into a matrix
from sklearn.metrics.pairwise import cosine_similarity  # Computes cosine similarity between user vectors


def _top_k_similar_users(user_idx: int, user_similarity: np.ndarray, k: int) -> np.ndarray:
    """
    Find the row indices of the k most similar users to the user at index `user_idx`.

    Parameters
    ----------
    user_idx : int
        The row index of the target user inside the interaction matrix.
        (Not the user_id itself; this is the positional index.)
    user_similarity : np.ndarray
        A precomputed user-user similarity matrix of shape (n_users, n_users),
        where user_similarity[i, j] is the cosine similarity between users i and j.
    k : int
        The number of neighbors (similar users) we want to retrieve.

    Returns
    -------
    np.ndarray
        An array of length k containing the indices of the most similar users,
        sorted from most similar to least similar.

    Notes
    -----
    - We explicitly exclude the target user themself so they don't appear as their own nearest neighbor.
    - We use np.argpartition for efficiency (faster than sorting the entire array),
      then we sort only the top-k candidates to get them in correct order.
    """
    # Copy the similarity row so we can safely modify it without changing the original matrix.
    # sim_vec[j] represents similarity between user_idx and user j.
    sim_vec = user_similarity[user_idx].copy()

    # A user is always perfectly similar to themself (similarity = 1),
    # but we don't want them included in their own neighbor list.
    # Setting it to -inf guarantees it will never appear in the top-k list.
    sim_vec[user_idx] = -np.inf  # exclude self

    # Safety: you can't pick more neighbors than the total number of other users available.
    # e.g., if there are 50 users, the max neighbors for one user is 49.
    k = min(k, sim_vec.shape[0] - 1)

    # If k becomes 0 or negative (e.g., dataset has only 1 user), return an empty array.
    if k <= 0:
        return np.array([], dtype=int)

    # np.argpartition does a "partial selection":
    # - It finds the indices of the top-k largest elements without fully sorting the whole array.
    # - This is much faster than np.argsort for large arrays.
    # We negate sim_vec because argpartition selects the *smallest* by default.
    candidate_idx = np.argpartition(-sim_vec, kth=k - 1)[:k]

    # argpartition doesn't guarantee the top-k are sorted, so we sort just those candidates.
    # np.argsort gives ascending; [::-1] flips to descending (most similar first).
    candidate_idx = candidate_idx[np.argsort(sim_vec[candidate_idx])[::-1]]

    return candidate_idx


def _recommend_top_n_for_user_with_scores(
    user_id: int,
    interaction: np.ndarray,
    users: np.ndarray,
    movies: np.ndarray,
    user_similarity: np.ndarray,
    k: int,
    top_n: int,
) -> list[dict]:
    """
    Recommend top_n movies for a given user using user-user collaborative filtering.

    Pipeline
    --------
    1) Find the k most similar users (neighbors).
    2) For each movie, compute the average rating among those k neighbors
       (only among neighbors who actually rated the movie).
    3) Filter out movies the target user already rated.
    4) Return the top_n movies with the highest neighbor-average rating.

    Output format
    -------------
    [
      {"title": "<movie title>", "score": <avg_neighbor_rating>},
      ...
    ]

    Assumptions
    -----------
    - The interaction matrix is dense and uses 0.0 to represent "unrated" entries.
    - Real ratings are positive (e.g., 1–5), so rating > 0 indicates the user rated the movie.
    """
    # Map real user IDs (like 1, 2, 42, ...) to the corresponding row index in `interaction`.
    # We cast to int to avoid issues where user IDs are numpy types.
    user_to_idx = {int(uid): i for i, uid in enumerate(users)}

    # If the requested user_id isn't in our matrix, we can't recommend anything.
    if int(user_id) not in user_to_idx:
        return []

    # Convert the target user_id into its row index in the interaction matrix.
    ui = user_to_idx[int(user_id)]

    # --------------------------
    # 1) Find k nearest neighbors
    # --------------------------
    neighbor_idx = _top_k_similar_users(ui, user_similarity, k=k)

    # If there are no neighbors (tiny dataset edge case), return no recommendations.
    if neighbor_idx.size == 0:
        return []

    # ------------------------------------------
    # 2) Compute movie average rating among neighbors
    # ------------------------------------------
    # Slice the matrix to keep only the neighbor rows.
    # Shape: (k, n_movies)
    neighbor_ratings = interaction[neighbor_idx, :]

    # Mask for where a neighbor actually rated a movie.
    # True means neighbor_ratings[row, col] is a real rating, not 0.0.
    rated_mask = neighbor_ratings > 0

    # Sum ratings per movie across the k neighbors.
    # Shape: (n_movies,)
    rating_sums = neighbor_ratings.sum(axis=0)

    # Count how many neighbors rated each movie.
    # Shape: (n_movies,)
    rating_counts = rated_mask.sum(axis=0)

    # Compute average rating per movie, but ONLY where rating_counts > 0 (to avoid divide-by-zero).
    # Movies with zero neighbor ratings get an average score of 0.0 ("no evidence").
    avg_ratings = np.where(rating_counts > 0, rating_sums / rating_counts, 0.0)

    # ------------------------------------------
    # 3) Exclude movies already rated by the user
    # ------------------------------------------
    # already_rated[col] is True if the target user has a rating > 0 for that movie.
    already_rated = interaction[ui, :] > 0

    # Candidate movies must meet two conditions:
    # - At least one neighbor rated it (rating_counts > 0)
    # - The target user has NOT rated it (~already_rated)
    # If a movie fails either condition, we force its candidate score to 0.0.
    candidate_scores = np.where((rating_counts > 0) & (~already_rated), avg_ratings, 0.0)

    # If top_n is invalid or none of the candidates have non-zero scores, return empty.
    if top_n <= 0 or np.all(candidate_scores == 0):
        return []

    # top_n can't exceed the number of available movies.
    top_n = min(top_n, candidate_scores.shape[0])

    # --------------------------
    # 4) Pick top_n movies
    # --------------------------
    # Use argpartition again to efficiently get top_n indices without sorting everything.
    top_idx = np.argpartition(-candidate_scores, kth=top_n - 1)[:top_n]

    # Sort those indices by score descending so output is in correct order.
    top_idx = top_idx[np.argsort(candidate_scores[top_idx])[::-1]]

    # Build the final list of recommendations with a rounded score for readability in JSON.
    recs: list[dict] = []
    for i in top_idx:
        score = float(candidate_scores[i])  # ensure JSON-serializable float

        # If score is 0 or negative (shouldn't happen with positive ratings), skip it.
        if score <= 0:
            continue

        recs.append(
            {
                "title": str(movies[i]),        # movie title from the column labels
                "score": round(score, 4),       # rounded average neighbor rating
            }
        )

    return recs


def main() -> None:
    """
    Entry point for the script.

    Responsibilities:
    - Read input CSV files (ratings + movie titles)
    - Create the interaction matrix
    - Compute user-user cosine similarity
    - Generate top-N recommendations for each user
    - Sort users alphabetically by username
    - Write results to a JSON file
    """
    # Set up CLI arguments so the script is configurable without editing code.
    parser = argparse.ArgumentParser(
        description="Export top-N user recommendations to JSON (user-user CF)."
    )
    parser.add_argument(
        "--ratings",
        default="./Movie_data.csv",
        help="Path to Movie_data.csv"
    )
    parser.add_argument(
        "--titles",
        default="./Movie_Id_Titles.csv",
        help="Path to Movie_Id_Titles.csv"
    )
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of similar users to use (k-neighbors)"
    )
    parser.add_argument(
        "--topn",
        type=int,
        default=5,
        help="Top N recommended movies per user"
    )
    parser.add_argument(
        "--out",
        default="user_top5_recs_sorted_by_username.json",
        help="Output JSON file path"
    )

    # Parse the command-line arguments into `args`.
    args = parser.parse_args()

    # --------------------------
    # Load the ratings dataset
    # --------------------------
    # NOTE: We pass `names=[...]`, which assumes the CSV has NO header row.
    # If the ratings CSV already contains a header row, we'd need to remove `names=...`
    # or specify header=0 to avoid treating the first data row incorrectly.
    df1 = pd.read_csv(
        args.ratings,
        names=["user_id", "username", "item_id", "rating", "timestamp"]
    )

    # Convert unix timestamps (seconds since epoch) to pandas datetime.
    # errors="coerce" turns invalid timestamps into NaT instead of crashing.
    df1["timestamp"] = pd.to_datetime(df1["timestamp"], unit="s", errors="coerce")

    # -------------------------------------------------
    # Create user_id -> username mapping
    # -------------------------------------------------
    # In a clean dataset, each user_id should map to exactly one username.
    # If there are multiple usernames for a given user_id (data quality issue),
    # this chooses the most frequently occurring username for that user_id.
    user_name_map = (
        df1.groupby("user_id")["username"]
        .agg(lambda s: s.value_counts().idxmax())
        .to_dict()
    )

    # --------------------------
    # Load movie titles dataset
    # --------------------------
    # This CSV is expected to have at least:
    # - item_id (numeric movie identifier)
    # - title (movie title string)
    df2 = pd.read_csv(args.titles)

    # ---------------------------------------------
    # Merge ratings with titles (left join)
    # ---------------------------------------------
    # Left join ensures we keep every rating row from df1.
    # If a movie ID doesn't exist in df2, title will become NaN.
    join_df = df1.merge(df2, on="item_id", how="left")

    # ---------------------------------------------
    # Build interaction matrix (users x movies)
    # ---------------------------------------------
    # pivot_table creates a matrix where:
    # - index="user_id" => each user_id becomes a row
    # - columns="title" => each title becomes a column
    # - values="rating" => cell contains rating
    # aggfunc="mean" handles duplicate (user_id, title) pairs by averaging.
    # fill_value=0.0 makes missing ratings explicit as 0.0 (our "unrated" marker).
    interaction_df = join_df.pivot_table(
        index="user_id",
        columns="title",
        values="rating",
        aggfunc="mean",
        fill_value=0.0
    )

    # Convert to NumPy array for efficient similarity computations.
    interaction = interaction_df.to_numpy()

    # Keep the row/column labels so we can convert matrix indices back to real IDs/titles.
    users = interaction_df.index.to_numpy()
    movies = interaction_df.columns.to_numpy()

    # ---------------------------------------------
    # Compute user-user cosine similarity
    # ---------------------------------------------
    # This produces a square matrix S where:
    # S[i, j] = cosine similarity between user i and user j
    user_similarity = cosine_similarity(interaction)

    # ------------------------------------------------------------
    # Build output as a list of objects so we can sort by username
    # ------------------------------------------------------------
    user_objects: list[dict] = []

    # For each user in our interaction matrix:
    for uid in users:
        uid_int = int(uid)

        # Look up that user's username; default to "Unknown" if missing.
        username = str(user_name_map.get(uid_int, "Unknown"))

        # Generate recommended movies + average neighbor rating score
        recs = _recommend_top_n_for_user_with_scores(
            user_id=uid_int,
            interaction=interaction,
            users=users,
            movies=movies,
            user_similarity=user_similarity,
            k=args.k,
            top_n=args.topn
        )

        # Add this user record to our list
        user_objects.append(
            {
                "user_id": uid_int,
                "username": username,
                "recommendations": recs
            }
        )

    # Sort alphabetically by username (case-insensitive so "alice" and "Alice" behave consistently).
    user_objects.sort(key=lambda x: x["username"].casefold())

    # ---------------------------------------------
    # Write JSON output
    # ---------------------------------------------
    # We write a LIST (not a dict) so ordering is preserved in the output file
    # (JSON objects/dicts historically don't guarantee order everywhere).
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(user_objects, f, ensure_ascii=False, indent=2)

    # Small confirmation message
    print(f"Wrote {args.topn} scored recs (+ usernames) for {len(user_objects):,} users -> {args.out}")


# Standard Python entrypoint guard:
# This ensures main() runs only when this file is executed directly:
#   python app.py
# and NOT when it is imported as a module from somewhere else.
if __name__ == "__main__":
    main()
