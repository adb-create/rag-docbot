Retrieval quality on 28 answerable + 6 unanswerable questions.
`hit@1600ch` = answer found within the best chunks that fit in 1600 characters of context (equal context for every chunk size).

| Embedder | Chunk size | Overlap | Chunks | hit@1 | hit@3 | hit@5 | hit@1600ch | MRR | Refuses unanswerable | Wrongly refuses |
|---|---|---|---|---|---|---|---|---|---|---|
| tfidf | 200 | 30 | 89 | 75% | 96% | 96% | 96% | 0.851 | 83% | 0% |
| tfidf | 400 | 60 | 39 | 86% | 96% | 96% | 96% | 0.906 | 83% | 0% |
| tfidf | 600 | 90 | 27 | 86% | 100% | 100% | 100% | 0.917 | 83% | 0% |
| tfidf | 800 | 120 | 21 | 93% | 96% | 96% | 93% | 0.943 | 83% | 0% |
| tfidf | 1200 | 180 | 14 | 100% | 100% | 100% | 100% | 1.000 | 83% | 0% |
