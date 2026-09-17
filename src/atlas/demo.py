import json
#Pull in the components needed for our retrieval test
from .chunking import split_text, split_sentences
from .retrieval import Retriever
from .evaluation import hit_at_k
from .reranking import Reranker


#Use a sample document with several different concepts
sample = (
    "Machine learning models learn patterns from data. "
    "Retrieval systems search for information relevant to a query. "
    "Language models generate text based on the context they receive. "
    "RAG combines retrieval with generation so a model can answer using external sources."
)


#Split the sample document into overlapping chunks
chunks = split_text(
    text=sample,
    source="sample.txt",
    size=100,
    overlap=20
)
#Improved approach: preserve complete sentence boundaries
chunks = split_sentences(
    sample,
    source="sample",
    size=100
)

#Create the retriever and index the chunks
retriever = Retriever()
retriever.index(chunks)

#Load the reranking model once so it can rescore retrieved candidates
reranker = Reranker()


#Define questions where we already know what evidence should be retrieved
with open("data/raw/eval_cases.json", "r") as file: 
    tests = json.load(file)


retrieval_hit1_total = 0
retrieval_hit3_total = 0

rerank_hit1_total = 0
rerank_hit3_total = 0


#Compare the original dense retrieval ranking with the reranked results
for test in tests:

    #Retrieve a broader candidate set before reranking
    retrieved = retriever.search(
        test["query"],
        k=4
    )

    #Rescore the retrieved candidates and keep the best three
    reranked = reranker.rerank(
        test["query"],
        retrieved,
        k=3
    )

    print(f"\nQuestion: {test['query']}")

    print("\nDense retrieval:")
    for rank, (piece, score) in enumerate(retrieved[:3], start=1):
        print(f"Result {rank} | Score: {score:.3f}")
        print(piece.text)

    print("\nReranked:")
    for rank, (piece, score) in enumerate(reranked, start=1):
        print(f"Result {rank} | Score: {score:.3f}")
        print(piece.text)

    retrieval_hit1 = hit_at_k(
        retrieved,
        expected_text=test["expected"],
        k=1
    )

    retrieval_hit3 = hit_at_k(
        retrieved,
        expected_text=test["expected"],
        k=3
    )

    rerank_hit1 = hit_at_k(
        reranked,
        expected_text=test["expected"],
        k=1
    )

    rerank_hit3 = hit_at_k(
        reranked,
        expected_text=test["expected"],
        k=3
    )

    retrieval_hit1_total += retrieval_hit1
    retrieval_hit3_total += retrieval_hit3

    rerank_hit1_total += rerank_hit1
    rerank_hit3_total += rerank_hit3


total = len(tests)

print("\nOverall retrieval performance")

print("\nDense retrieval:")
print(f"Hit@1: {retrieval_hit1_total / total:.2f}")
print(f"Hit@3: {retrieval_hit3_total / total:.2f}")

print("\nReranked retrieval:")
print(f"Hit@1: {rerank_hit1_total / total:.2f}")
print(f"Hit@3: {rerank_hit3_total / total:.2f}")