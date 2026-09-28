import json
#Pull in the components needed for our retrieval test
from .chunking import split_text, split_sentences
from .retrieval import Retriever
from .evaluation import hit_at_k
from .reranking import Reranker
from .paper import split_paper_sections
from .summarization import summarize_paper_sections, format_paper_summary

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

#Create one mock research paper so we can test the complete paper-understanding pipeline before introducing PDF extraction
sample_paper = """
Abstract
This study examines the relationship between sleep duration and academic performance among university students.

1. Introduction
Sleep is associated with memory, attention, and cognitive performance.
Previous research has reported mixed findings regarding the amount of sleep associated with stronger academic outcomes.
This study investigates whether sleep duration is associated with university exam performance.

2. Methods
We recruited 500 university students.
Participants reported their average nightly sleep duration and weekly exercise habits.
Academic performance was measured using average exam scores.
The analysis controlled for participant age.

3. Results
Students sleeping seven to nine hours had higher average exam scores.
Students sleeping fewer than five hours had the lowest average scores.
The relationship between sleep duration and academic performance remained after controlling for age.
Exercise habits were also recorded but were not the primary focus of the analysis.

4. Discussion
The findings suggest that moderate sleep duration is associated with stronger academic performance.
Very short sleep duration was associated with poorer performance.
Additional research would be needed to determine whether the relationship is causal.

5. Conclusion
Students reporting seven to nine hours of sleep generally had stronger academic outcomes than students reporting very short sleep durations.

References
Smith et al.
"""


#Separate the paper into standardized sections so summarization can process Introduction, Methods, Results, and other components independently
paper_sections = split_paper_sections(
    sample_paper
)


print("\nParsed paper sections:")

#Print the detected section names first so we can verify that paper.py correctly understood the document structure before evaluating its summaries
for section_name in paper_sections:
    print(f"- {section_name}")


#Generate a separate transformer summary for each detected section so information from different parts of the research paper remains organized
paper_summary = summarize_paper_sections(
    paper_sections
)


#Convert the internal summary dictionary into a readable overview that resembles the format Atlas could eventually show to a user
formatted_summary = format_paper_summary(
    paper_summary
)


print("\nAtlas paper overview")
print(formatted_summary)