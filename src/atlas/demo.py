"""Run Atlas without API calls unless answer generation is requested."""

import argparse
import json
import os
from pathlib import Path
import sys

from .config import AtlasConfig
from .generation import GenerationError


ROOT = Path(__file__).resolve().parents[2]


def show_answer(result):
    if result.generation:
        print(result.generation.answer)
        print(f"\nStatus: {result.generation.status}")
        if result.generation.reason:
            print(f"Reason: {result.generation.reason}")
    else:
        print("Selected evidence (retrieval only):")
        for passage in result.passages:
            print(f"\n[{passage.id}] {passage.chunk.text}")
    used_ids = ({e.passage_id for c in result.generation.claims for e in c.evidence}
                if result.generation and result.generation.status == "answered"
                else {p.id for p in result.passages})
    print("\nSources:")
    for passage in result.passages:
        if passage.id in used_ids:
            print(f"[{passage.id}] {passage.chunk.source} "
                  f"(characters {passage.chunk.start}:{passage.chunk.end})")
    if result.generation and result.generation.status == "answered":
        print("\nSupporting quotes:")
        for claim in result.generation.claims:
            for evidence in claim.evidence:
                print(f'[{evidence.passage_id}] "{evidence.quote}"')
    print("\nTimings (ms):", json.dumps(result.timings_ms, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Atlas grounded research Q&A")
    parser.add_argument("--documents", type=Path, help="A PDF, text file, or document directory")
    parser.add_argument("--question", help="Generate an answer unless --retrieve-only is set")
    parser.add_argument("--retrieve-only", action="store_true", help="Make no generation API calls")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate retrieval; no API calls by default")
    parser.add_argument("--generate-eval", action="store_true", help="Generate one answer per evaluation case")
    parser.add_argument("--cases", type=Path, help="Custom evaluation JSON")
    parser.add_argument("--output", type=Path, help="Write complete answer/evaluation JSON")
    parser.add_argument("--model", help="Override the generation model")
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--context-k", type=int, default=4)
    parser.add_argument("--max-output-tokens", type=int, default=1600)
    parser.add_argument("--baseline-overview", action="store_true",
                        help="Run the existing DistilBART paper overview on text input")
    args = parser.parse_args()
    if args.generate_eval and not args.evaluate:
        parser.error("--generate-eval requires --evaluate.")
    if args.evaluate and args.question:
        parser.error("Choose --evaluate or --question.")
    if args.retrieve_only and args.generate_eval:
        parser.error("--retrieve-only conflicts with --generate-eval.")
    if args.baseline_overview and (args.evaluate or args.question):
        parser.error("Run --baseline-overview separately.")
    if args.cases and not args.evaluate:
        parser.error("--cases requires --evaluate.")
    if args.documents and args.evaluate and not args.cases:
        parser.error("Supply --cases for your documents; built-in labels describe the example corpus.")

    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env", override=False)
    documents = args.documents or ROOT / "data/examples"
    if args.baseline_overview:
        from .paper import split_paper_sections
        from .summarization import summarize_paper_sections, format_paper_summary
        paper = args.documents or ROOT / "data/examples/sleep_study.txt"
        if not paper.is_file() or paper.suffix.lower() not in {".txt", ".md"}:
            parser.error("The baseline overview needs one .txt or .md paper.")
        print(format_paper_summary(summarize_paper_sections(
            split_paper_sections(paper.read_text(encoding="utf-8"))
        )))
        return 0

    generates = args.generate_eval or (args.question and not args.retrieve_only)
    if generates and not os.getenv("OPENAI_API_KEY"):
        parser.error("Add OPENAI_API_KEY to .env. Retrieval evaluation requires no key.")
    from .pipeline import Atlas
    from .evaluation import evaluate, load_cases
    config = AtlasConfig(
        generation_model=args.model or os.getenv("ATLAS_GENERATION_MODEL")
                         or AtlasConfig.generation_model,
        candidate_k=args.candidate_k, context_k=args.context_k,
        max_output_tokens=args.max_output_tokens
    )
    atlas = Atlas(config, cache_dir=ROOT / ".cache/embeddings")
    atlas.index_path(documents)
    if args.evaluate or not args.question:
        cases = args.cases or ROOT / "data/evaluation/rag_cases.json"
        payload = evaluate(atlas, load_cases(cases), generate=args.generate_eval)
        print(json.dumps(payload["summary"], indent=2))
    else:
        result = (atlas.search(args.question) if args.retrieve_only
                  else atlas.ask(args.question))
        show_answer(result)
        payload = result.to_dict()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GenerationError, ValueError, OSError, RuntimeError) as error:
        print(f"Atlas: {error}", file=sys.stderr)
        sys.exit(1)
