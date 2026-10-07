"""The same discovery, reading, and evaluation services used by the web app."""

import argparse
import json
import os
from pathlib import Path
import sys

from dotenv import load_dotenv

from .config import AtlasConfig
from .generation import GenerationError
from .metadata import MetadataClient, MetadataError

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description="Atlas end-to-end research assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    discover = commands.add_parser("discover", help="Shortlist scholarly papers from public metadata")
    discover.add_argument("topic")
    discover.add_argument("--count", type=int, default=5)
    discover.add_argument("--preference", choices=["balanced", "foundational", "recent"], default="balanced")
    discover.add_argument("--since-year", type=int)
    discover.add_argument("--open-access", action="store_true")
    ask = commands.add_parser("ask", help="Ask a question about uploaded/local documents")
    ask.add_argument("question")
    ask.add_argument("--documents", type=Path, required=True)
    ask.add_argument("--retrieve-only", action="store_true")
    overview = commands.add_parser("overview", help="Generate four cited overview sections")
    overview.add_argument("--documents", type=Path, required=True)
    evaluation = commands.add_parser("evaluate", help="Evaluate retrieval; generation is opt-in")
    evaluation.add_argument("--documents", type=Path)
    evaluation.add_argument("--cases", type=Path)
    evaluation.add_argument("--generate", action="store_true")
    for command in [discover, ask, overview, evaluation]:
        command.add_argument("--output", type=Path, help="Export JSON with provenance and evidence")
    args = parser.parse_args()
    load_dotenv(ROOT / ".env", override=False)
    requires_key = args.command == "overview" or (args.command == "ask" and not args.retrieve_only) or (
        args.command == "evaluate" and args.generate
    )
    if requires_key and not os.getenv("OPENAI_API_KEY"):
        parser.error("Add OPENAI_API_KEY to .env to generate answers; discovery/retrieval require no OpenAI key.")
    if args.command == "evaluate" and bool(args.documents) != bool(args.cases):
        parser.error("Supply both --documents and --cases, or neither for the graded synthetic demo.")
    from .reranking import Reranker
    if args.command == "discover":
        from .discovery import PaperDiscovery
        metadata = MetadataClient(cache_dir=ROOT / ".cache/metadata",
                                  openalex_key=os.getenv("OPENALEX_API_KEY") or None,
                                  contact_email=os.getenv("CROSSREF_CONTACT_EMAIL") or None)
        try:
            payload = PaperDiscovery(metadata, Reranker()).recommend(
                args.topic, count=args.count, preference=args.preference,
                since_year=args.since_year, open_access=args.open_access
            )
        finally:
            metadata.close()
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        from .pipeline import Atlas
        from .evaluation import evaluate, load_cases
        config = AtlasConfig(generation_model=os.getenv("ATLAS_GENERATION_MODEL") or AtlasConfig.generation_model)
        # Local reading indexes are session-scoped; only model weights and public metadata are cached.
        atlas = Atlas(config)
        atlas.index_path(args.documents or ROOT / "data/graded_demo")
        if args.command == "ask":
            result = atlas.search(args.question) if args.retrieve_only else atlas.ask(args.question)
            from .demo import show_answer
            show_answer(result)
            payload = result.to_dict()
        elif args.command == "overview":
            from .overview import build_overview
            payload = build_overview(atlas)
            for section in payload["sections"]:
                print(section["heading"] + "\n" + section["result"]["generation"]["answer"] + "\n")
        else:
            cases = load_cases(args.cases or ROOT / "data/evaluation/graded_cases.json")
            payload = evaluate(atlas, cases, generate=args.generate)
            print(json.dumps(payload["summary"], indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (GenerationError, MetadataError, ValueError, OSError, RuntimeError) as error:
        print(f"Atlas: {error}", file=sys.stderr)
        sys.exit(1)
