import re

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
SUMMARY_MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

MAX_CHUNK_TOKENS = 900


#Keep the models empty until a function actually needs them so importing summarization.py does not immediately load large neural-network weights into memory
_embedding_model = None
_summary_tokenizer = None
_summary_model = None


def get_embedding_model():
    """
    Load and return the embedding model used by the extractive baseline.
    """

    global _embedding_model

    #Only load the embedding model the first time it is needed so later calls can reuse the same model instead of repeatedly loading its weights
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def get_summary_model():
    """
    Load and return the tokenizer and transformer used by the
    abstractive summarization baseline.
    """

    global _summary_tokenizer
    global _summary_model

    #Load the tokenizer the first time it is needed so normal text can be converted into the token IDs expected by DistilBART
    if _summary_tokenizer is None:
        _summary_tokenizer = AutoTokenizer.from_pretrained(
            SUMMARY_MODEL_NAME
        )

    #Load the transformer the first time it is needed so its pretrained weights can be reused for every later summary instead of loading again
    if _summary_model is None:
        _summary_model = AutoModelForSeq2SeqLM.from_pretrained(
            SUMMARY_MODEL_NAME
        )

        #Switch the model into evaluation mode because Atlas is generating summaries rather than training or changing its weights
        _summary_model.eval()

        #Set BART's required beginning-of-sequence token directly in the generation configuration so later generate() calls use the expected setup
        _summary_model.generation_config.forced_bos_token_id = 0

    #Return both objects only after confirming that each one has been loaded so other functions can safely tokenize text and run the transformer
    return _summary_tokenizer, _summary_model


def split_sentences(text):
    """
    Split text into individual sentences.

    Returns a list of sentence strings.
    """

    #Normalize repeated whitespace so line breaks and formatting artifacts from parsed papers do not interfere with sentence detection
    cleaned_text = " ".join(
        text.split()
    )

    #Split after sentence-ending punctuation while keeping the punctuation attached so each result remains a complete readable sentence
    sentences = re.split(
        r"(?<=[.!?])\s+",
        cleaned_text
    )

    #Remove empty results and surrounding whitespace so downstream models only receive usable sentence text
    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def extractive_summary(
    text,
    num_sentences=3
):
    """
    Select the sentences that best represent the overall semantic
    meaning of the supplied text.

    This is Atlas's extractive summarization baseline.
    """

    #Break the text into individual sentences because this baseline summarizes by selecting existing sentences instead of generating new language
    sentences = split_sentences(text)

    #Return all available sentences when the section is already shorter than the requested summary because ranking would not remove anything
    if len(sentences) <= num_sentences:
        return sentences

    #Retrieve the embedding model only when this baseline is actually used so the model does not consume memory unnecessarily
    embedding_model = get_embedding_model()

    #Convert each sentence into a semantic vector so similarity can be based on meaning rather than exact word overlap
    embeddings = embedding_model.encode(
        sentences,
        convert_to_tensor=True
    )

    #Average the sentence vectors across dimension 0 to create one vector representing the approximate semantic center of the complete section
    section_embedding = embeddings.mean(
        dim=0
    )

    #Add a dimension so PyTorch can compare the single section vector against every sentence vector in one operation
    section_embedding = section_embedding.unsqueeze(0)

    #Calculate how closely every sentence points in the same semantic direction as the section center; higher values indicate more representative sentences
    similarities = F.cosine_similarity(
        embeddings,
        section_embedding,
        dim=1
    )

    #Retrieve the positions of the highest-scoring sentences so the baseline keeps the text most representative of the section's overall meaning
    top_indices = torch.topk(
        similarities,
        k=num_sentences
    ).indices

    #Sort the selected positions because semantic ranking may reorder them, while the final summary should preserve the paper's original reading order
    selected_indices = sorted(
        top_indices.tolist()
    )

    #Map the selected numerical positions back to the original sentence strings so the function returns readable text instead of tensor indices
    summary_sentences = [
        sentences[index]
        for index in selected_indices
    ]

    return summary_sentences


def split_for_summarization(
    text,
    max_tokens=MAX_CHUNK_TOKENS
):
    """
    Divide long text into model-sized chunks while preserving
    complete sentences whenever possible.
    """

    #Load the tokenizer because transformer input limits are measured in tokens rather than characters or words
    summary_tokenizer, _ = get_summary_model()

    #Split into sentences first so Atlas can create chunks at natural language boundaries rather than cutting sentences arbitrarily
    sentences = split_sentences(text)

    chunks = []

    current_sentences = []
    current_token_count = 0

    for sentence in sentences:

        #Convert the sentence into token IDs only to measure how much of the transformer's input capacity that sentence would consume
        sentence_tokens = summary_tokenizer.encode(
            sentence,
            add_special_tokens=False
        )

        #Handle a single unusually long sentence separately because it cannot fit inside one normal model-sized chunk
        if len(sentence_tokens) > max_tokens:

            #Save any normal sentences that were already collected before processing the oversized sentence so the original text order remains intact
            if current_sentences:
                chunks.append(
                    " ".join(current_sentences)
                )

                current_sentences = []
                current_token_count = 0

            #Break the oversized sentence directly at token boundaries because preserving it as one sentence would exceed the model's input capacity
            for start in range(
                0,
                len(sentence_tokens),
                max_tokens
            ):
                token_chunk = sentence_tokens[
                    start:start + max_tokens
                ]

                #Convert each token block back into readable text so it can later be passed through the normal summarization pipeline
                chunks.append(
                    summary_tokenizer.decode(
                        token_chunk,
                        skip_special_tokens=True
                    )
                )

            continue

        #Finish the current chunk before adding a sentence that would cause the combined token count to exceed the safe input limit
        if (
            current_sentences
            and current_token_count + len(sentence_tokens) > max_tokens
        ):
            chunks.append(
                " ".join(current_sentences)
            )

            current_sentences = []
            current_token_count = 0

        #Add the sentence after confirming that it fits and update the running token count so the next sentence can be checked correctly
        current_sentences.append(sentence)
        current_token_count += len(sentence_tokens)

    #Save the final partially filled chunk because reaching the end of the loop does not automatically trigger the token-limit condition above
    if current_sentences:
        chunks.append(
            " ".join(current_sentences)
        )

    return chunks


def generate_chunk_summary(
    text,
    max_summary_tokens=120
):
    """
    Generate an abstractive summary for one model-sized block of text.

    This currently uses Atlas's pretrained transformer baseline.
    """

    #Load the tokenizer and pretrained transformer only when abstractive summarization is actually requested
    summary_tokenizer, summary_model = get_summary_model()

    #Convert the source text into numerical token tensors because the transformer processes token IDs rather than raw strings
    inputs = summary_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=1024
    )

    #Measure the source length so very short sections can be handled differently from sections that actually require compression
    input_token_count = inputs["input_ids"].shape[1]

    #Avoid forcing a generative model to rewrite extremely short sections because they are already concise and generation can unnecessarily distort their meaning
    if input_token_count < 60:
        return " ".join(
            text.split()
        )

    #Allow enough output space for the transformer to finish complete thoughts while still keeping the result substantially shorter than a long source section
    output_max_length = min(
        max_summary_tokens,
        max(
            60,
            int(input_token_count * 0.6)
        )
    )

    #Require only a modest minimum length so the model can stop naturally once it has captured the important information
    output_min_length = min(
        25,
        max(
            10,
            output_max_length // 4
        )
    )

    #Disable gradient calculation because this stage only uses the pretrained weights for inference rather than updating them through training
    with torch.no_grad():

        #Use deterministic beam search to compare several possible generated sequences while avoiding randomness during baseline evaluation
        summary_tokens = summary_model.generate(
            **inputs,
            min_length=output_min_length,
            max_length=output_max_length,
            num_beams=4,
            do_sample=False,
            no_repeat_ngram_size=3,
            length_penalty=2.0,
            early_stopping=True,
            forced_bos_token_id=0
        )

    #Convert the generated token IDs back into readable text and remove model-specific special tokens that should not appear in the summary
    summary = summary_tokenizer.decode(
        summary_tokens[0],
        skip_special_tokens=True
    )

    #Clean spacing before punctuation because this particular baseline model can occasionally decode periods and commas with unnecessary leading spaces
    summary = re.sub(
        r"\s+([.,!?;:])",
        r"\1",
        summary
    )

    return summary.strip()


def generate_summary(
    text,
    max_chunk_tokens=MAX_CHUNK_TOKENS,
    max_summary_tokens=120
):
    """
    Generate an abstractive summary of short or long text.

    Long text is summarized hierarchically:
        original text
            ->
        model-sized chunks
            ->
        chunk summaries
            ->
        final combined summary
    """

    #Return an empty result for empty input so Atlas does not send meaningless content through the transformer
    if not text or not text.strip():
        return ""

    #Divide the original text into model-sized blocks so sections larger than the transformer's context limit can still be summarized
    chunks = split_for_summarization(
        text,
        max_tokens=max_chunk_tokens
    )

    #Generate a condensed representation of every chunk so information from the entire section can survive the first summarization stage
    chunk_summaries = [
        generate_chunk_summary(
            chunk,
            max_summary_tokens=max_summary_tokens
        )
        for chunk in chunks
    ]

    #Return immediately when the complete text fit into one chunk because no second summarization stage is necessary
    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    #Combine the first-stage summaries so Atlas can synthesize information originally spread across several parts of a long section
    combined_summary = " ".join(
        chunk_summaries
    )

    #Split the combined summaries again in case a very long paper still produces more condensed text than the transformer can safely process at once
    second_level_chunks = split_for_summarization(
        combined_summary,
        max_tokens=max_chunk_tokens
    )

    #If the first-stage summaries now fit together, generate one final synthesis that turns them into a cohesive section-level summary
    if len(second_level_chunks) == 1:
        return generate_chunk_summary(
            combined_summary,
            max_summary_tokens=max_summary_tokens
        )

    #Summarize each second-level block when extremely long source material still cannot fit into one final transformer input
    second_level_summaries = [
        generate_chunk_summary(
            chunk,
            max_summary_tokens=max_summary_tokens
        )
        for chunk in second_level_chunks
    ]

    #Combine the heavily condensed second-level summaries so the final generation step has information representing every part of the original section
    final_input = " ".join(
        second_level_summaries
    )

    #Generate one final section summary rather than returning several disconnected intermediate summaries to the user
    return generate_chunk_summary(
        final_input,
        max_summary_tokens=max_summary_tokens
    )


def summarize_paper_sections(sections):
    """
    Generate structured summaries for the major research-paper
    sections identified by paper.py.

    Returns a dictionary containing only sections that were found.
    """

    #Define the section order explicitly so the final overview follows the logical structure of a research paper rather than arbitrary dictionary ordering
    section_order = [
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "limitations",
        "conclusion"
    ]

    paper_summary = {}

    for section_name in section_order:

        #Skip sections that paper.py did not find because research papers do not always use every standard heading
        if section_name not in sections:
            continue

        section_text = sections[
            section_name
        ]

        #Generate a condensed version of each available section so Atlas can eventually assemble them into one structured paper overview
        paper_summary[section_name] = generate_summary(
            section_text
        )

    return paper_summary


def format_paper_summary(paper_summary):
    """
    Convert Atlas's internal summary dictionary into a readable
    structured research-paper overview.
    """

    #Map internal standardized section names to reader-friendly labels so implementation terminology does not need to appear in the final product
    display_names = {
        "abstract": "Abstract",
        "introduction": "Background / Introduction",
        "methods": "Methods",
        "results": "Main Results",
        "discussion": "Discussion",
        "limitations": "Limitations",
        "conclusion": "Conclusion"
    }

    formatted_sections = []

    for section_name, summary in paper_summary.items():

        #Use a predefined display name when available while allowing future section types to fall back to normal title capitalization
        heading = display_names.get(
            section_name,
            section_name.title()
        )

        #Combine each heading with its generated summary so the final synopsis remains structured and easy for a user to scan
        formatted_sections.append(
            f"{heading}\n"
            f"{'-' * len(heading)}\n"
            f"{summary}"
        )

    #Insert blank lines between sections so the complete Atlas overview remains readable when printed in the terminal or shown in a later interface
    return "\n\n".join(
        formatted_sections
    )


if __name__ == "__main__":

    #Use the same sample text for both approaches so we can directly compare sentence selection against transformer-generated summarization
    sample_results = """
    We recruited 500 university students for the study.
    Students reported their average nightly sleep duration.
    Students sleeping seven to nine hours had higher average exam scores.
    Students sleeping fewer than five hours had the lowest average scores.
    Participants also reported their weekly exercise habits.
    The relationship between sleep duration and academic performance remained after controlling for age.
    """

    #Run the embedding-based method first so Atlas retains a simple and low-hallucination reference point for later model evaluation
    baseline = extractive_summary(
        sample_results,
        num_sentences=3
    )

    print("\nEXTRACTIVE BASELINE")

    for sentence in baseline:
        print(f"- {sentence}")

    #Run the pretrained transformer on exactly the same source so we can observe both its improved synthesis and any unsupported information it introduces
    transformer_summary = generate_summary(
        sample_results
    )

    print("\nTRANSFORMER BASELINE")
    print(transformer_summary)