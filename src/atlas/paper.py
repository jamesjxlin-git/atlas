import re


SECTION_ALIASES = {
    "abstract": "abstract",
    "introduction": "introduction",
    "background": "introduction",

    "methods": "methods",
    "methodology": "methods",
    "materials and methods": "methods",
    "experimental methods": "methods",

    "results": "results",
    "findings": "results",

    "discussion": "discussion",

    "conclusion": "conclusion",
    "conclusions": "conclusion",

    "limitations": "limitations",

    "references": "references",
    "bibliography": "references",
}


def identify_section_heading(line):
    """
    Check whether a line looks like a major research-paper section heading.

    Returns the standardized section name if it finds one.
    Otherwise returns None.
    """

    #Normalize the line so headings with different capitalization or surrounding spaces can still match our dictionary
    heading = line.strip().lower()

    #Remove numeric prefixes such as "1", "2.", or "3.1" so numbered headings can still match values like "methods" or "results"
    heading = re.sub(
        r"^\d+(?:\.\d+)*\.?\s*",
        "",
        heading
    )

    #Normalize repeated whitespace so headings like "Materials   and   Methods" still match "materials and methods"
    heading = " ".join(heading.split())

    #Look up the cleaned heading in our alias dictionary; get() returns the standardized name if found and None if it is not recognized
    return SECTION_ALIASES.get(heading)


def split_paper_sections(text):
    """
    Split research-paper text into major sections.

    Returns a dictionary where:
        key = standardized section name
        value = text belonging to that section
    """

    #Store the final parsed sections so Atlas can later access sections individually instead of treating the whole paper as one block of text
    sections = {}

    #Track which section we are currently reading so each following line can be assigned to the correct part of the paper
    current_section = None

    #Temporarily collect the lines belonging to the current section before combining them into one continuous string
    current_lines = []

    #Split the full paper at line breaks so each line can be checked separately for a possible section heading
    for line in text.splitlines():

        #Check whether the current line is a recognized heading; normal paper text will return None
        section_heading = identify_section_heading(line)

        if section_heading:

            #A new heading means the previous section has ended, so save the text we collected before switching sections
            if current_section and current_lines:

                #Join the collected lines into one string because later summarization will work with section-level text rather than a list of lines
                section_text = " ".join(current_lines).strip()

                #If a standardized section appears multiple times, append the new text instead of overwriting the earlier content
                if current_section in sections:
                    sections[current_section] += "\n\n" + section_text
                else:
                    sections[current_section] = section_text

            #Stop parsing at the bibliography because reference entries should not be treated as research-paper content for summarization
            if section_heading == "references":
                return sections

            #Update the tracker so future lines are assigned to the newly detected section
            current_section = section_heading

            #Clear the old collected lines because we are starting a new section
            current_lines = []

            #Skip the rest of this loop iteration so the heading itself is not accidentally stored as section content
            continue

        #Only save non-empty text after a recognized section has started so front-matter and blank lines do not pollute the parsed sections
        if current_section and line.strip():

            #Remove surrounding whitespace before storing the line so the final section text stays clean
            current_lines.append(line.strip())

    #The loop only saves a section when a new heading is encountered, so this block saves the final section after the loop ends
    if current_section and current_lines:

        #Combine the remaining lines into one final section string
        section_text = " ".join(current_lines).strip()

        #Append instead of overwrite in case the final standardized section already appeared earlier in the paper
        if current_section in sections:
            sections[current_section] += "\n\n" + section_text
        else:
            sections[current_section] = section_text

    #Return the structured section dictionary so other Atlas modules can use sections independently
    return sections


if __name__ == "__main__":

    #Create a small fake research paper so we can test the parser without needing PDF extraction yet
    sample_paper = """
    Abstract
    This paper studies how sleep affects academic performance.

    1. Introduction
    Sleep affects several aspects of cognitive performance.
    Previous studies have reported mixed findings.

    2. Methods
    We recruited 500 university students.
    Participants completed a sleep questionnaire.

    3. Results
    Students sleeping seven to nine hours performed better.

    4. Conclusion
    Moderate sleep duration was associated with better performance.

    References
    Smith et al.
    """

    #Run the parser on the sample paper so we can verify that each section is being identified and stored correctly
    sections = split_paper_sections(sample_paper)

    #Print each parsed section so we can manually inspect whether the output matches the structure of the fake paper
    for name, content in sections.items():
        print(f"\n{name.upper()}")
        print(content)