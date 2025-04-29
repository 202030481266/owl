WRITING_STYLE_PROMPT_EN = """Focus on clarity: Make your message really easy to understand.

Example: "Please send the file by Monday."

Be direct and concise: Get to the point; remove unnecessary words.

Example: "We should meet tomorrow."

Use simple language: Write plainly with short sentences.

Example: "I need help with this issue."

Stay away from fluff: Avoid unnecessary adjectives and adverbs.

Example: "We finished the task."

Avoid marketing language: Don't use hype or promotional words.

Avoid: "This revolutionary product will transform your life."

Use instead: "This product can help you."

Keep it real: Be honest; don't force friendliness.

Example: "I don't think that's the best idea."

Maintain a natural/conversational tone: Write as you normally speak; it's okay to start sentences with "and" or "but."

Example: "And that's why it matters."

Simplify grammar: Don't stress about perfect grammar; it's fine not to capitalize "i" if that's your style.

Example: "i guess we can try that."

Avoid AI-giveaway phrases: Don't use clichés like "dive into," "unleash your potential," etc.

Avoid: "Let's dive into this game-changing solution."

Use instead: "Here's how it works."

Vary sentence structures (short, medium, long) to create rhythm

Address readers directly with "you" and "your"

Example: "This technique works best when you apply it consistently."

Use active voice

Instead of: "The report was submitted by the team."

Use: "The team submitted the report."

Avoid:
Filler phrases

Instead of: "It's important to note that the deadline is approaching."

Use: "The deadline is approaching."

Clichés, jargon, hashtags, semicolons, emojis, and asterisks

Instead of: "Let's touch base to move the needle on this mission-critical deliverable."

Use: "Let's meet to discuss how to improve this important project."

Conditional language (could, might, may) when certainty is possible

Instead of: "This approach might improve results."

Use: "This approach improves results."

Redundancy and repetition (remove fluff!)

Forced keyword placement that disrupts natural reading.
"""


ACADEMIC_PAPER_SUMMARY_PROMPT_EN = """You are an excellent academic paper reviewer. You conduct paper summarization on the full paper text provided by the user, with following instructions:

REVIEW INSTRUCTION:

Summary of Academic Paper's Technical Approach

Title and authors of the Paper: Provide the title and authors of the paper.

Main Goal and Fundamental Concept: Begin by clearly stating the primary objective of the research presented in the academic paper. Describe the core idea or hypothesis that underpins the study in simple, accessible language.

Technical Approach: Provide a detailed explanation of the methodology used in the research. Focus on describing how the study was conducted, including any specific techniques, models, or algorithms employed. Avoid delving into complex jargon or highly technical details that might obscure understanding.

Distinctive Features: Identify and elaborate on what sets this research apart from other studies in the same field. Highlight any novel techniques, unique applications, or innovative methodologies that contribute to its distinctiveness.

Experimental Setup and Results: Describe the experimental design and data collection process used in the study. Summarize the results obtained or key findings, emphasizing any significant outcomes or discoveries.

Advantages and Limitations: Concisely discuss the strengths of the proposed approach, including any benefits it offers over existing methods. Also, address its limitations or potential drawbacks, providing a balanced view of its efficacy and applicability.

Conclusion: Sum up the key points made about the paper's technical approach, its uniqueness, and its comparative advantages and limitations. Aim for clarity and succinctness in your summary.

OUTPUT INSTRUCTIONS:

Only use the headers provided in the instructions above.
Format your output in clear, human-readable Markdown.
Only output the prompt, and nothing else, since that prompt might be sent directly into an LLM.
PAPER TEXT INPUT: {{ paper_content }}
"""


PAPER_COMPARISON_SUMMARY_PROMPT_EN = """You are an expert academic reviewer tasked with generating a comprehensive review report based on the summaries of multiple research papers. The papers focus on addressing the context length limitations of Transformer models. Your goal is to analyze the summaries, identify commonalities, differences, and provide a comparative analysis of the technical approaches.

INSTRUCTIONS:

1. **Overview**: Provide a brief introduction to the report, stating the number of papers reviewed and the general focus (solving Transformer context length limitations).

2. **Research Direction Classification**: Dynamically identify and categorize the main research directions based on the summaries. For each direction, list the papers that fall under it and briefly describe the focus of that direction.

3. **Comparative Analysis**:
   - **Commonalities**: Identify shared goals, techniques, or methodologies across the papers.
   - **Differences**: Highlight key differences in approaches, such as distinct algorithms, model architectures, or experimental setups.
   - **Strengths and Weaknesses**: Compare the advantages and limitations of the approaches, emphasizing which methods seem more promising and why.

4. **Summary of Findings**: Summarize the key insights from the analysis, including which research directions appear most impactful and any gaps or future directions suggested by the papers.

5. **Detailed Paper Summaries**: Include the full summary of each paper, clearly labeled with its title.

OUTPUT INSTRUCTIONS:
- Format the output in clear, human-readable Markdown.
- Use the headers provided above (Overview, Research Direction Classification, Comparative Analysis, Summary of Findings, Detailed Paper Summaries).
- Ensure the content is concise yet comprehensive, avoiding unnecessary repetition.

INPUT:
The following are summaries of {{ paper_number }} papers, each with its title:

{{ paper_content }}
"""

PAPER_VISUALIZE_PROMPT_EN = """I want to create a frontend web page for collecting and recommending research papers. 
Each paper should be displayed with its title, summary, and some key data visualizations. 
The page should be built using plain HTML, CSS, and JavaScript (the frontend trio), and include some interactive animations. 
Clicking on a paper should navigate to the corresponding paper's page. The overall style should be clean and minimalist.
"""


