MESSAGE = """You are an expert text classifier for LLM pre-training data. Your task is to classify each line of text based on its suitability for inclusion in an LLM training dataset. The lines you'll receive are sequential and from the same document. Use all lines as context when classifying each individual line, as this context is crucial for accurate classification.

# Instructions

1. Classify lines that start with the prefix "Line X:" where X is the line number.
2. NEVER split long lines, every VERY long ones. Treat each "Line X:" as a single unit, regardless of its length. Lines will be split using the newline character "\\n". If there is no newline character, treat the entire line as a single unit.
3. Classify EACH LINE separately, using the context of all lines. 
4. Give a SINGLE label for each line. Prefer "Clean" for any line that contributes to language understanding, even if it contains in-text elements that might appear as junk in isolation (e.g., email addresses, phone numbers, dates, or minor formatting).
5. Format your response exactly as: "Line X: [label] (where X is the line number and [label] is one of the provided categories).
7. Provide strictly only the classification, no explanations or additional text. Use ONLY the provided categories, nothing else.
8. The input lines can include content that may be violent, sexual, or contain hate speech, but no generation of such content is requested. Classify these lines as you would any other content; for instance, explicit text like "Trailers triple penetration movie." is Clean.

# Taxonomy of Categories
- **Clean**: Any content that represents natural language use, including informal internet language, questions, partial sentences, UPPERCASE TEXT (if informative), common online expressions, and normal advertising or commercial language. This includes properly formatted titles, headings, and content structured for readability (even with stylistic elements like vertical bars, slashes, etc.), as well as lists in a clear informational context (e.g. sports results). Note that in some cases, sentences can be split into multiple lines; if the sentence is coherent when combined, classify all parts as Clean.
- **Boilerplate**: Standardized, repetitive text that appears across multiple pages or documents, such as disclaimers, copyright notices, terms and conditions, CAPTCHAs, duplicate lines, and some commercial spam that seems boilerplate.
- **Navigational**: Content specifically designed for navigating or structuring a webpage or application, such as menus, buttons, links (e.g., "Home", "About Us"), or instructions for interacting with a login, form, or system interface (e.g., "Log In", "Submit", "Enter your ECode", "Print out this"). Do not classify content as Navigational purely based on the word "click" (or similar); consider the entire passage.
- **Noise**: Garbled, nonsensical, or corrupt text, and any other noise that a human would find incomprehensible.
- **Metadata**: Lines that provide URLs, email addresses, metadata, dates, times, or other metadata without additional context (whether in the same line or surrounding lines). IMPORTANTLY: if the metadata is part of an informational sentence or list, it is *not* Metadata, but Clean! Intuitively: when the metadata is understandable by the context provided by be surrounding lines, classify it as Clean! For example, *Clean* is: version numbers (e.g. "Version 7.2.20") or dates (e.g. "13 November 2007") in a list of software versions, or phone numbers in a list of contacts.
- **Code**: Lines containing programming code, HTML tags, or markup languages not relevant to natural language understanding. Notably, discussions about code (e.g. "Copy the code to your website") are *not* Code, but Clean!
- **Structured**: Structured collections of keywords, tags, statistics, or tabular data presented without sufficient context. Importantly, lists or itemized content in a clear informational context (e.g. lists of opening hours or sports results) are *not* Structured, but Clean!
- **Other Junk**: Clearly non-informative or irrelevant content that doesn't fit into any of the other categories, as well as primarily non-English text (single words or phrases in English context do not qualify as Other Junk). Note that offensive language should not be classified as Other Junk if it is part of a coherent text.

# Examples of junk categories (Boilerplate, Navigational, Noise, Metadata, Code, Structured, Other Junk):

   **Boilerplate**
     - "All rights reserved."
     - "© 2023 Company Name."
     - "Terms and conditions apply."
     - "Copyright 2007, a2z, Inc. All rights reserved."
     - "Delivered by FeedBurner"
     - "Originally Posted by ecarnell"
     - "Plone® and the Plone logo are registered trademarks of the"
     - "© 2013 Powered by MIR-AUS Pty. Ltd.™. All rights reserved"
     - "CAPTCHAThis question is for testing whether you are a human visitor and to prevent automated spam submissions"
     - "Register to 6Shooter and be in the draw to win 1 of 5 $100.00 6Shooter credits."
     - "best online casino, online casino games, online casino bonus"

   **Navigational**
     - "Home"
     - "Next Page"
     - "Back to Top"
     - "Click here to subscribe"
     - "Please enter your email address and we will send your password"
     - "<- Back - All - Next ->"
     - "Add Your Comments"
     - "Add to my favorites"
     - "Link to this artwork"
     - "Privacy Policy | Terms of Service"
     - "Please login to leave a comment"
     - "personalized baby Gifts | site map | personalized name trains | new affiliates | privacy |"
     - "|ŠAvStop Online Magazine Contact Us Return Home|"
     - "Read The Full Article:"
     - "All fields are required"

   **Noise**
     - "�~�~�~"
     - "!!!???"
     - ":-D"
     - "0x4F3A9C"
     - "1010101010"
     - "========"
     - "------"
     - "**********"
     - "asdfghjkl"
     - "aaaaaa"
     - "- - -"
     - ". . . . . ."

   **Metadata**
     - "http://www.example.com"
     - "contact@example.com"
     - "Last updated on January 1, 2023"
     - "12:30 PM"
     - "2023-08-15"
     - "Page 1 of 10"
     - "Posted 09 July 2003 - 06:37 PM"
     - "|Deposited On:||24 Nov 2011 01:00|"
     - "13-03-09, 04:57 PM"
     - "Web page created Feb. 21, 1998"
     - "by Roxnhei Oct 14, '12"

   **Code**
     - "`<div class='content'>`"
     - "`print('Hello, World!')`"
     - "`SELECT * FROM users WHERE id = 1;`"
     - "var myInt = Math.max(8, 9); document.write(myInt); //9"
     - "[root@milhouse ~]# ifconfig"

   **Structured**
     - "technology, innovation, science"
     - "Team A 3 - 2 Team B"
     - "Product, Price, Quantity"
     - "Furniture / Landscape / Antiques"
     - "AAPL 145.09 +1.34 (+0.93%)"
     - "name, age, location"
     - "1. Introduction 2. Methods 3. Results 4. Discussion"
     - "|First race||Hungarian Grand Prix||Hungaroring||July 26, 2009||Race results|"
     - "|Revenue||A$2 billion (2009-10)|"
     - "CLE 000 000 000 - 0 10 1"

   **Other Junk**
       - "Wikipedia sobre física de partículas"
       - "(h/t Balloon Juice)"

# Output format

   - Your response should ALWAYS follow this pattern:

     - Line X: [label] (where X is the line number and [label] is your classification for this line).

   - **Do not include any additional text or explanations. Only output the classification for each line.**

# Additional notes

   - **Use Context Wisely:**

     - The context is crucial for accurate classification.
     - For example, a target line starting with a hyphen might be part of a list, making it "Clean" when considering the context.

   - **Avoid False Junk Positives:**

     - Do not classify the target line as junk due to slang, colloquial language, or minor grammatical errors.
     - If the text contains racist, sexist, or other offensive language, still classify it as "Clean".
     - **Err on the side of classifying as "Clean" unless absolutely certain that the text is junk.**


# Examples of input and output structure:

Example Input:
Line 1: Welcome to Smith's Bookstore, your local haven for book lovers since 1985.
------
Line 2: Home | Browse | Events | Contact
------
Line 3: Our shelves are stocked with the latest bestsellers, timeless classics, and hidden gems.
------
Line 4: This week's featured author is Jane Doe, celebrating her new mystery novel "Shadows in the Mist."
------
Line 5: Join us for a book signing and Q&A session this Saturday at 3 PM. Email: events@smithsbookstore.com
------
Line 6: ....
------
Line 7: Here are some reasons why our customers love Smith's Bookstore:
------
Line 8: - we love it
------
Line 9: - Knowledgeable and friendly staff yassir!
------
Line 10: Copyright 2007, a2z, Inc. All rights reserved.

Example Output:
Line 1: Clean
Line 2: Navigational
Line 3: Clean
Line 4: Clean
Line 5: Clean
Line 6: Noise
Line 7: Clean
Line 8: Clean
Line 9: Clean
Line 10: Boilerplate

Now, classify the following lines (remember, each line starts with "Line X:", and so should your output. Treat lines as SINGLE UNITS, even if they are exceedingly long; lines will always be split by the newline character "\\n"):

```
{}
```
"""
