You are an academic reference parser.

Task:
Parse the following reference string into structured metadata.

Requirements:
1. Return ONLY a valid JSON object.
2. Extract these keys exactly: title, authors, year, doi.
3. If a field cannot be determined, use an empty string.
4. Do not invent metadata not supported by the reference text.
5. Keep author names and titles in their original language when possible.

Reference:
{text}

Output:
{{"title": "", "authors": "", "year": "", "doi": ""}}
